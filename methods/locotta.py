"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

logger = logging.getLogger(__name__)

def update_ema(ema, new_data, alpha=0.9):
    if ema is None:
        return new_data
    else:
        return alpha * ema + (1 - alpha) * new_data

def update_ema_list(ema, new_data, alpha=0.9):
    if ema is None:
        return new_data
    else:
        return [alpha * e + (1 - alpha) * n for e, n in zip(ema, new_data)]

class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, batch):
        # batch = np.asarray(batch)
        batch_size = batch.size(0)
        if batch_size == 0:
            return

        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0, unbiased=True)

        new_n = self.n + batch_size
        delta = batch_mean - self.mean

        self.mean += delta * batch_size / new_n
        self.M2 += batch_var * batch_size + delta ** 2 * self.n * batch_size / new_n
        self.n = new_n

    def finalize(self):
        if self.n < 2:
            return float('nan'), float('nan')
        variance = self.M2 / (self.n - 1)
        stddev = torch.sqrt(variance)
        return self.mean, stddev
    
    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


@ADAPTATION_REGISTRY.register()
class LOCOTTA(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.num_classes = num_classes

        # Collect trainable parameters with names for tent updates
        self.init_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.init_params[name] = param.clone().detach()


        self.num_samples_update_1 = 0  # number of samples after first filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after second filtering, exclude both unreliable and redundant samples
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)
        self.tradeoff = cfg.LOCOTTA.TRADEOFF
        self.l_w = cfg.LOCOTTA.L_W

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)
        self.softmax_entropy = Entropy()
        
    def loss_calculation(self, x, filter_ids_1=None, filter_ids_2=None):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs)

        if filter_ids_1 is None:
            entropys1 = self.softmax_entropy(outputs).clone().detach()
            filter_ids_1 = torch.where(entropys1 < self.e_margin)
        entropys = entropys[filter_ids_1]
        
        if self.current_model_probs is not None:
            if filter_ids_2 is None:
                cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.current_model_probs = updated_probs
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples

        perform_update = True if len(entropys) > 0 else False
        # if not perform_update:
            # logger.info(f"No samples to update.")

        if perform_update:
            loss = entropys.mean(0) 

        return outputs, loss, perform_update, filter_ids_1, filter_ids_2

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, perform_update = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, perform_update, id1, id2= self.loss_calculation(x)

            reg_loss = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    cosine_weight = (F.cosine_similarity(param.flatten().clone().detach(), self.init_params[name].flatten(), dim=0))
                    reg_loss +=  torch.sum((param - self.init_params[name]) ** 2).mul(cosine_weight)
                    
            if perform_update:
                if self.l_w > 1.0:
                    loss = loss * self.l_w

                loss += self.tradeoff * reg_loss
                loss.backward()
                self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def configure_conv_optim(self):
        """Configure optimizer for tent updates."""
        # collect affine parameters from nn.Conv2d layers
        conv_params, _ = self.collect_params(layers=(nn.Conv2d))
        # configure optimizer
        self.conv_optimizer = torch.optim.SGD(conv_params, lr=self.cfg.OPTIM.LR)

    def collect_params(self, layers=(nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, layers):
                clist = ['weight', 'bias']
                for np, p in m.named_parameters():
                    if p.requires_grad:
                        if np in clist:#['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        # if self.freeze_last:
        layers = ["layer4", "block3", "stage_3"] #   
        # else:
        #     layers = []
        for nm, m in self.model.named_modules():    
            if isinstance(m, nn.BatchNorm2d):
                if any([c in nm for c in layers]):
                    m.train()
                    m.requires_grad_(False)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False # True
                else:
                    m.train()
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                # m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                if any([c in nm for c in layers]):
                    m.eval()
                    m.requires_grad_(False)
                else:
                    m.train()
                    m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if any([c in nm for c in layers]):
                    m.eval()
                    m.requires_grad_(False)
                else:
                    m.train()
                    m.requires_grad_(True)

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
