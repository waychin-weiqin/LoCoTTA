"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

import math

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
            

@ADAPTATION_REGISTRY.register()
class RGN(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # setup loss function
        self.softmax_entropy = Entropy()
        self.output_norm = []
        self.entropys = []
        self.errs = []
        self.n_margin = None

        self.num_samples_update_1 = 0  # number of samples after first filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after second filtering, exclude both unreliable and redundant samples
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)
        self.current_model_probs = None

    def loss_calculation_eta(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        outputs_ = outputs.clone().detach()
        entropys = self.softmax_entropy(outputs)
        
        entropys1 = self.softmax_entropy(outputs_).clone().detach()
        filter_ids_1 = torch.where(entropys1 < self.e_margin)
        entropys = entropys[filter_ids_1]
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs_[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, outputs_[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs_[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.current_model_probs = updated_probs
        entropys = entropys.mul(coeff) 
        loss = entropys.mean(0)
        return outputs, loss

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs)
        loss = entropys.mean(0)
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            outputs, loss = self.loss_calculation_eta(x)
             
            loss.backward() 

            ##########################################################################################
            # Auto-RGN from Surgical Fine Tuning
            grad_weight = {}
            # Get gradients, calculate norm of gradient and param and take the ratio 
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad, p=2)
                    param_norm = torch.norm(param, p=2)
                    ratio = grad_norm / param_norm
                    grad_weight[name] = ratio.item()

            # Normalize grad_weight to [0, 1]
            max_ratio = max(grad_weight.values())
            min_ratio = min(grad_weight.values())
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_weight[name] = (grad_weight[name] - min_ratio) / (max_ratio - min_ratio + 1e-8)
            
            # Apply grad_weight to gradient
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad = param.grad * grad_weight[name]
            ##########################################################################################

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.output_norm.append(torch.norm(outputs, dim=1, p=2).mean().item())
            self.entropys.append(loss.item())
        
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
