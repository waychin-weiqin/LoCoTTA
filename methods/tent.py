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


@ADAPTATION_REGISTRY.register()
class Tent(TTAMethod):
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

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        ########################################################################
        # norm = torch.norm(outputs, dim=1, p=2).unsqueeze(dim=1)
        # if self.n_margin is None:
        #     self.n_margin = 5 #math.ceil(norm.mean().item())
        #     print(f"Original norm: {norm.mean().item():.2f}, Set n_margin to {self.n_margin}")
        
        # outputs = outputs / norm * self.n_margin #torch.clip(norm, max=self.n_margin)
        ########################################################################

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
            loss.backward() 

            ##########################################################################################
            # # Auto-RGN from Surgical Fine Tuning
            # grad_weight = {}
            # # Get gradients, calculate norm of gradient and param and take the ratio 
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = torch.norm(param.grad, p=2)
            #         param_norm = torch.norm(param, p=2)
            #         ratio = grad_norm / param_norm
            #         grad_weight[name] = ratio.item()

            # # Normalize grad_weight to [0, 1]
            # max_ratio = max(grad_weight.values())
            # min_ratio = min(grad_weight.values())
            # # mean_ratio = sum(grad_weight.values()) / len(grad_weight)
            # mean_ratio = sorted(grad_weight.values())[len(grad_weight) // 2]
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         if grad_weight[name] > mean_ratio:
            #             grad_weight[name] = 0
            #         else:
            #             grad_weight[name] = (grad_weight[name] - min_ratio) / (max_ratio - min_ratio)
            
            # # print(max(grad_weight.values()), min(grad_weight.values()))
            # # Apply grad_weight to gradient
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         param.grad = param.grad * grad_weight[name]
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
        # layers = ["block3", "stage_3"]
        for nm, m in self.model.named_modules():
            # if any([c in nm for c in layers]):
                # continue
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
