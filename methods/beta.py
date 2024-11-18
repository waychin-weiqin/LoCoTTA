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

class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, BN_module):
        super(CustomBatchNorm2d, self).__init__(BN_module.num_features, eps=BN_module.eps, momentum=BN_module.momentum, affine=BN_module.affine, track_running_stats=BN_module.track_running_stats)
        self.current_domain = 0  # Default to the first domain
        
        self.num_features = BN_module.num_features

        # Copy the parameters from the original BN module
        self.weight.data = BN_module.weight.data.clone()
        self.bias.data = BN_module.bias.data.clone()
        self.running_mean = BN_module.running_mean.clone()
        self.running_var = BN_module.running_var.clone()

    def forward(self, x):
        
        # if self.running_mean is None and self.running_var is None: # first time, replace with batch statistics
        self.running_mean = x.mean([0, 2, 3]).clone().detach()
        self.running_var = x.var([0, 2, 3], unbiased=False).clone().detach()

        # self.running_mean = torch.randn_like(self.running_mean)
        # self.running_var = torch.randn_like(self.running_var)

        x1 =F.batch_norm(
            x, 
            self.running_mean, 
            self.running_var, 
            self.weight, 
            self.bias, 
            True, 
            0, 
            self.eps
        ) 

        x2 =F.batch_norm(
            x, 
            self.running_mean, 
            self.running_var, 
            self.weight, 
            self.bias, 
            False, 
            self.momentum, 
            self.eps
        )  

        return (x1 + x2) / 2

        # return F.batch_norm(
        #     x, 
        #     self.running_mean, 
        #     self.running_var, 
        #     self.weight, 
        #     self.bias, 
        #     True, 
        #     1, 
        #     self.eps
        # )
    
    # def forward(self, x):
        
    #     # if self.running_mean is None and self.running_var is None: # first time, replace with batch statistics
    #     #     self.running_mean = x.mean([0, 2, 3]).clone().detach()
    #     #     self.running_var = x.var([0, 2, 3], unbiased=False).clone().detach()
    #     # else:
    #     #     self.update_bn_stats(x)

    #     self.running_mean = x.mean([0, 2, 3]).detach()
    #     self.running_var = x.var([0, 2, 3], unbiased=False).detach()

    #     # Manual batch norm
    #     x = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
    #     x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    #     return x
    
    def update_bn_stats(self, x):
        # Calculate batch statistics
        mean = x.mean([0, 2, 3])
        var = x.var([0, 2, 3], unbiased=False)
        n = x.numel() / x.size(1)
        # Update running statistics
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / (n - 1)

        self.running_mean = self.running_mean.detach()
        self.running_var = self.running_var.detach()


    def __stats_len__(self):
        return len(self.memory_mean)



def replace_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            # num_features = module.num_features
            new_bn = CustomBatchNorm2d(module)
            # new_bn.copy_params_from(module)
            # new_bn.allow_update()
            setattr(model, name, new_bn)

        elif len(list(module.children())) > 0:
            replace_bn(module)



@ADAPTATION_REGISTRY.register()
class Beta(TTAMethod):
    """Beta adapts a model by entropy minimization during testing."""
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # setup loss function
        self.softmax_entropy = Entropy()
        self.output_norm = []
        self.entropys = []
        self.errs = []

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
        # if self.mixed_precision and self.device == "cuda":
        #     with torch.cuda.amp.autocast():
        #         outputs, loss = self.loss_calculation(x)
        #     self.scaler.scale(loss).backward()
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        #     self.optimizer.zero_grad()
        # else:
        outputs, loss = self.loss_calculation(x)
        loss.backward() 

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.output_norm.append(torch.norm(outputs, dim=1, p=2).mean().item())
        self.entropys.append(loss.item())
        self.model.zero_grad()
        
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
            # if nm.find("layer3") > -1:
            #     print(f"Skipping {nm}")
            #     continue
            # if nm.find("layer4") > -1:
            #     print(f"Skipping {nm}")
            #     continue
            # if nm.find("stage_3") > -1:
            #     continue 
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    print(nm)
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
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

        # # # configure norm for tent updates: enable grad + force batch statisics
        # for nm, m in self.model.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.requires_grad_(True)
        #         # force use of batch stats in train and eval modes
        #         m.track_running_stats = False
        #         m.running_mean = None
        #         m.running_var = None
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
        #         m.requires_grad_(True)
        #     elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        #         m.requires_grad_(True)

        replace_bn(self.model)
        # # Replace batch normalization layers in ResNet50
        # replace_bn_layers_with_trained_params(self.model, num_domains=0)
