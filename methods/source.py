from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY

import torch
import torch.nn as nn 
from utils.losses import Entropy

@ADAPTATION_REGISTRY.register()
class Source(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.output_norm = []
        self.entropys = []
        self.errs = []
        self.softmax_entropy = Entropy()
        
    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        self.output_norm.append(torch.norm(outputs, dim=1, p=2).mean().item())
        self.entropys.append(self.softmax_entropy(outputs).mean().item())
    
        return outputs

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    # def configure_model(self):
    #     self.model.eval()
    #     self.model.requires_grad_(False)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        # """Configure model for use with tent."""
        # # train mode, because tent optimizes the model to minimize entropy
        # # self.model.train()
        # self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # # disable grad, to (re-)enable only what tent updates
        # self.model.requires_grad_(False)
        # # configure norm for tent updates: enable grad + force batch statisics
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
