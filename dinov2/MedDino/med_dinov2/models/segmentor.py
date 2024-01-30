import torch
import torch.nn as nn
from OrigDino.dinov2.hub.utils import CenterPadding

class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, train_backbone=False, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._train_backbone = train_backbone
        self.backbone = backbone
        
        # Done in pre-processing transforms
        # # Takes in 2 args, module and input
        # self.register_forward_pre_hook(lambda _, x: CenterPadding(backbone.patch_size)(x[0]))
        
        self.decode_head = decode_head
        self.n_concat_bb = decode_head.n_concat
        
    
    @property    
    def train_backbone(self):
        return self._train_backbone
    @train_backbone.setter
    def train_backbone(self, new_val):
        if new_val == True:
            self.backbone.train()
            self._train_backbone = new_val
        elif new_val==False:
            self.backbone.eval()
            self._train_backbone = new_val
        else:
            raise Exception(f'has to be a boolean type but got {type(new_val)}')
            
    def train(self, mode: bool = True):
        if self.train_backbone:
            return super().train(mode)
        else:
            super().train(mode)
            self.backbone.eval()
            return self
        
    def forward_backbone(self, x):
        if self.train_backbone:
            return self.backbone.get_intermediate_layers(x, n=self.n_concat_bb, reshape=True)
        else:
            with torch.no_grad():
                return self.backbone.get_intermediate_layers(x, n=self.n_concat_bb, reshape=True)
    
    def forward(self, x):
        feats = self.forward_backbone(x)
        out = self.decode_head(feats)
        return out
    
    