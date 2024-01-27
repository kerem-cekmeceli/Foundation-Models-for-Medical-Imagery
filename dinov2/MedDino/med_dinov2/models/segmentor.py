import torch
import torch.nn as nn

class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, train_backbone=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._train_backbone = train_backbone
        self.backbone = backbone
        
        self.decode_head = decode_head
    
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
            return self.backbone(x)
        else:
            with torch.no_grad():
                return self.backbone(x)
    
    def forward(self, x):
        feats = self.forward_backbone(x)
        #@ TODO resize and concat inputs 
        out = self.decode_head(feats)
        return 
    
    