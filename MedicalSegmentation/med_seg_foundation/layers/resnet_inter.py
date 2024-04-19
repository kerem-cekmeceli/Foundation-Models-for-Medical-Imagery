import torch
from torch import nn
from torchvision.models import ResNet

class ResnetInter(nn.Module):
    def __init__(self, resnet:ResNet, n_out:int) -> None:
        super().__init__()
        self.resnet = resnet
        
        self.n_out = n_out
        
    
    def get_int_res_bottleneck(self, layer_idx, blk_idxm, n_out):
        pass
        
    def get_int_res_basic(self, layer_idx, blk_idxm, n_out):
        layer = getattr(self, f'layer{layer_idx}', None) 
        

    def get_int_res(self):
        pass
        
        