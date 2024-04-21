import torch
from torch import nn
from torchvision.models import ResNet

class ResnetInter(nn.Module):
    def __init__(self, resnet:ResNet, n_out:int) -> None:
        super().__init__()
        self.resnet = resnet
        
        self.n_out = n_out
        
    
    def get_int_res_bottleneck(self, x, layer_idx, blk_idx, n_out):
        layer = getattr(self, f'layer{layer_idx}', None) 
        assert blk_idx<len(layer)
        
        # identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # return out
        
        
    def get_int_res_basic(self, layer_idx, blk_idx, n_out):
        layer = getattr(self, f'layer{layer_idx}', None) 
        

    def get_int_res(self):
        pass
        
        