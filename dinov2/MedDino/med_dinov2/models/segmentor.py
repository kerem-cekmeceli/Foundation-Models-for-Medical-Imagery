import torch
import torch.nn as nn
from OrigDino.dinov2.hub.utils import CenterPadding
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from MedDino.med_dinov2.layers.segmentation import DecBase
from mmseg.ops import resize
import lightning as L
from typing import Union, Optional


class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, train_backbone=False, 
                 reshape_dec_oup=False, align_corners=False, \
                 ) -> None:

        super().__init__()
        
        self._train_backbone = train_backbone
        self.backbone = backbone
        
        # params for the reshaping of the dec out
        self.reshape_dec_oup = reshape_dec_oup
        self.align_corners = align_corners
        
        self.decode_head = decode_head
        if isinstance(decode_head, BaseDecodeHead):
            n_concat = len(decode_head.in_index)
        elif isinstance(decode_head, DecBase):
            n_concat = decode_head.nb_inputs
        else:
            raise Exception(f'Unknown decode head type: {type(decode_head)}')
        self.n_concat_bb = n_concat
    
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
            
    def reshape_dec_out(self, model_out, reshaped_size): 
        up = reshaped_size[0] > model_out.shape[-2] and reshaped_size[1] > model_out.shape[-1]
        # Interpolate to get pixel logits frmo patch logits
        pix_logits = resize(input=model_out,
                            size=reshaped_size,
                            mode="bilinear" if up >= 1 else "area",
                            align_corners=self.align_corners if up >= 1 else None)
        # [B, N_class, H, W]
        return pix_logits
    
    def forward(self, x):
        feats = self.forward_backbone(x)
        out = self.decode_head(feats)
        
        if self.reshape_dec_oup:
            out = self.reshape_dec_out(out, x.shape[-2:])
            
        assert x.shape[-2:] == out.shape[-2:], \
            f'input and output image shapes do not match, {x.shape[:-2]} =! {out.shape[:-2]}'
        return out


class LitBaseModule(L.LightningModule):
    def __init__(self,
                 optimizer_config:dict,
                 schedulers_config:Union[list, dict]) -> None:
        
        super().__init__()  
        self.optimizer_config = optimizer_config
        
        
    def _get_optimizer(self, optimizer_config):
        optimizer_name = optimizer_config['name']
        optimizer_params = optimizer_config['params']

        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **optimizer_params)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

        return optimizer

class LitSegmentor(L.LightningModule):
    def __init__(self,
                 backbone, 
                 decode_head, 
                 train_backbone=False, 
                 reshape_dec_oup=False, 
                 align_corners=False,
                 ) -> None:
        super().__init__()
        self.segmentor = Segmentor(backbone=backbone,
                                   decode_head=decode_head,
                                   train_backbone=train_backbone,
                                   reshape_dec_oup=reshape_dec_oup,
                                   align_corners=align_corners)
        
    def forward(self, x):
        return self.segmentor(x)
    
    
    
    def configure_optimizers(self):
        optm = None
        scheduler = None
        return [optm], [scheduler]
    
    
    
    
    
    
        
