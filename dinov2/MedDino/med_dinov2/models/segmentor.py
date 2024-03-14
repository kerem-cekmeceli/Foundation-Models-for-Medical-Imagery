
import torch
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from MedDino.med_dinov2.layers.segmentation import DecBase
from mmseg.ops import resize
from MedDino.prep_model import get_dino_backbone
from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead
# from OrigDino.dinov2.models.vision_transformer import DinoVisionTransformer
from MedDino.med_dinov2.layers.backbone import BackBoneBase, DinoBackBone
from typing import Union, Optional, Sequence, Callable, Any


class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, train_backbone=False, 
                 reshape_dec_oup=False, align_corners=False, \
                 ) -> None:

        super().__init__()
        
        if isinstance(backbone, dict):
            # config is given
            bb_name = backbone['name']
            bb_params = backbone['params']
            
            if bb_name == DinoBackBone.__name__:
                self.backbone = DinoBackBone(**bb_params)
            
            else:
                ValueError(f"Backbone {bb_name} is not supported from config.")
                
        else:
            assert isinstance(backbone, BackBoneBase)
            # Model is given
            self.backbone = backbone
        
        assert isinstance(train_backbone, bool)    
        self.backbone.train_backbone = train_backbone
            
            
        if isinstance(decode_head, dict):
            # config is given
            dec_head_name = decode_head['name']
            dec_head_params = decode_head['params']
            
            if dec_head_name == ConvHeadLinear.__name__:
                self.decode_head = ConvHeadLinear(**dec_head_params)
                
            elif dec_head_name == FCNHead.__name__:
                self.decode_head = FCNHead(**dec_head_params)
                
            elif dec_head_name == PSPHead.__name__:
                self.decode_head = PSPHead(**dec_head_params)
                
            elif dec_head_name == DAHead.__name__:
                self.decode_head = DAHead(**dec_head_params)
                
            elif dec_head_name == SegformerHead.__name__:
                self.decode_head = SegformerHead(**dec_head_params)
                
            elif dec_head_name == ResNetHead.__name__:
                self.decode_head = ResNetHead(**dec_head_params)
                
            elif dec_head_name == UNetHead.__name__:
                self.decode_head = UNetHead(**dec_head_params)
                
            else:
                ValueError(f"Decode head {dec_head_name} is not supported from config.")
        else:
            # Model is given
            self.decode_head = decode_head
    
        # Nb inputs from the decode head
        if isinstance(self.decode_head, BaseDecodeHead) or isinstance(self.decode_head, DecBase):
            n_concat = len(self.decode_head.in_index)
        else:
            raise Exception(f'Unknown decode head type: {type(decode_head)}')
        # self.n_concat_bb = n_concat
        
        
         # params for the reshaping of the dec out
        self.reshape_dec_oup = reshape_dec_oup
        self.align_corners = align_corners
        
            
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
        feats = self.backbone(x)
        out = self.decode_head(feats)
        
        if self.decode_head.__class__.__name__ == 'DAHead':
            out = out[0]
        
        if self.reshape_dec_oup:
            out = self.reshape_dec_out(out, x.shape[-2:])
            
        assert x.shape[-2:] == out.shape[-2:], \
            f'input and output image shapes do not match, {x.shape[:-2]} =! {out.shape[:-2]}'
        return out


