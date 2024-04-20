
# import torch
import torch.nn as nn

# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from MedDino.med_dinov2.layers.segmentation import DecBase
from mmseg.ops import resize
# from MedDino.prep_model import get_dino_backbone
# from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
# from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead
# from OrigDino.dinov2.models.vision_transformer import DinoVisionTransformer
from layers.decode_head_wrapper import implemented_dec_heads, DecHeadBase, ConvHeadLinear,\
    ResNetHead, UNetHead, FCNHead, PSPHead, DAHead, SegformerHead, SAMdecHead
from layers.backbone_wrapper import implemented_backbones, BackBoneBase, DinoBackBone, SamBackBone, ResNetBackBone, LadderBackbone
from typing import Union, Optional, Sequence, Callable, Any


class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head,
                 reshape_dec_oup=False, align_corners=False, \
                 ) -> None:

        super().__init__()
        
        if isinstance(backbone, dict):
            # config is given
            bb_name = backbone['name']
            bb_params = backbone['params']
            
            if bb_name not in implemented_backbones:
                ValueError(f"Backbone {bb_name} is not supported from config.")
            
            self.backbone = globals()[bb_name](**bb_params)
            
        else:
            assert isinstance(backbone, BackBoneBase)
            # Model is given
            self.backbone = backbone
            
            
        if isinstance(decode_head, dict):
            # config is given
            dec_head_name = decode_head['name']
            dec_head_params = decode_head['params']
            
            if dec_head_name not in implemented_dec_heads:
                ValueError(f"Decode head {dec_head_name} is not supported from config.")
            
            self.decode_head = globals()[dec_head_name](**dict(backbone=self.backbone,
                                                               cfg=dec_head_params))
            
        else:
            assert isinstance(decode_head, DecHeadBase)
            # Model is given
            self.decode_head = decode_head
        
        
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
        
        if self.reshape_dec_oup:
            out = self.reshape_dec_out(out, x.shape[-2:])
            
        assert x.shape[-2:] == out.shape[-2:], \
            f'input and output image shapes do not match, {x.shape[:-2]} =! {out.shape[:-2]}'
        return out


