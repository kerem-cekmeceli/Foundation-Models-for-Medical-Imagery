
# import torch
import torch.nn as nn

# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from MedDino.med_dinov2.layers.segmentation import DecBase
from abc import abstractmethod
from mmseg.ops import resize
# from MedDino.prep_model import get_dino_backbone
# from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
# from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead
# from OrigDino.dinov2.models.vision_transformer import DinoVisionTransformer
from layers.decode_head_wrapper import implemented_dec_heads, DecHeadBase, ConvHeadLinear,\
    ResNetHead, UNetHead, FCNHead, PSPHead, DAHead, SegformerHead, SAMdecHead, HSAMdecHead, HQSAMdecHead, HQHSAMdecHead
from layers.backbone_wrapper import implemented_backbones, BackBoneBase, DinoBackBone, SamBackBone, ResNetBackBone,\
    LadderBackbone, DinoReinBackbone, SamReinBackBone, MAEBackbone
from typing import Union, Optional, Sequence, Callable, Any

from models.benchmarks import implemented_models, UNet, SwinTransformerSys

class SegmentorBase(nn.Module):
    def __init__(self, reshape_dec_oup=False, align_corners=False, target_inp_shape=None,) -> None:
        super().__init__()
        
        # Target input shape
        if target_inp_shape is not None:
            if isinstance(target_inp_shape, int):
                target_inp_shape = (target_inp_shape, target_inp_shape)
            else:
                assert isinstance(target_inp_shape, tuple) and len(target_inp_shape)==2
        self.target_inp_shape = target_inp_shape
        
        # params for the reshaping 
        self.reshape_dec_oup = reshape_dec_oup
        self.align_corners = align_corners
        
    def reshape_tensor_2d(self, input, reshaped_size): 
        
        up = reshaped_size[0] > input.shape[-2] and reshaped_size[1] > input.shape[-1]
        # Interpolate to get pixel logits frmo patch logits
        pix_logits = resize(input=input,
                            size=reshaped_size,
                            mode="bilinear" if up >= 1 else "area",
                            align_corners=self.align_corners if up >= 1 else None)
        # [B, N_class, H, W]
        return pix_logits
    
    @abstractmethod
    def get_masks(self, x):
        pass
    
    def forward(self, x):
        orig_shape = x.shape[-2:]
        
        if self.target_inp_shape is not None:
            x = self.reshape_tensor_2d(o, self.target_inp_shape)
        
        out = self.get_masks(x)
        
        if not isinstance(out, list):
            out = [out]  
      
        for i, o in enumerate(out):
            if self.reshape_dec_oup:
                out[i] = self.reshape_tensor_2d(o, orig_shape)
            assert x.shape[-2:] == out[i].shape[-2:], \
                f'input and output image shapes do not match, {x.shape[:-2]} =! {o.shape[:-2]}'
                
        if len(out)==1:
            out = out[0]
        return out
        

class Segmentor(SegmentorBase):
    def __init__(self, backbone, decode_head,
                 reshape_dec_oup=False, align_corners=False, target_inp_shape=None,
                 ) -> None:
        super().__init__(reshape_dec_oup=reshape_dec_oup, align_corners=align_corners, target_inp_shape=target_inp_shape)
        
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
        
    def get_masks(self, x):
        feats = self.backbone(x)
        out = self.decode_head(feats)
        return out
    
class SegmentorModel(SegmentorBase):
    def __init__(self, model,
                 reshape_dec_oup=False, align_corners=False, target_inp_shape=None,
                 ) -> None:
        super().__init__(reshape_dec_oup=reshape_dec_oup, align_corners=align_corners, target_inp_shape=target_inp_shape)
    
        if isinstance(model, dict):
            # config is given
            model_name = model['name']
            model_params = model['params']
            
            if model_name not in implemented_models:
                ValueError(f"Model {model_name} is not supported from config.")
            
            self.model = globals()[model_name](**model_params)
            
        else:
            # Model is given
            self.model = model
            
    def get_masks(self, x):
        return self.model(x)
 
 
    
implemented_segmentors = [Segmentor.__name__, SegmentorModel.__name__]

    
