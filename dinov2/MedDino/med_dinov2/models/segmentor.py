import torch
import torch.nn as nn
from OrigDino.dinov2.hub.utils import CenterPadding
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from MedDino.med_dinov2.layers.segmentation import DecBase
from mmseg.ops import resize


class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, train_backbone=False, 
                 reshape_dec_oup=False, align_corners=False, interp_mode='bilinear', \
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        
        self._train_backbone = train_backbone
        self.backbone = backbone
        
        # params for the reshaping of the dec out
        self.reshape_dec_oup = reshape_dec_oup
        self.align_corners = align_corners
        self.interp_mode = interp_mode
        
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
            
    def reshape_dec_out(self, model_out):        
        # Interpolate to get pixel logits frmo patch logits
        pix_logits = resize(input=model_out,
                            scale_factor=self.backbone.patch_size,
                            mode=self.interp_mode,
                            align_corners=self.align_corners)
        # [B, N_class, H, W]
        return pix_logits
    
    def forward(self, x):
        feats = self.forward_backbone(x)
        out = self.decode_head(feats)
        
        if self.reshape_dec_oup:
            out = self.reshape_dec_out(out)
            
        assert x.shape[-2:] == out.shape[-2:], \
            f'input and output image shapes do not match, {x.shape[:-2]} =! {out.shape[:-2]}'
        return out
    


# class SegmentorLightning(nn.Module):
#     def __init__(self, backbone, decode_head, train_backbone=False, 
#                  reshape_dec_oup=False, align_corners=False, interp_mode='bilinear', \
#                  *args, **kwargs) -> None:

#         super().__init__(*args, **kwargs)
        
#         self._train_backbone = train_backbone
#         self.backbone = backbone
        
#         # params for the reshaping of the dec out
#         self.reshape_dec_oup = reshape_dec_oup
#         self.align_corners = align_corners
#         self.interp_mode = interp_mode
        
#         self.decode_head = decode_head
#         if isinstance(decode_head, BaseDecodeHead):
#             n_concat = len(decode_head.in_index)
#         elif isinstance(decode_head, DecBase):
#             n_concat = decode_head.n_concat
#         else:
#             raise Exception(f'Unknown decode head type: {type(decode_head)}')
#         self.n_concat_bb = n_concat
    
#     @property    
#     def train_backbone(self):
#         return self._train_backbone
#     @train_backbone.setter
#     def train_backbone(self, new_val):
#         if new_val == True:
#             self.backbone.train()
#             self._train_backbone = new_val
#         elif new_val==False:
#             self.backbone.eval()
#             self._train_backbone = new_val
#         else:
#             raise Exception(f'has to be a boolean type but got {type(new_val)}')
            
#     def train(self, mode: bool = True):
#         if self.train_backbone:
#             return super().train(mode)
#         else:
#             super().train(mode)
#             self.backbone.eval()
#             return self
        
#     def forward_backbone(self, x):
#         if self.train_backbone:
#             return self.backbone.get_intermediate_layers(x, n=self.n_concat_bb, reshape=True)
#         else:
#             with torch.no_grad():
#                 return self.backbone.get_intermediate_layers(x, n=self.n_concat_bb, reshape=True)
            
#     def reshape_dec_out(self, model_out):        
#         # Interpolate to get pixel logits frmo patch logits
#         pix_logits = resize(input=model_out,
#                             scale_factor=self.backbone.patch_size,
#                             mode=self.interp_mode,
#                             align_corners=self.align_corners)
#         # [B, N_class, H, W]
#         return pix_logits
    
#     def forward(self, x):
#         feats = self.forward_backbone(x)
#         out = self.decode_head(feats)
        
#         if self.reshape_dec_oup:
#             out = self.reshape_dec_out(out)
            
#         assert x.shape[-2:] == out.shape[-2:], \
#             f'input and output image shapes do not match, {x.shape[:-2]} =! {out.shape[:-2]}'
#         return out
