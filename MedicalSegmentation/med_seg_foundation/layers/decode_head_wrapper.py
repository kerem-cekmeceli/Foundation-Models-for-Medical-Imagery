# import torch
from torch import nn 

from layers.backbone_wrapper import BackBoneBase
from layers import segmentation as seg   #import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models import decode_heads as seg_mmcv # FCNHead, PSPHead, DAHead, SegformerHead

from abc import abstractmethod

class DecHeadBase(nn.Module):
    def __init__(self, 
                 backbone:BackBoneBase,
                 cfg:dict,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        nb_ins = backbone.nb_outs
        in_feat_chs = backbone.out_feat_channels
        
        self.cfg = cfg
        self.cfg['in_channels'] = [in_feat_chs]*nb_ins
        
        self.decoder = self._get_dec_from_cfg()
        
        #@TODO freeze all but the cls_seg and dropout (if exists)
    
    @abstractmethod    
    def _get_dec_from_cfg(self, cfg:dict):
        pass
    
    def forward(self, feats):
        return self.decoder(feats)
    
    #Freeze backbone setter to only enable grad for cls_seg (and dropout if exists) freeze the rest
    # Also redifine train method accordingly
        

class ConvHeadLinear(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['out_upsample_fac'] = backbone.hw_shrink_fac
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg.ConvHeadLinear(**self.cfg)
    
class ResNetHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg.ResNetHead(**self.cfg)
    
class UNetHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg.UNetHead(**self.cfg)
    
class FCNHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['channels'] = backbone.out_feat_channels
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg_mmcv.FCNHead(**self.cfg)
    
class PSPHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['channels'] = backbone.out_feat_channels  # Conv channels
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg_mmcv.PSPHead(**self.cfg)
    
class DAHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['channels'] = backbone.out_feat_channels  # Conv channels
        cfg['pam_channels'] = backbone.out_feat_channels  # Conv channels
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg_mmcv.DAHead(**self.cfg)
    
    def forward(self, feats):
        return self.decoder(feats)[0]
    
class SegformerHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['channels'] = backbone.out_feat_channels  # Conv channels
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg_mmcv.SegformerHead(**self.cfg)
    
class SAMdecHead(DecHeadBase):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['bb_embedding_hw_shrink_fac'] = backbone.hw_shrink_fac
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg.SAMdecHead(**self.cfg)
    
implemented_dec_heads = [ConvHeadLinear.__name__,
                         ResNetHead.__name__,
                         UNetHead.__name__,
                         FCNHead.__name__, 
                         PSPHead.__name__,
                         DAHead.__name__,
                         SegformerHead.__name__,
                         SAMdecHead.__name__]    
    