# import torch
from torch import nn 

from layers.backbone_wrapper import BackBoneBase
from layers import segmentation as seg   #import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models import decode_heads as seg_mmcv # FCNHead, PSPHead, DAHead, SegformerHead
# from mmdet.models import dense_heads as seg_mmdet

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
        if isinstance(in_feat_chs, int):
            self.cfg['in_channels'] = [in_feat_chs]*nb_ins
        else:
            nb_ins == len(in_feat_chs)
            self.cfg['in_channels'] = in_feat_chs
            
        self.num_classes = cfg['num_classes']
        
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
        # cfg['out_upsample_fac'] = backbone.hw_shrink_fac
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
        cfg['channels'] = 120 # 384  # Conv channels
        cfg['pam_channels'] = 120 # 384  # Conv channels
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
        cfg['patch_sz'] = backbone.hw_shrink_fac
        cfg['image_pe_size'] = backbone.target_size
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg.SAMdecHead(**self.cfg)

    
class HSAMdecHead(SAMdecHead):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['nb_patches']=backbone.nb_patches
        # cfg['patch_sz'] = backbone.hw_shrink_fac
        # cfg['image_pe_size'] = backbone.target_size
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        return seg.HSAMdecHead(**self.cfg)
    
    
class HQSAMdecHead(SAMdecHead):
    def __init__(self, backbone: BackBoneBase, cfg: dict) -> None:
        assert backbone.nb_outs==2
        assert backbone.last_out_first
        super().__init__(backbone, cfg)
        
    def _get_dec_from_cfg(self):
        [last_out_ch, first_out_ch] = self.cfg.pop('in_channels')
        self.cfg['last_out_ch']=last_out_ch
        self.cfg['first_out_ch']=first_out_ch
        return seg.HQSAMdecHead(**self.cfg)
    

class HQHSAMdecHead(SAMdecHead):
    def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
        cfg['nb_patches']=backbone.nb_patches
        assert backbone.nb_outs==2
        assert backbone.last_out_first
        super().__init__(backbone, cfg, *args, **kwargs)
        
    def _get_dec_from_cfg(self):
        [last_out_ch, first_out_ch] = self.cfg.pop('in_channels')
        self.cfg['last_out_ch']=last_out_ch
        self.cfg['first_out_ch']=first_out_ch
        return seg.HQHSAMdecHead(**self.cfg)
    
    
# class Mask2FormerHead(DecHeadBase):
#     def __init__(self, backbone: BackBoneBase, cfg: dict, *args, **kwargs) -> None:
#         # cfg[''] = None
#         super().__init__(backbone, cfg, *args, **kwargs)
        
#     def _get_dec_from_cfg(self):
#         num_cls = self.cfg.pop('num_classes')
#         self.cfg['num_things_classes'] = num_cls
#         self.cfg['num_stuff_classes'] = 0
#         return seg_mmdet.Mask2FormerHead(**self.cfg)
    
implemented_dec_heads = [ConvHeadLinear.__name__,
                         ResNetHead.__name__,
                         UNetHead.__name__,
                         FCNHead.__name__, 
                         PSPHead.__name__,
                         DAHead.__name__,
                         SegformerHead.__name__,
                         SAMdecHead.__name__,
                         HSAMdecHead.__name__,
                         HQSAMdecHead.__name__,
                         HQHSAMdecHead.__name__,]    
    