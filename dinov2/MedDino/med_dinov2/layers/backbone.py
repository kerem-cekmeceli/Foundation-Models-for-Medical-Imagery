import torch 
from torch import nn
# import lightning as L

from MedDino.prep_model import get_dino_backbone

from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead

from abc import abstractmethod
from typing import Union, Optional, Sequence, Callable, Any

class BackBoneBase(nn.Module):
    def __init__(self, 
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)
        
        assert (bb_model is None) ^ (cfg is None), "Either cfg or model is mandatory and max one of them"
        
        # Assign the backbone
        if cfg is not None:
            # config is given
            self.backbone = self._get_bb_from_cfg(cfg)
        else:
            # Model is given
            self.backbone = bb_model
            
        self.cfg = cfg
        self.__train_backbone = None
        self.__store_trainable_params_n_cfg_bb(train_bb=train)
        
        
        
    def __store_trainable_params_n_cfg_bb(self, train_bb:bool):
        assert hasattr(self, 'backbone'), 'Must first set a backbone'
        assert not hasattr(self, 'backbone_req_grad'), 'Should only be called once'
        assert self.train_backbone is None
        
        # Store which backbone parameters require grad
        backbone_req_grad = []
        for p in self.backbone.parameters():
            backbone_req_grad.append(p.requires_grad)
        self.backbone_req_grad = backbone_req_grad
        
        # Set the attribute through it's setter 
        self.train_backbone = train_bb
    
    @property    
    def train_backbone(self):
        return self.__train_backbone
    
    @train_backbone.setter  
    def train_backbone(self, val):
        assert isinstance(val, bool), f'has to be a boolean type but got {type(val)}'
        assert hasattr(self, 'backbone_req_grad'), "Should only be called from a child class after init !"
        
        if val:
            self.train()
        else:
            self.eval()
            
        for i, p in enumerate(self.backbone.parameters()):
            if not val:
                p.requires_grad = val
            else:
                # Restore parameter training flags to the original state 
                p.requires_grad = self.backbone_req_grad[i]
                
        self.__train_backbone = val
        
    def train(self, mode: bool = True):
        assert len(list(self.parameters()))==len(list(self.backbone.parameters())),\
            "This wrapper should not contain any additional parameters other than the backbone"
        
        if self.train_backbone:
            return super().train(mode=mode)
        else:
            self.backbone.eval()
            return self
        
    @abstractmethod    
    def forward_backbone(self, x):
        pass
    
    @abstractmethod 
    def _get_bb_from_cfg(self, cfg:dict):
        pass
    
    @abstractmethod
    def get_out_nb_ch(self):
        pass
        
    def forward(self, x):
        if self.train_backbone:
            return self.forward_backbone(x)
        else:
            with torch.no_grad():
                return self.forward_backbone(x)
            
    def get_input_sz_multiple(self):
        """Returns the constriaint, nb of pixels that the input must be amultiple of on height and width"""
        return 1
            
    
class DinoBackBone(BackBoneBase):
    def __init__(self, 
                 n_out:int,
                 last_out_first:bool=True,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(bb_model=bb_model, cfg=cfg, train=train, *args, **kwargs)
        
        assert n_out > 1, f'n_out should be at least 1, but got: {n_out}'
        self.n_out = n_out
        self.last_out_first = last_out_first
        
        
    def _get_bb_from_cfg(self, cfg:dict):
        return get_dino_backbone(**cfg)
            
    
    def forward_backbone(self, x):
        out_patch_feats = self.backbone.get_intermediate_layers(x, n=self.n_out, reshape=True)
           
        if self.last_out_first:
            # Output of the last ViT block first in the tuple
            return out_patch_feats[::-1]
        else:
            # Output of the last ViT block last in the tuple
            return out_patch_feats
        
    def get_input_sz_multiple(self):
        """Returns the constriaint, nb of pixels that the input must be amultiple of on height and width"""
        return self.backbone.patch_size
    
    
    def get_out_nb_ch(self):
        return self.backbone.embed_dim
        
        
class ResNetBackBone(BackBoneBase):
    def __init__(self, 
                 
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        
        super().__init__(bb_model=bb_model, cfg=cfg, train=train, *args, **kwargs)
        
    
    def _get_bb_from_cfg(self, cfg:dict):
        pass
        
    def forward_backbone(self, x):
        pass
    
    def get_out_nb_ch(self):
        pass
        