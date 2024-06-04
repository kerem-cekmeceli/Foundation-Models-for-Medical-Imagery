import torch 
from torch import nn
from mmseg.ops import resize
from ModelSpecific.DinoMedical.prep_model import get_dino_backbone
from ModelSpecific.SamMedical.img_enc import get_sam_vit_backbone, get_sam_neck
from torchvision.models import get_model

from MedicalSegmentation.med_seg_foundation.models.EncDec.decoder.decoders import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead

from abc import abstractmethod
from typing import Union, Optional, Tuple, Callable, Sequence, Any, List
# from OrigModels.SAM.segment_anything.modeling.common import LayerNorm2d
from math import ceil, floor
from ModelSpecific.Reins.reins import Reins, LoRAReins
from ModelSpecific.MAE.prep_mae import get_mae_bb


class BackBoneBase(nn.Module):
    def __init__(self, 
                 name:str,
                 last_out_first:bool=True,
                 ) -> None:
        super().__init__()
        
        self.name = name
        self._input_sz_multiple = 1
        self.last_out_first = last_out_first
        
    @property  
    @abstractmethod    
    def target_size(self):
        pass
        
    @property  
    @abstractmethod    
    def train_backbone(self):
        pass
    
    @train_backbone.setter  
    @abstractmethod 
    def train_backbone(self, val):
        pass
        
    @property
    @abstractmethod    
    def hw_shrink_fac(self):
        pass
    
    @property
    @abstractmethod    
    def nb_outs(self):
        pass
    
    @property
    @abstractmethod    
    def out_feat_channels(self):
        pass
    
    @abstractmethod
    def get_pre_processing_cfg_list():
        pass
     
    @property        
    def input_sz_multiple(self):
        """Returns the constriaint, nb of pixels that the input must be amultiple of on height and width"""
        return self._input_sz_multiple
    
    @abstractmethod
    def forward(self, x):
        pass
    

class BackBoneBase1(BackBoneBase):
    def __init__(self, 
                 name:str,
                 nb_outs:int,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 pre_normalize:bool=False,
                 pix_mean:Optional[list]=None,
                 pix_std:Optional[list]=None,
                 target_size:Optional[Union[int, Tuple[int]]]=None,
                 interp_feats_to_orig_inp_size:Optional[bool]=False,
                 last_out_first:bool=True,
                 to_bgr:bool=False,
                 to_0_1:bool=False,
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
        super().__init__(name=name, last_out_first=last_out_first)
        assert isinstance(nb_outs, int) and nb_outs>0
        self._nb_outs = nb_outs
        
        if out_idx is not None:
            assert len(out_idx)==self.nb_outs
            max(out_idx)<self.nb_outs
            min(out_idx)>=0
        self.out_idx = out_idx
        
        assert (bb_model is None) ^ (cfg is None), "Either cfg or model is mandatory and max one of them"
        self._to_bgr = to_bgr
        self._to_0_1 = to_0_1
        
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
        
        self._input_sz_multiple = 1
        
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self._target_size = target_size
        
        self.interp_feats_to_orig_inp_size = interp_feats_to_orig_inp_size
        
        self.pre_normalize = pre_normalize
        if not pre_normalize:
            assert pix_mean is not None
            assert pix_std is not None
            self.register_buffer("pixel_mean", torch.Tensor(pix_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pix_std).view(-1, 1, 1), False)
        else:
            self.pixel_mean = pix_mean
            self.pixel_std = pix_std    
    
    @property
    def target_size(self):
        return self._target_size
            
    @property
    def nb_outs(self):
        return self._nb_outs 
    
    @property
    def nb_patches(self):
        if self._target_size is None:
            return (256//self.hw_shrink_fac)**2
        else:
            return (self._target_size[0]//self.hw_shrink_fac) * (self._target_size[1]//self.hw_shrink_fac)
    
    def _norm(self, x):
        # [B, C, H, W]
        
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        return x
        
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
        
        self.__train_backbone = val
        
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
                
        
    def train(self, mode: bool = True):            
        super().train(mode=mode)
        
        if not self.train_backbone:
           self.backbone.eval()
           
        return self
        
    
    @abstractmethod 
    def _get_bb_from_cfg(self, cfg:dict):
        pass
    
    @abstractmethod 
    def _get_pre_processing_cfg_list(self, cfg:dict):
        pass
    
    def get_pre_processing_cfg_list(self):
        processing = []
        if self.pre_normalize:
            # ImageNet values      
            processing.append(dict(type='Normalize', 
                                mean=torch.squeeze(self.pixel_mean).tolist(),  #RGB
                                std=torch.squeeze(self.pixel_std).tolist(),  #RGB
                                to_rgb=False))  # No need the chenge the order, already RGB from dataset
        
        processing.extend(self._get_pre_processing_cfg_list())
        return processing
    
    
    def interp_bb_inp(self, x):
        """Interpolates the input to the target size """
        assert hasattr(self, 'target_size')
        
        oldh, oldw = x.shape[-2:]
        (newh, neww) = self._target_size 
        
        # Interpolate to the supported image shape
        up = newh > oldh and neww > oldw
        x_new = resize(x, size=self._target_size, mode="bilinear" if up else 'area')
        
        if self.interp_feats_to_orig_inp_size:
            # target BB oup feat shape
            out_feat_h = oldh // self.hw_shrink_fac
            out_feat_w = oldw // self.hw_shrink_fac
            self.bb_target_oup_feat_shape = (out_feat_h, out_feat_w)
        
        return x_new
    
    
    def interp_bb_oup(self, out_feat): 
        """Interpolates the output features as if the original input size was used passing through the backbone"""
        
        assert self.interp_feats_to_orig_inp_size
        assert hasattr(self, 'bb_target_oup_feat_shape')
        
        h, w = out_feat.shape[-2:]
        h_t, w_t = self.bb_target_oup_feat_shape
        
        if self.bb_target_oup_feat_shape[0] != out_feat.shape[-2] or \
            self.bb_target_oup_feat_shape[1] != out_feat.shape[-1]:
            # Interpolate the oup feats 
            up = self.bb_target_oup_feat_shape[0] > out_feat.shape[-2] and self.bb_target_oup_feat_shape[1] > out_feat.shape[-1]
            reshaped_feats = resize(out_feat,
                                    size=self.bb_target_oup_feat_shape,
                                    mode="bilinear" if up else "area",
                                    )
            # [B, N_class, H', W']
            return reshaped_feats   
        else:
            return out_feat
    
    
    @abstractmethod    
    def _forward_backbone(self, x):
        pass
    
    def __forward_backbone(self, x):
        # [0, 255] --> [0, 1]
        if self._to_0_1:
            x = x/255.0
            assert 0<=torch.min(x) and torch.max(x)<=255.0
        
        # Normalize if no pre-normalizing is applied from the dataset
        if not self.pre_normalize:
            x = self._norm(x)
        
        # [B, C, H, W]    
        if self._to_bgr:
            x = x[:, [2, 1, 0], ...]
            
        
        # Interpolate the input to the target shape (if given)    
        if not self._target_size is None:
            x = self.interp_bb_inp(x)
        
        out_feats = self._forward_backbone(x)
        
        # Interpolate the output features as if no input interpolation was performed (if requested)
        if self.interp_feats_to_orig_inp_size:
            out_feats = list(out_feats)
            for i in range(len(out_feats)):
                out_feats[i] = self.interp_bb_oup(out_feats[i])
            out_feats = tuple(out_feats)
        
        # Order of the output features    
        if self.last_out_first:
            # Output of the last ViT block first in the tuple
            return out_feats[::-1]
        else:
            # Output of the last ViT block last in the tuple
            return out_feats            

    def forward(self, x):    
        if self.train_backbone:
            return self.__forward_backbone(x)
        else:
            with torch.no_grad():
                feats = self.__forward_backbone(x)
            return feats
        
        
class BlockBackboneBase(BackBoneBase1):
    def __init__(self, 
                 name: str,
                 nb_outs:int, 
                 bb_model: Optional[nn.Module] = None, 
                 cfg: Optional[dict] = None, 
                 train: bool = False,
                 pre_normalize: bool = False, 
                 pix_mean: Optional[list] = None, 
                 pix_std: Optional[list] = None, 
                 target_size: Optional[Union[int, Tuple[int]]] = None, 
                 interp_feats_to_orig_inp_size: Optional[bool] = False, 
                 last_out_first: bool = True, 
                 to_bgr: bool = False, 
                 to_0_1: bool = False, 
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
        super().__init__(name=name, nb_outs=nb_outs, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize, 
                         pix_mean=pix_mean, pix_std=pix_std, target_size=target_size, 
                         interp_feats_to_orig_inp_size=interp_feats_to_orig_inp_size, 
                         last_out_first=last_out_first, to_bgr=to_bgr, to_0_1=to_0_1, 
                         out_idx=out_idx,)
        
    
    @property
    @abstractmethod
    def blocks(self):
        """returns the transformer blocks"""
        pass
    
    def prep_pre_hook(self, x):
        """ Preperation step before starting running the blocks"""
        return x
    
    def blk_pre_hook(self, x):
        return x
    
    def blk_post_hook(self, x, i):
        return x
    
    def oup_before_append_hook(self, x, B, h, w, i):
        return x
    
    def oups_end_hook(self, outputs):
        return outputs
    
    def get_inter_layers(self, x, n=1):
        # Save the shape
        B, _, h, w = x.shape
        
        # prep pre-hook 
        x = self.prep_pre_hook(x)
        
        # Run the blocks and sace the outputs
        outputs, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, num in enumerate(blocks_to_take):
            if num<0:
                blocks_to_take[i] = len(self.blocks)+num
        assert min(blocks_to_take)>=0
        assert max(blocks_to_take)<len(self.blocks)
        for i, blk in enumerate(self.blocks):
            self.blk_pre_hook(x)
            x = blk(x)
            self.blk_post_hook(x, i)
            
            # Save the oup
            if i in blocks_to_take:
                oup = self.oup_before_append_hook(x, B, h, w, i)
                outputs.append(oup)
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
           
        return self.oups_end_hook(outputs)     
    
    def _forward_backbone(self, x):
        out_patch_feats = self.get_inter_layers(x, n=self.nb_outs if self.out_idx is None else self.out_idx)
        return out_patch_feats     
             
class DinoBackBone(BlockBackboneBase):
    def __init__(self, 
                 nb_outs:int,
                 name:str,
                 last_out_first:bool=True,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 disable_mask_tokens=True,
                 pre_normalize:bool=False,
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
        super().__init__(name=name, nb_outs=nb_outs, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize,
                         pix_mean=[123.675, 116.28, 103.53], pix_std=[58.395, 57.12, 57.375], target_size=224,#None,
                         interp_feats_to_orig_inp_size=False, 
                         last_out_first=last_out_first, to_bgr=True, to_0_1=False, out_idx=out_idx,
                         )
        # DINO NEEDS BGR ORDER !
        
        assert nb_outs >= 1, f'n_out should be at least 1, but got: {nb_outs}'
        assert nb_outs <= self.backbone.n_blocks, f'Requested n_out={nb_outs}, but only available {self.backbone.n_blocks}'
        self._input_sz_multiple = self.backbone.patch_size
        
        if disable_mask_tokens:
            self.backbone.mask_token.requires_grad = False
    
    @property
    def blocks(self):
        return self.backbone.blocks
    
    def prep_pre_hook(self, x):
        return self.backbone.prepare_tokens_with_masks(x)
    
    def oup_before_append_hook(self, x, B, h, w, i, norm=True):
        if norm:
            # Normalize
            out = self.backbone.norm(x)
        else:
            out = x
        
        # Remove cls and register tokens
        out = out[:, 1 + self.backbone.num_register_tokens:]
        
        # Reshape
        # Each oup is of shape: [B, N, embed_dim] --reshape--> [B, embed_dim, h_patches, w_patches]
        out = out.reshape(B, h // self.backbone.patch_size, w // self.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()

        return out    
        
        
    def _get_bb_from_cfg(self, cfg:dict):
        return get_dino_backbone(**cfg)
    
    def _get_pre_processing_cfg_list(self, central_crop=True):
        processing = []

        if central_crop:
            processing.append(dict(type='CentralCrop',  
                                    size_divisor=self.hw_shrink_fac))
        else:
            processing.append(dict(type='CentralPad',  
                                    size_divisor=self.hw_shrink_fac,
                                    pad_val=0, seg_pad_val=0))
            
        return processing
    
    @property  
    def out_feat_channels(self):
        return self.backbone.embed_dim
    
    @property
    def hw_shrink_fac(self):
        return self.backbone.patch_size
        

class DinoReinBackbone(DinoBackBone):
    def __init__(self, 
                 nb_outs: int, 
                 name: str, 
                 train_ft:bool=True,
                 last_out_first: bool = True, 
                 bb_model: Optional[nn.Module] = None, 
                 cfg: Optional[dict] = None, 
                 disable_mask_tokens=True, 
                 pre_normalize: bool = False, 
                 lora_reins:bool=False,
                 link_token_to_query:bool=False,
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
        
        super().__init__(nb_outs=nb_outs, name=name, last_out_first=last_out_first, bb_model=bb_model, 
                         cfg=cfg, train=False, disable_mask_tokens=disable_mask_tokens, pre_normalize=pre_normalize, 
                         out_idx=out_idx, )
        
        self.train_ft = train_ft
        self.lora_reins = lora_reins
        
        lora_params = dict(num_layers=len(self.backbone.blocks),
                           embed_dims=self.backbone.embed_dim,
                           patch_size=self.backbone.patch_size,
                           link_token_to_query=link_token_to_query)
        if not self.lora_reins:
            self.reins: Reins = Reins(**lora_params)
        else:
            self.reins: LoRAReins = LoRAReins(**lora_params)
            
        # Turn fine tuning training off if necessary
        if not self.train_ft:
            for param in self.reins.parameters():
                param.requires_grad=False
        
        # If to apply layer norm before append
        self.do_norm = False
        
        if not self.do_norm:
            for param in self.backbone.norm.parameters():
                param.requires_grad=False
    
    def blk_post_hook(self, x, i):
        # Apply Rein
        x = self.reins.forward(
            x,
            i,
            batch_first=True,
            has_cls_token=True,
        )
        return x    
    
    def oups_end_hook(self, outputs):
        return self.reins.return_auto(outputs)
    
    def oup_before_append_hook(self, x, B, h, w, i):
        return super().oup_before_append_hook(x, B, h, w, i, norm=self.do_norm)
        
  
class SamBackBone(BlockBackboneBase):
    def __init__(self, 
                 nb_outs:int,
                 name,
                 last_out_first:bool=True,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 interp_to_inp_shape:bool=True,
                 pre_normalize:bool=False,
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
        
        super().__init__(name=name, nb_outs=nb_outs, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize,
                         pix_mean=[123.675, 116.28, 103.53], pix_std=[58.395, 57.12, 57.375], target_size=224,#1024, 
                         interp_feats_to_orig_inp_size=interp_to_inp_shape, 
                         last_out_first=last_out_first, to_bgr=False, to_0_1=False, out_idx=out_idx, )
        
        assert self.nb_outs >= 1, f'n_out should be at least 1, but got: {self.nb_outs}'
        assert self.nb_outs <= self.backbone.n_blocks, f'Requested n_out={self.nb_outs}, but only available {self.backbone.n_blocks}'     
    
    @property
    def blocks(self):
        return self.backbone.blocks
    
    def _get_bb_from_cfg(self, cfg:dict):
        return get_sam_vit_backbone(**cfg)
    
    def _get_pre_processing_cfg_list(self):
        processing = []
        
        #  Pad to a square input
        processing.append(dict(type='CentralPad', 
                               make_square=True,
                               pad_val=0, seg_pad_val=0))
        return processing
    
    @property  
    def out_feat_channels(self):
        vit_ch = self.backbone.patch_embed.proj.out_channels
        if self.backbone.neck is None:
            return vit_ch
        else:
            neck_ch = self.backbone.neck[-2].out_channels
            if self.nb_outs==1:
                return neck_ch
            else:
                n_o = [neck_ch]
                v_o = [vit_ch]*(self.nb_outs-1)
                if self.last_out_first:
                    return n_o + v_o
                else:
                    return v_o + n_o   
    
    @property
    def hw_shrink_fac(self):
        return self.backbone.patch_embed.proj.kernel_size[0]
    
    def prep_pre_hook(self, x):
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        return x
    
    def oup_before_append_hook(self, x, B, h, w, i, norm=True):
        out = x.permute(0, 3, 1, 2).contiguous()
        if (i==len(self.blocks)-1) and self.backbone.neck is not None:
            out = self.backbone.neck(out)
        return out    
    
    
class SamReinBackBone(SamBackBone):
    def __init__(self, 
                 nb_outs: int, 
                 name, 
                 train_ft:bool=True,
                 last_out_first: bool = True, 
                 bb_model: Optional[nn.Module]= None, 
                 cfg: Optional[dict] = None, 
                 interp_to_inp_shape: bool = True, 
                 pre_normalize: bool = False, 
                 lora_reins:bool=False,
                 link_token_to_query:bool=False,
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
        
        super().__init__(nb_outs=nb_outs, name=name, last_out_first=last_out_first, bb_model=bb_model, 
                         cfg=cfg, train=False, interp_to_inp_shape=interp_to_inp_shape, 
                         pre_normalize=pre_normalize, out_idx=out_idx, )
        
        self.train_ft = train_ft
        self.lora_reins = lora_reins
        lora_params = dict(num_layers=len(self.backbone.blocks),
                           embed_dims=self.backbone.patch_embed.proj.out_channels,
                           patch_size=self.hw_shrink_fac,
                           link_token_to_query=link_token_to_query)
        
        if not self.lora_reins:
            self.reins: Reins = Reins(**lora_params)
        else:
            self.reins: LoRAReins = LoRAReins(**lora_params)   
            
        # Turn fine tuning training off if necessary
        if not self.train_ft:
            for param in self.reins.parameters():
                param.requires_grad=False
        
        
    def blk_post_hook(self, x, i):
        B = x.shape[0]
        return self.reins.forward(x.view(B,-1,self.backbone.patch_embed.proj.out_channels),
                                         i,
                                         batch_first=True,
                                         has_cls_token=False)
 
 
class MAEBackbone(BlockBackboneBase):
    def __init__(self, 
                 name: str, 
                 nb_outs: int, 
                 bb_model: Optional[nn.Module] = None, 
                 cfg: Optional[dict] = None, 
                 train: bool = False, 
                 pre_normalize: bool = False, 
                 last_out_first: bool = True, 
                 out_idx:Optional[List[int]]=None,
                 ) -> None:
         super().__init__(name=name, nb_outs=nb_outs, bb_model=bb_model, cfg=cfg, train=train, 
                          pre_normalize=pre_normalize,  pix_mean=[0.485, 0.456, 0.406], pix_std=[0.229, 0.224, 0.225],
                          target_size=224, interp_feats_to_orig_inp_size=False, 
                          last_out_first=last_out_first, to_bgr=False, to_0_1=True, out_idx=out_idx,
                          )
         
    @property
    def blocks(self):
        return self.backbone.blocks
    
    @property
    def hw_shrink_fac(self):
        return self.backbone.patch_embed.patch_size[0]
    
    @property
    def out_feat_channels(self):
        return self.backbone.patch_embed.proj.out_channels
    
    def get_pre_processing_cfg_list(self):
        processing = []
        processing.append(dict(type='CentralCrop',  
                          size_divisor=self.hw_shrink_fac))
        return processing
    
    def _get_bb_from_cfg(self, cfg:dict):
        return get_mae_bb(**cfg)
    
    def prep_pre_hook(self, x):
        # embed patches
        x = self.backbone.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.backbone.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def blk_post_hook(self, x, i):
        if i == len(self.blocks) - 1:
            x = self.backbone.norm(x)
        return x
    
    def oup_before_append_hook(self, x, B, h, w, i):
        out = x[:, 1:]
        B, _, C = out.shape
        out = out.reshape(B, h//self.hw_shrink_fac, w//self.hw_shrink_fac, C).permute(0, 3, 1, 2).contiguous()
        return out
    
    def oups_end_hook(self, outputs):
        return tuple(outputs)
    
      
    
class MAEReinBackbone(MAEBackbone):
    def __init__(self, 
                 name: str, 
                 nb_outs: int, 
                 bb_model: Optional[nn.Module] = None, 
                 cfg: Optional[dict] = None, 
                 train_ft: bool = True, 
                 pre_normalize: bool = False, 
                 last_out_first: bool = True, 
                 out_idx:Optional[List[int]]=None,
                 lora_reins:bool=False,
                 link_token_to_query:bool=False,
                 ):
        super().__init__(name=name, nb_outs=nb_outs, bb_model=bb_model, cfg=cfg, train=False,
                         pre_normalize=pre_normalize, last_out_first=last_out_first, out_idx=out_idx) 
        
        self.train_ft=train_ft
        self.lora_reins = lora_reins
        lora_params = dict(num_layers=len(self.backbone.blocks),
                           embed_dims=self.backbone.patch_embed.proj.out_channels,
                           patch_size=self.hw_shrink_fac,
                           link_token_to_query=link_token_to_query)
        if not self.lora_reins:
            self.reins: Reins = Reins(**lora_params)
        else:
            self.reins: LoRAReins = LoRAReins(**lora_params) 
            
        # Turn fine tuning training off if necessary
        if not self.train_ft:
            for param in self.reins.parameters():
                param.requires_grad=False
            
        # If to do Layernorm before reins (for last block only)
        self.do_norm=False
        
        if not self.do_norm:
            for param in self.backbone.norm.parameters():
                param.requires_grad=False
            
        
    def blk_post_hook(self, x, i):
        # Do layer norm (for last blk only)
        if self.do_norm:
            x = super().blk_post_hook(x=x, i=i)
            
        # Apply Rein
        x = self.reins.forward(
            x,
            i,
            batch_first=True,
            has_cls_token=True,
        )
        
        return x
        
         
# ResNet NEEDS RGB ORDER !              
class ResNetBackBone(BackBoneBase1): 
    def __init__(self, 
                 name,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 nb_layers:int=50,
                 skip_last_layer:bool=False,
                 pre_normalize:bool=False,
                 nb_outs:int=1,
                 last_out_first:bool=True,
                 uniform_oup_size:bool=True,
                 out_idx:Optional[List[int]]=None,
                 outs_from_diff_conv_layers:bool=False,
                ) -> None:
        
        super().__init__(name=name, nb_outs=nb_outs, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize,
                         pix_mean=[0.485, 0.456, 0.406], pix_std=[0.229, 0.224, 0.225], 
                         target_size=224, interp_feats_to_orig_inp_size=False, 
                         last_out_first=last_out_first,  to_bgr=False, to_0_1=True)
        
        assert nb_layers in [18, 34, 50, 101, 152], f'Nb of layers {nb_layers} is not a valid resnet number'
        self._nb_layers = nb_layers
        self._hw_shrink_fac = 32 if not skip_last_layer else 16
        assert self.nb_outs>1
        self._skip_last_layer = skip_last_layer
        
        if self._skip_last_layer:
            self.backbone.layer4 = None
            
        self.backbone.avgpool = None 
        self.backbone.fc = None     
        
        self.nb_conv_layers = 3 if skip_last_layer else 4
        
        # Which conv layers to take the outs from
        if out_idx is None:
            out_idx = [i for i in range(self.nb_conv_layers)]
        else:
            assert isinstance(out_idx, list)
            for i, oi in enumerate(out_idx):
                if oi<0:
                    out_idx[i] = self.nb_conv_layers+oi
            assert max(out_idx)<self.nb_conv_layers
            assert min(out_idx)>=0
        out_idx.sort(reverse=True)  # Start taking oups from the latest layer
        self.out_idx = out_idx
        
        # If to take outputs from all conv layers (starting from tha last)
        self.outs_from_diff_conv_layers=outs_from_diff_conv_layers
        
        self.assign_nb_outs_per_layer()
        
        self.uniform_oup_size = uniform_oup_size
        
        if self.uniform_oup_size:
            necks = []
            for layer_i, nb_outs_i in enumerate(self.nb_outs_per_l):
                if nb_outs_i>0:
                    layer_i_out_ch = self.get_out_channels(getattr(self.backbone, f'layer{layer_i+1}'))
                    
                    assert self.out_feat_channels >= layer_i_out_ch
                    assert self.out_feat_channels % layer_i_out_ch == 0
                    factor =  self.out_feat_channels // layer_i_out_ch
        
                    for j in range(nb_outs_i):
                        if factor>1:
                            assert factor%2==0
                            necks.append(self.get_inter_out_neck(layer_i_out_ch, factor))
                        else:
                            necks.append(nn.Identity())
            assert len(necks) == self.nb_outs
            self.necks_per_out = nn.ModuleList(necks)
                 
    
    def get_inter_out_neck(self, nb_out_channels, factor):
        simple = True
        if simple:
            return nn.Sequential(nn.Conv2d(nb_out_channels, self.out_feat_channels, kernel_size=1),
                                 nn.SyncBatchNorm(self.out_feat_channels),
                                 nn.Upsample(scale_factor=1/factor, mode='bilinear' if (1/factor)>1 else 'area'))
        else:
            assert self.out_ch_target is None
            # Bottleneck blocks
            if self._nb_layers>34:
                assert nb_out_channels%2==0
                return nn.Sequential(nn.Conv2d(nb_out_channels , nb_out_channels//2, kernel_size=1),
                                    nn.SyncBatchNorm(nb_out_channels//2),
                                    nn.ReLU(),
                                    nn.Conv2d(nb_out_channels//2, nb_out_channels//2, 
                                            kernel_size=3, stride=factor, dilation=factor//2, padding=factor//2),
                                    nn.SyncBatchNorm(nb_out_channels//2),
                                    nn.ReLU(),
                                    nn.Conv2d(nb_out_channels//2, self.out_feat_channels, kernel_size=1),
                                    nn.SyncBatchNorm(self.out_feat_channels),
                                    nn.ReLU())
        
            
            # Basic blocks
            else:
                return nn.Sequential(nn.Conv2d(nb_out_channels, self.out_feat_channels, 
                                            kernel_size=3, stride=factor, dilation=factor//2, padding=factor//2),
                                    nn.SyncBatchNorm(self.out_feat_channels),
                                    nn.ReLU())
    
    def assign_nb_outs_per_layer(self):
        assert hasattr(self, 'nb_outs')
        assert hasattr(self, '_skip_last_layer')
        self.nb_outs_per_l = [0]*self.nb_conv_layers
        
        if not self.outs_from_diff_conv_layers:
            # Taking starting from the last until we reach the nb of desired oups
            nb_outs = self.nb_outs
            for i in self.out_idx:
                if nb_outs>0:
                    self.nb_outs_per_l[i] = min(len(getattr(self.backbone, f'layer{i+1}')), nb_outs)
                nb_outs -= self.nb_outs_per_l[i]
            assert nb_outs<=0, f'Requested nb_outs={self.nb_outs}, but only {self.nb_outs+nb_outs} available'    
            
        else:
            # Taking from all layers round-robin (satrting from the last)
            while(sum(self.nb_outs_per_l)<self.nb_outs):
                took_oup = False
                for i in self.out_idx:
                    if sum(self.nb_outs_per_l)<self.nb_outs:
                        if len(getattr(self.backbone, f'layer{i+1}'))>self.nb_outs_per_l[i]:
                            self.nb_outs_per_l[i] += 1
                            took_oup = True
                    
                assert took_oup, "Requested nb of output is larger than the max available !"
                
        assert sum(self.nb_outs_per_l) == self.nb_outs
        assert len(self.nb_outs_per_l)==self.nb_conv_layers
                
        # Get the nb channels for oups
        self.oup_channels = []

        for i, nb_outs in enumerate(self.nb_outs_per_l):
            self.oup_channels.extend([self.get_out_channels(getattr(self.backbone, f'layer{i+1}'))]*self.nb_outs_per_l[i])
        
        assert len(self.oup_channels)==self.nb_outs
            
    def get_inter_res_from_layer(layer, n, x):
        """ layer: layer to take the blocks from,
            n: number of blocks (from the end) to take,
            x: input """
            
        output, total_block_len = [], len(layer)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        if len(blocks_to_take)>0:
            assert max(blocks_to_take)<total_block_len
        for i, blk in enumerate(layer):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take) == n, f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return x, output
    
    def get_inter_res(self, x):
        # Initial convolutions
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # 4 Consecutive layers
        outputs = []
        for i in range(self.nb_conv_layers):
            layer = getattr(self.backbone, f'layer{i+1}')
            x, out_i = ResNetBackBone.get_inter_res_from_layer(layer, self.nb_outs_per_l[i], x)
            outputs.extend(out_i)
            
        if self.uniform_oup_size:
            for i, neck in enumerate(self.necks_per_out):
                outputs[i] = neck(outputs[i])
                assert outputs[i].shape == outputs[0].shape
            
        return outputs
                    
    
    def _get_bb_from_cfg(self, cfg:dict):
        # cfg = dict(name=f'resnet{self._nb_layers}', weights=f'ResNet{self._nb_layers}_Weights.DEFAULT')
        # No weights - random initialization
        return get_model(**cfg)
    
    def _forward_backbone(self, x):
        return self.get_inter_res(x)
        
    def forward_no_inter(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        if not self._skip_last_layer:
            x = self.backbone.layer4(x)
        return (x,)
    
    def get_out_channels(self, layer):
        if self._nb_layers>34:
            return layer[-1].conv3.out_channels
        else:
            return layer[-1].conv2.out_channels
        
    @property  
    def out_feat_channels(self):
        if not self.uniform_oup_size:
            if self.last_out_first:
                return self.oup_channels[::-1]
            else:
                return self.oup_channels
        
        else:
            if self._skip_last_layer:
                return self.get_out_channels(self.backbone.layer3)
            else:
                return self.get_out_channels(self.backbone.layer4)
    
    @property
    def hw_shrink_fac(self):
        # assert self.uniform_oup_size, "Not implemented yet"
        return self._hw_shrink_fac
    
    def get_pre_processing_cfg_list(self):
        processing = []
        
        #  Pad to a square input
        processing.append(dict(type='CentralPad', 
                               make_square=True,
                               pad_val=0, seg_pad_val=0))
        return processing
    
class LadderBackbone(BackBoneBase):
    def __init__(self, 
                 name: str,
                 bb1_name_params:dict, 
                 bb2_name_params:dict, 
                ) -> None:
        super().__init__(name,)
                
        bb1_name = bb1_name_params['name']
        bb1_params = bb1_name_params['params']
        self.bb1 = globals()[bb1_name](**bb1_params)
        assert isinstance(self.bb1, BlockBackboneBase)
        
        bb2_name = bb2_name_params['name']
        bb2_params = bb2_name_params['params']
        self.bb2 = globals()[bb2_name](**bb2_params)
        assert isinstance(self.bb2, BackBoneBase1)
        if not hasattr(self.bb2, 'train_ft'):
            assert self.bb2.train_backbone
        else:
            assert self.bb2.train_backbone or self.bb2.train_ft
        
        self._input_sz_multiple = self.bb1.input_sz_multiple
        
        # same number of outputs for both backbones  
        assert self.bb1.nb_outs == self.bb2.nb_outs
        assert self.bb1.last_out_first == self.bb2.last_out_first
        
        # This takes care of matching the channel sizes
        bb1_out_chs = [self.bb1.out_feat_channels]*self.nb_outs if isinstance(self.bb1.out_feat_channels, int) else self.bb1.out_feat_channels
        bb2_out_chs = [self.bb2.out_feat_channels]*self.nb_outs if isinstance(self.bb2.out_feat_channels, int) else self.bb2.out_feat_channels
        
        necks_ch = []
        for bb1_out, bb2_out in zip(bb1_out_chs, bb2_out_chs):
            if bb1_out != bb2_out:
                # Might try MLP
                necks_ch.append(nn.Sequential(nn.Conv2d(bb2_out, bb1_out, kernel_size=1),
                                              nn.SyncBatchNorm(bb1_out)) )
            else:
                necks_ch.append(nn.Identity())
        self.necks_y2_ch = nn.ModuleList(necks_ch)
            
        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(self.bb1.nb_outs)])
        self.Sigmoid = nn.Sigmoid()
    
    @property
    def nb_patches(self):
        return self.bb1.nb_patches
 
    @property
    def hw_shrink_fac(self):
        return self.bb1.hw_shrink_fac
    
    @property  
    def out_feat_channels(self):
        return self.bb1.out_feat_channels
    
    @property
    def nb_outs(self):
        return self.bb1.nb_outs
    
    @property  
    @abstractmethod    
    def train_backbone(self):
        return self.bb1.train_backbone
    
    @train_backbone.setter  
    @abstractmethod 
    def train_backbone(self, val):
        self.bb1.train_backbone = val
    
    
    def get_pre_processing_cfg_list(self):
        processing = []
        
        if isinstance(self.bb1, DinoBackBone):
            multiple = self.bb1.hw_shrink_fac
        elif isinstance(self.bb2, DinoBackBone):
            multiple = self.bb2.hw_shrink_fac
        else:
            multiple = None
        
        if not multiple is None:
            processing.append(dict(type='CentralCrop',  
                                    size_divisor=multiple))
        return processing
    
    @property
    def target_size(self):
        return self.bb1.target_size
    
    def forward(self, x):
        y1s = self.bb1(x)
        y2s = self.bb2(x)
        
        assert len(y1s)==len(y2s)==len(self.alpha)
                
        # Tuple to list
        y1s = list(y1s)
        y2s = list(y2s)
        ys = []
        
        for y1, y2, a, neck_y2_ch in zip(y1s, y2s, self.alpha, self.necks_y2_ch):        
            # Same ch dim
            y2 = neck_y2_ch(y2)
            
            if y1.shape[-2:] != y2.shape[-2:]:
                up = y1.shape[-2] > y2.shape[-2] and y1.shape[-1] > y2.shape[-1]
                # Interpolate to get pixel logits frmo patch logits
                y2 = resize(input=y2,
                            size=y1.shape[-2:],
                            mode="bilinear" if up >= 1 else "area",
                            align_corners=False if up else None)
            
            gate = self.Sigmoid(a)
            y = gate * y1 + (1-gate) * y2
            ys.append(y)
            
        return tuple(ys)
        
    
implemented_backbones = [DinoBackBone.__name__,
                         SamBackBone.__name__,
                         ResNetBackBone.__name__,
                         LadderBackbone.__name__,
                         DinoReinBackbone.__name__,
                         SamReinBackBone.__name__,
                         MAEBackbone.__name__]
        
        