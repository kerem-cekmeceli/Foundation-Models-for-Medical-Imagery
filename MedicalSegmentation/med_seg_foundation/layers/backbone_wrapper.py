import torch 
from torch import nn
from mmseg.ops import resize
from ModelSpecific.DinoMedical.prep_model import get_dino_backbone
from ModelSpecific.SamMedical.img_enc import get_sam_vit_backbone, get_sam_neck
from torchvision.models import get_model

from layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead

from abc import abstractmethod
from typing import Union, Optional, Tuple, Callable, Any
# from OrigModels.SAM.segment_anything.modeling.common import LayerNorm2d



class BackBoneBase(nn.Module):
    def __init__(self, 
                 name:str,
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)
        
        self.name = name
        self._input_sz_multiple = 1
        
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
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 pre_normalize:bool=False,
                 pix_mean:Optional[list]=None,
                 pix_std:Optional[list]=None,
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(name=name, *args, **kwargs)
                
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
        
        self._input_sz_multiple = 1
        
        self.pre_normalize = pre_normalize
        if not pre_normalize:
            assert pix_mean is not None
            assert pix_std is not None
            self.register_buffer("pixel_mean", torch.Tensor(pix_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pix_std).view(-1, 1, 1), False)
        else:
            self.pixel_mean = pix_mean
            self.pixel_std = pix_std
            
    def _norm(self, x):
        # [B, C, H, W]
        
        # Order BGR -> RGB
        x = torch.flip(x, dims=[1])
        
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
        assert len(list(self.parameters()))==len(list(self.backbone.parameters())),\
            "This wrapper should not contain any additional parameters other than the backbone"
            
        super().train(mode=mode)
        
        if not self.train_backbone:
           self.backbone.eval()
           
        return self
        
    @abstractmethod    
    def forward_backbone(self, x):
        pass
    
    @abstractmethod 
    def _get_bb_from_cfg(self, cfg:dict):
        pass
    

    def forward(self, x):
        if not self.pre_normalize:
            x = self._norm(x)
            
        if self.train_backbone:
            return self.forward_backbone(x)
        else:
            with torch.no_grad():
                feats = self.forward_backbone(x)
            return feats
     
            
class DinoBackBone(BackBoneBase1):
    def __init__(self, 
                 nb_outs:int,
                 name:str,
                 last_out_first:bool=True,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 disable_mask_tokens=True,
                 pre_normalize:bool=False,
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(name=name, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize,
                         pix_mean=[123.675, 116.28, 103.53], pix_std=[58.395, 57.12, 57.375], *args, **kwargs)
        
        assert nb_outs >= 1, f'n_out should be at least 1, but got: {nb_outs}'
        assert nb_outs <= self.backbone.n_blocks, f'Requested n_out={nb_outs}, but only available {self.backbone.n_blocks}'
        self._nb_outs = nb_outs
        self.last_out_first = last_out_first
        self._input_sz_multiple = self.backbone.patch_size
        
        if disable_mask_tokens:
            self.backbone.mask_token.requires_grad = False
        
        
    def _get_bb_from_cfg(self, cfg:dict):
        return get_dino_backbone(**cfg)
    
    def get_pre_processing_cfg_list(self, central_crop=True):
        processing = []
        
        # img_scale_fac = 1  # Keep at 1 
        # if img_scale_fac != 1:
        #     processing.append(dict(type='Resize2',
        #                         scale_factor=float(img_scale_fac), #HW
        #                         keep_ratio=True))
        
        if self.pre_normalize:
            # ImageNet values      
            processing.append(dict(type='Normalize', 
                                mean=self.pixel_mean,  #RGB
                                std=self.pixel_std,  #RGB
                                to_rgb=True))  # Converts BGR (prev steps) to RGB initially
        
        if central_crop:
            processing.append(dict(type='CentralCrop',  
                                    size_divisor=self.hw_shrink_fac))
        else:
            processing.append(dict(type='CentralPad',  
                                    size_divisor=self.hw_shrink_fac,
                                    pad_val=0, seg_pad_val=0))
            
        return processing
            
    
    def forward_backbone(self, x):
        out_patch_feats = self.backbone.get_intermediate_layers(x, n=self._nb_outs, reshape=True)
           
        if self.last_out_first:
            # Output of the last ViT block first in the tuple
            return out_patch_feats[::-1]
        else:
            # Output of the last ViT block last in the tuple
            return out_patch_feats
        
    @property
    def nb_outs(self):
        return self._nb_outs 
    
    @property  
    def out_feat_channels(self):
        return self.backbone.embed_dim
    
    @property
    def hw_shrink_fac(self):
        return self.backbone.patch_size
        

class SamBackBone(BackBoneBase1):
    def __init__(self, 
                 nb_outs:int,
                 name,
                 last_out_first:bool=True,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 interp_to_inp_shape:bool=True,
                 pre_normalize:bool=False,
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        
        super().__init__(name=name, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize,
                         pix_mean=[123.675, 116.28, 103.53], pix_std=[58.395, 57.12, 57.375], *args, **kwargs)
        
        assert nb_outs >= 1, f'n_out should be at least 1, but got: {nb_outs}'
        assert nb_outs <= self.backbone.n_blocks, f'Requested n_out={nb_outs}, but only available {self.backbone.n_blocks}'
        self._nb_outs = nb_outs
        self.last_out_first = last_out_first
        self.interp_to_inp_shape = interp_to_inp_shape
        
    def reshape_bb_inp(self, x):
        oldh, oldw = x.shape[-2:]
        scale = self.backbone.img_size * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        target_size = (newh, neww)
        
        # Interpolate to the supported image shape
        up = newh > oldh and neww > oldw
        x_new = resize(x, size=target_size, mode="bilinear" if up else 'area')
        
        # BB oup feat shape
        out_feat_h = oldh // self.hw_shrink_fac
        out_feat_w = oldw // self.hw_shrink_fac
        self.bb_oup_feat_shape = (out_feat_h, out_feat_w)
        
        return x_new
        
    def reshape_bb_oup(self, out_feat): 
        assert hasattr(self, 'bb_oup_feat_shape')
        
        # Interpolate the oup feats 
        up = self.bb_oup_feat_shape[0] > out_feat.shape[-2] and self.bb_oup_feat_shape[1] > out_feat.shape[-1]
        reshaped_feats = resize(out_feat,
                                size=self.bb_oup_feat_shape,
                                mode="bilinear" if up else "area",
                                )
        # [B, N_class, H', W']
        return reshaped_feats    
        
    
    def _get_bb_from_cfg(self, cfg:dict):
        return get_sam_vit_backbone(**cfg)
    
    def get_pre_processing_cfg_list(self):
        processing = []
        
        if self.pre_normalize:
            # ImageNet values      
            processing.append(dict(type='Normalize', 
                                mean=self.pixel_mean,  #RGB
                                std=self.pixel_std,  #RGB
                                to_rgb=True))  # Converts BGR (prev steps) to RGB initially
        
        #  Pad to a square input
        processing.append(dict(type='CentralPad',  
                            #    size=self.backbone.img_size if not self.interp_to_inp_shape else None,
                            #    make_square=self.interp_to_inp_shape,
                               make_square=True,
                               pad_val=0, seg_pad_val=0))
            
        return processing
        
    def forward_backbone(self, x):
        x = self.reshape_bb_inp(x)
        out_patch_feats = self.backbone.get_intermediate_layers(x, n=self._nb_outs)
        
        if self.interp_to_inp_shape:
            out_patch_feats = list(out_patch_feats)
            for i in range(len(out_patch_feats)):
                out_patch_feats[i] = self.reshape_bb_oup(out_patch_feats[i])
            out_patch_feats = tuple(out_patch_feats)
           
        if self.last_out_first:
            # Output of the last ViT block first in the tuple
            return out_patch_feats[::-1]
        else:
            # Output of the last ViT block last in the tuple
            return out_patch_feats
    
    @property
    def nb_outs(self):
        return self._nb_outs
    
    @property  
    def out_feat_channels(self):
        if self.backbone.neck is None:
            return self.backbone.patch_embed.proj.out_channels
        else:
            return self.backbone.neck[-2].out_channels
        
    
    @property
    def hw_shrink_fac(self):
        return self.backbone.patch_embed.proj.kernel_size[0]
    
            
class ResNetBackBone(BackBoneBase1): 
    def __init__(self, 
                 name,
                 bb_model:Optional[nn.Module]=None,
                 cfg:Optional[dict]=None,
                 train: bool = False, 
                 nb_layers:int=50,
                 skip_last_layer:bool=False,
                 pre_normalize:bool=False,
                 *args: torch.Any, **kwargs: torch.Any) -> None:
        
        super().__init__(name=name, bb_model=bb_model, cfg=cfg, train=train, pre_normalize=pre_normalize,
                         pix_mean=[123.675, 116.28, 103.53], pix_std=[58.395, 57.12, 57.375], *args, **kwargs)
        
        assert nb_layers in [18, 34, 50, 101, 152], f'Nb of layers {nb_layers} is not a valid resnet number'
        self._nb_layers = nb_layers
        self._expected_inp_size = 224
        self._hw_shrink_fac = 32 if not skip_last_layer else 16
        self._nb_outs=1
        self._skip_last_layer = skip_last_layer
        
        # Turn off gradient computation for the final FC and avg pooling layers of the resnet model (Required for DDP)
        params_no_train = [self.backbone.avgpool, self.backbone.fc]
        if self._skip_last_layer:
            params_no_train.append(self.backbone.layer4)
        for layer in params_no_train:
            for param in layer.parameters():
                param.requires_grad = False
                    
    
    def _get_bb_from_cfg(self, cfg:dict):
        # cfg = dict(name=f'resnet{self._nb_layers}', weights=f'ResNet{self._nb_layers}_Weights.DEFAULT')
        # No weights - random initialization
        return get_model(**cfg)
        
    def forward_backbone(self, x):
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
    
    @property
    def nb_outs(self):
        return self._nb_outs
    
    @property  
    def out_feat_channels(self):
        if self._skip_last_layer:
            last_layer = self.backbone.layer3[-1]
        else:
            last_layer = self.backbone.layer4[-1]
            
        if self._nb_layers>34:
            return last_layer.conv3.out_channels
        else:
            return last_layer.conv2.out_channels
    
    @property
    def hw_shrink_fac(self):
        return self._hw_shrink_fac
    
    def get_pre_processing_cfg_list(self):
        processing = []
        
        #  Crop 
        processing.append(dict(type='CentralCrop',  
                               size_divisor=self._hw_shrink_fac))
      
        processing.append(dict(type='Resize2',
                            scale=self._expected_inp_size, #HW
                            keep_ratio=True))
    
        if self.pre_normalize:
            # ImageNet values      
            processing.append(dict(type='Normalize', 
                                mean=self.pixel_mean,  #RGB
                                std=self.pixel_std,  #RGB
                                to_rgb=True))  # Converts BGR (prev steps) to RGB initially
        
        return processing
    
    
class LadderBackbone(BackBoneBase):
    def __init__(self, 
                 name: str,
                 bb1_name_params:dict, 
                 bb2_name_params:dict, 
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(name, *args, **kwargs)
        
        self.img_size = 224
        
        bb1_name = bb1_name_params['name']
        bb1_params = bb1_name_params['params']
        self.bb1 = globals()[bb1_name](**bb1_params)
        assert isinstance(self.bb1, BackBoneBase)
        
        bb2_name = bb2_name_params['name']
        bb2_params = bb2_name_params['params']
        self.bb2 = globals()[bb2_name](**bb2_params)
        assert isinstance(self.bb2, BackBoneBase)
        assert self.bb2.train_backbone
        
        self._input_sz_multiple = self.bb1.input_sz_multiple
        
        assert self.img_size%self.bb2.hw_shrink_fac==0, f'bb2, WH is not evenly divisible by {self.img_size}'
        assert self.img_size%self.bb1.hw_shrink_fac==0, f'bb1, WH is not evenly divisible by {self.img_size}'
        
        # This takes care of matching the channel sizes
        if self.bb1.out_feat_channels != self.bb2.out_feat_channels:
            self.neck_channels = get_sam_neck(in_channels=self.bb2.out_feat_channels, out_channels=self.bb1.out_feat_channels)
        else:
            self.neck_channels = None
            
        # This takes care of having the same number of outputs for both backbones  
        if self.bb1.nb_outs != self.bb2.nb_outs:
            assert self.bb2.nb_outs == 1
            self.necks_bb2 = nn.ModuleList([nn.Sequential(nn.Conv2d(self.bb2.out_feat_channels, self.bb2.out_feat_channels, 3, padding='same'),
                                                          nn.ReLU(),
                                                          nn.SyncBatchNorm(self.bb2.out_feat_channels)) 
                                            for i in range(self.bb1.nb_outs)])
        else:
            self.necks_bb2 = None
            
        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(self.bb1.nb_outs)])
        self.Sigmoid = nn.Sigmoid()
        
 
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
        
        #  Crop to a square input
        processing.append(dict(type='CentralCrop',  
                               make_square=True))
      
        # Resize to 224x224
        processing.append(dict(type='Resize2',
                            scale=self.img_size, #HW
                            keep_ratio=True))
        return processing
    
    # def check_size(self, x):
    #     assert x.shape[-2]%self.bb2.hw_shrink_fac==0, 'bb2, H is not evenly divisible'
    #     assert x.shape[-2]%self.bb1.hw_shrink_fac==0, 'bb1, H is not evenly divisible'
    #     assert x.shape[-1]%self.bb2.hw_shrink_fac==0, 'bb2, W is not evenly divisible'
    #     assert x.shape[-1]%self.bb1.hw_shrink_fac==0, 'bb1, W is not evenly divisible'
    
    def forward(self, x):
        # self.check_size(x)
        
        y1s = self.bb1(x)
        y2s = self.bb2(x)
        
        if self.necks_bb2 is None:
            assert len(y1s)==len(y2s)==len(self.alpha)
        else:
            assert len(y2s)==1
            y2s_n = []
            for neck2 in self.necks_bb2:
                y2s_n.append(neck2(y2s[0]))    
            y2s = y2s_n
                
        # Tuple to list
        y1s = list(y1s)
        y2s = list(y2s)
        ys = []
        
        for y1, y2, a in zip(y1s, y2s, self.alpha):        
            if self.neck_channels is not None:
                y2 = self.neck_channels(y2)
            
            if y1.shape[-2:] != y2.shape[-2:]:
                up = y1.shape[-2] > y2.shape[-2] and y1.shape[-1] > y2.shape[-1]
                # Interpolate to get pixel logits frmo patch logits
                y2 = resize(input=y2,
                            size=y1.shape[-2:],
                            mode="bilinear" if up >= 1 else "area",
                            align_corners=False if up else None)
            
            gate = self.Sigmoid(a)
            y = gate * y1 + (1-gate) * y2
            # ys.append(y)
            ys.append(y1)
            
        return tuple(ys)
        
    
implemented_backbones = [DinoBackBone.__class__.__name__,
                         SamBackBone.__class__.__name__,
                         ResNetBackBone.__class__.__name__,
                         LadderBackbone.__class__.__name__,]
        
        