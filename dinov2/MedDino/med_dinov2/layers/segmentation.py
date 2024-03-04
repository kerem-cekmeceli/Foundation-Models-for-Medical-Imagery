import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from mmseg.ops import resize
from typing import Sequence, Union, Optional
from torch import functional as F
import math


class DecBase(nn.Module, ABC):
    def __init__(self, 
                 in_channels: Union[int,Sequence[int]],
                 num_classses: int, 
                 cls_in_channels: Optional[int]=None,
                 in_index: Optional[Union[int,Sequence[int]]]=None,
                 input_transform: Optional[str]=None,
                 in_resize_factors: Optional[Union[int,Sequence[int]]]=None,
                 align_corners: bool=False,
                 dropout_rat:float=0.,
                 out_upsample_fac:Optional[int]=None,
                 bilinear:bool=True) -> None:
        """_summary_

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            num_classses (int): _description_
            cls_in_channels (int): number of input channels to the final cls 2d convolution, if None = in_channel
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed. 
            in_resize_factors (Optional[Union[int,Sequence[int]]], optional): _description_. Defaults to None.
            align_corners (bool, optional): if multiple_selct align corners for interpolation. Defaults to False.
            dropout_rat (float, optional): dropout layer for training before cls layer. Defaults to 0..
        """
        super().__init__()
        
        self._init_inputs(in_channels, in_index, input_transform, in_resize_factors)
        
        assert num_classses > 0
        self.num_classes=num_classses
        
        if cls_in_channels is None:
            self.cls_in_channels = self.in_channels
        else:
            assert cls_in_channels > 0
            self.cls_in_channels = cls_in_channels
        
        # Interpolation, make all the same dim for resize_concat
        self.align_corners=align_corners
        
        # Classification final conv layer
        assert dropout_rat>=0
        if dropout_rat>0:
            self.dropout = nn.Dropout2d(dropout_rat)  # Randomly zero out entire channels
        else:
            self.dropout = None
        
        self.conv_seg = nn.Conv2d(self.cls_in_channels, num_classses, kernel_size=1)
        
        # Upsampling / Transposed convloution shape matching
        if out_upsample_fac is not None:
            assert out_upsample_fac > 1
            self.out_upsample_fac=out_upsample_fac
            
            self.bilinear = bilinear
            if bilinear:
                self.up = nn.Upsample(scale_factor=out_upsample_fac, mode='bilinear', align_corners=False)
            else:
                assert isinstance(out_upsample_fac, int)
                self.up = nn.ConvTranspose2d(num_classses , num_classses, kernel_size=out_upsample_fac, stride=out_upsample_fac)
        self.out_upsample_fac = out_upsample_fac
        
    def _init_inputs(self, in_channels, in_index, input_transform, in_resize_factors):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform
        """
        # Verify input transform
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        
        # Input channel dimensions
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        assert isinstance(in_channels, (list, tuple))
        self.nb_inputs=len(in_channels)
        
        # Chosen channel indexes
        if in_index is None:
            in_index = [i for i in range(len(in_channels))]
        else:
            if isinstance(in_index, int):
                in_index = [in_index]
        assert isinstance(in_index, (list, tuple))
        assert max(in_index) < len(in_channels)
        self.in_index = in_index
        
        # Input Resize factors
        if in_resize_factors is not None:
            if isinstance(in_resize_factors, int):
                in_resize_factors = [in_resize_factors]
            assert isinstance(in_resize_factors, (list, tuple))
            assert len(in_resize_factors) == len(in_channels)
        self.in_resize_factors = in_resize_factors
            
        # Assign the number of input channels
        if input_transform == 'resize_concat':
            self.in_channels = sum(in_channels)
        else:
            self.in_channels = in_channels

    
    def _transform_inputs(self, 
                          inputs : list[torch.Tensor]):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        
        # Verify the input length
        assert self.nb_inputs == len(inputs), f'input len:{len(inputs)}, expected: {self.nb_inputs}'
        
        # accept lists (for cls token)
        input_list = []
        for x in inputs:
            if isinstance(x, list):
                input_list.extend(x)
            else:
                input_list.append(x)
        inputs = input_list
        
        # an image descriptor can be a local descriptor with resolution 1x1
        for i, x in enumerate(inputs):
            if len(x.shape) == 2:
                inputs[i] = x[:, :, None, None]
                
        # select indices
        inputs = [inputs[i] for i in self.in_index]
        
        # Resizing the inputs
        if self.in_resize_factors is not None:
            # print("before", *(x.shape for x in inputs))
            inputs = [
                resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                for x, f in zip(inputs, self.in_resize_factors)
            ]
            # print("after", *(x.shape for x in inputs))
        
        # Input transform specific
        if self.input_transform == "resize_concat":
            # Make all inputs the same shape aas the first one
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index[0]]

        return inputs
    
    
    def cls_segmentation(self, feat):
        """Classify each pixel."""
        
        if self.dropout is not None:
            feat = self.dropout(feat)
            
        output = self.conv_seg(feat)
        return output
    
    
    @abstractmethod
    def compute_feats(self, x):
        pass
        
        
    def forward(self, inputs):
        # Concat list of inputs 
        x = self._transform_inputs(inputs)
        
        # segmentation method acting on patch features
        feats = self.compute_feats(x)  # [B, N_class, H0, W0]
        
        # Classify segmentation
        cls_out = self.cls_segmentation(feats)
        
        # Apply output upsampling / transposed_conv
        if self.out_upsample_fac is not None:
            cls_out = self.up(cls_out)
        
        return cls_out
        
    
class ConvHeadLinear(DecBase):
    
    def __init__(self, 
                 in_channels: Union[int,Sequence[int]],
                 num_classses: int, 
                 in_index: Optional[Union[int,Sequence[int]]]=None,
                 in_resize_factors: Optional[Union[int,Sequence[int]]]=None,
                 align_corners: bool=False,
                 dropout_rat:float=0.,
                 out_upsample_fac: Optional[int]=None,
                 bilinear:bool=True,) -> None:
        super().__init__(in_channels=in_channels,
                         cls_in_channels=None,
                         num_classses=num_classses, 
                         in_index=in_index,
                         input_transform='resize_concat',
                         in_resize_factors=in_resize_factors,
                         align_corners=align_corners,
                         dropout_rat=dropout_rat,
                         out_upsample_fac=out_upsample_fac,
                         bilinear=bilinear)
        
        self.batch_norm = nn.SyncBatchNorm(self.in_channels)  
        
    def compute_feats(self, x):
        # Batch norm
        x = self.batch_norm(x)
        
        return x
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ConvBlk(nn.Module):
    def __init__(self, 
                 in_channel:int,
                 out_channel:int,
                 kernel_size:int,
                 padding:Union[str, int]=0,
                 batch_norm:bool=True,
                 non_linearity:Union[str, nn.Module]='ReLU',
                 recurrent:bool=False,
                 recursion_steps:int=4,
                 **kwargs) -> None:
        super().__init__()
        
        assert in_channel>1 and out_channel>1
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, **kwargs)
        
        self.do_batch_norm = batch_norm
        
        self.recurrent = recurrent
        if self.recurrent:
            assert recursion_steps>=2
            self.recursion_steps = recursion_steps
            
            assert in_channel == out_channel, f"For RCL in_channel and out_channel can't be different but got {in_channel} and {out_channel}"
            self.rec_conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, **kwargs)
            
        if batch_norm:
            if self.recurrent:
                self.batch_norm = nn.ModuleList([nn.SyncBatchNorm(out_channel) for i in range(recursion_steps)])
            else:
                self.batch_norm = nn.SyncBatchNorm(out_channel)
        
        if isinstance(non_linearity, str):
            if non_linearity == 'ReLU':
                self.nonlin = nn.ReLU()
            elif non_linearity == 'Tanh':
                self.nonlin = nn.Tanh()
            elif non_linearity == 'GELU':
                self.nonlin = nn.GELU()
            elif non_linearity == 'Sigmoid':
                self.nonlin = nn.Sigmoid()
            else:
                raise Exception( f'Text non-linearity {non_linearity} is not supported')
        else:
            self.nonlin = non_linearity
            
            
    def forward_recurrent(self, x):
        rec_x = x
        rec_out = self.rec_conv(rec_x)
        
        for i in range(self.recursion_steps):
            if i==0:
                z = self.conv(x)
            else:
                z = self.conv(x) + rec_out
            
            x = self.nonlin(z)
            x = self.batch_norm[i](x)
            
        return x
        
    # BN after ReLU gives better performance
    def forward_ff(self, x):
        x = self.conv(x)
        
        x = self.nonlin(x)
        
        if self.do_batch_norm:
            x = self.batch_norm(x)
            
        return x
        
            
    def forward(self, x):
        if self.recurrent:
            y = self.forward_recurrent(x)
        else:
            y = self.forward_ff(x)
        return y
      

class NConv(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 kernel_sz:Union[int, Sequence[int]],
                 mid_channels:Optional[int]=None,
                 res_con:bool=False, 
                 res_con_interv:Optional[int]=None,
                 skip_first_res_con:bool=False,
                 nb_convs:int=3,
                 batch_norm:Union[bool, Sequence[bool]]=True,
                 non_linearity:Union[str, nn.Module]='ReLU',
                 padding:Union[str, int, Sequence[str], Sequence[int]]='same',
                 recurrent:bool=False,
                 recursion_steps:int=4,
                 ) -> None:
        super().__init__()
        assert nb_convs>0
        self.nb_convs = nb_convs
        
        if not mid_channels:
            mid_channels = out_channels
            
        # Residual connections    
        self.res_con=res_con
        self.do_res_con_first = (in_channels == mid_channels) and not skip_first_res_con
        self.skipped_res_cons_from_beg = 0 if self.do_res_con_first else 1
        self.do_res_con_last = mid_channels == out_channels
        
        if self.res_con:
            if res_con_interv is None:
                res_con_interv = nb_convs
                res_con_interv = res_con_interv if self.do_res_con_first else res_con_interv-1
                assert res_con_interv > 0, "First res con is skipped, need 1 more conv blk"
                res_con_interv = res_con_interv if self.do_res_con_last else res_con_interv-1
                assert res_con_interv > 0, "When mid_channles != out_channels last res con can't be made need 1 more conv blk"
                
            # assert nb_convs>=2, 'Need at least 2 convolutions for a resifual connection'
            # assert mid_channels == out_channels, \
            #     f'mid channels and out_channels should be the same for residual connections, {mid_channels}!={out_channels}'
            
            assert res_con_interv > 0, f'res_con_interv must be >0 but got {res_con_interv}'
            offset = 0
            offset = offset if self.do_res_con_first else offset+1
            offset = offset if self.do_res_con_last else offset+1
            assert nb_convs-offset > 0, f'Insufficient nb of conv blks for residual connections, need {offset-nb_convs} more'
            assert res_con_interv <= nb_convs-offset
            self.res_con_interv = res_con_interv
        
        
        kernel_sz = self.get_attr_list(kernel_sz)
        self.kernel_sz = kernel_sz
        batch_norm = self.get_attr_list(batch_norm)
        self.batch_norm = batch_norm
        non_linearity = self.get_attr_list(non_linearity)
        self.non_linearity = non_linearity
        padding = self.get_attr_list('same')
        self.padding = padding
        
        conv_blk_list = [ConvBlk(in_channel=in_channels,
                                 out_channel=mid_channels,
                                 kernel_size=kernel_sz[0],
                                 batch_norm=batch_norm[0],
                                 non_linearity=non_linearity[0],
                                 padding=padding[0],
                                 recurrent=recurrent and in_channels==mid_channels,
                                 recursion_steps=recursion_steps)]
        if nb_convs>2:
            for i in range(1, nb_convs-1):
                conv_blk_list.append(ConvBlk(in_channel=mid_channels,
                                             out_channel=mid_channels,
                                             kernel_size=kernel_sz[i],
                                             batch_norm=batch_norm[i],
                                             non_linearity=non_linearity[i],
                                             padding=padding[i],
                                             recurrent=recurrent,
                                             recursion_steps=recursion_steps))
                
        conv_blk_list.append(ConvBlk(in_channel=mid_channels,
                                    out_channel=out_channels,
                                    kernel_size=kernel_sz[-1],
                                    batch_norm=batch_norm[-1],
                                    non_linearity=non_linearity[-1],
                                    padding=padding[-1],
                                    recurrent=recurrent and mid_channels==out_channels,
                                    recursion_steps=recursion_steps))
        
        self.conv_blks = nn.ModuleList(conv_blk_list)
        
        
    def get_attr_list(self, attr):
        if isinstance(attr, list):
            assert len(attr) == self.nb_convs
        else:
            attr = [attr]*self.nb_convs
        return attr
    
    
    def forward(self, x):        
        # Fwd Blocks without residual connections
        for blk in self.conv_blks[:self.skipped_res_cons_from_beg]:
            # Forward conv blk
            x = blk(x)
        
        # Fwd Blocks with residual connections
        blks_res = self.conv_blks[self.skipped_res_cons_from_beg:]
        for i, blk in enumerate(blks_res):
            if self.res_con:
                if i%self.res_con_interv==0:
                    x_residual = x
                    
            # Forward conv blk
            x = blk(x)
            
            if self.res_con:
                if (i+1)%self.res_con_interv == 0:
                    if i<len(blks_res)-1 or self.do_res_con_last:
                        x = x + x_residual     
        return x
        
            
class UpRes(nn.Module):
    """Upscaling by f, channel depth reduction by f then n times conv"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 bilinear=False, 
                 res_con=True,
                 res_con_interv=1,
                 skip_first_res_con=False,
                 fact=2, 
                 nb_convs=2,
                 kernel_size=3,
                 batch_norm=True,
                 non_linearity='ReLU',
                 recurrent:bool=False,
                 recursion_steps:int=4,):
        super().__init__()
        assert fact>1 and isinstance(fact, int)
        assert nb_convs > 1
        
        self.fact=fact
        
        # If to use residual connections 
        self.res_con=res_con
        if res_con:
            self.res_con_interv = res_con_interv

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=fact, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_channels , in_channels // fact, kernel_size=fact, stride=fact),)
                                    #nn.ReLU())
            
        self.conv_xn = NConv(in_channels = in_channels if bilinear else in_channels//fact,
                                out_channels=out_channels,
                                mid_channels=in_channels//fact,
                                kernel_sz=kernel_size,
                                nb_convs=nb_convs,
                                batch_norm=batch_norm,
                                non_linearity=non_linearity,
                                res_con=res_con,
                                res_con_interv=res_con_interv,
                                skip_first_res_con=skip_first_res_con,
                                recurrent=recurrent,
                                recursion_steps=recursion_steps)
        
    def forward(self, x):
        return self.conv_xn(self.up(x))
    

class UpUNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 in_channels_cat,
                 out_channels, 
                 bilinear=False, 
                 res_con=True,
                 res_con_interv=1,
                 skip_first_res_con=False,
                 fact=2, 
                 fact_cat_inp=2,
                 nb_convs=2,
                 kernel_size=3,
                 batch_norm=True,
                 non_linearity='ReLU',
                 recurrent:bool=False,
                 recursion_steps:int=4,
                 resnet_cat_inp_upscaling:bool=True):
        super().__init__()
        assert fact>1 and isinstance(fact, int)
        assert fact_cat_inp>1 and isinstance(fact, int)
        assert nb_convs > 1
        
        self.fact=fact
        self.fact_cat_inp=fact_cat_inp
        
        assert in_channels // fact_cat_inp >= 1
        
        # If to use residual connections 
        self.res_con=res_con
        if res_con:
            self.res_con_interv = res_con_interv

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=fact, mode='bilinear', align_corners=True)
            if not resnet_cat_inp_upscaling:
                self.up_cat = nn.Upsample(scale_factor=fact, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_channels , in_channels // fact, kernel_size=fact, stride=fact),)
                                    #nn.ReLU())
            if not resnet_cat_inp_upscaling:
                self.up_cat = nn.Sequential(nn.ConvTranspose2d(in_channels_cat , in_channels // fact, kernel_size=fact_cat_inp, stride=fact_cat_inp),)
                                        #nn.ReLU())
        if resnet_cat_inp_upscaling:
            nb_ups = math.log(fact_cat_inp) / math.log(fact)
            assert nb_ups == int(nb_ups)
            nb_ups = int(nb_ups)
            
            inp_ups = [UpRes(in_channels=in_channels_cat // (fact**i), 
                            out_channels=in_channels_cat // fact // (fact**i), 
                            bilinear=bilinear, 
                            res_con=res_con,
                            res_con_interv=res_con_interv,
                            skip_first_res_con=skip_first_res_con,
                            fact=fact, 
                            nb_convs=nb_convs,
                            kernel_size=kernel_size,
                            batch_norm=kernel_size,
                            non_linearity=non_linearity,
                            recurrent=recurrent,
                            recursion_steps=recursion_steps,) for i in range(nb_ups)]
            
            assert in_channels_cat // fact // (fact**(nb_ups - 1)) == in_channels // fact
            self.up_cat = nn.Sequential(*inp_ups)
       
            
        self.conv_xn = NConv(in_channels = 2*in_channels if bilinear else 2*in_channels//fact,
                                out_channels=out_channels,
                                mid_channels=in_channels//fact,
                                kernel_sz=kernel_size,
                                nb_convs=nb_convs,
                                batch_norm=batch_norm,
                                non_linearity=non_linearity,
                                res_con=res_con,
                                res_con_interv=res_con_interv,
                                skip_first_res_con=skip_first_res_con,
                                recurrent=recurrent,
                                recursion_steps=recursion_steps)
        
    def forward(self, x, x_cat):
        # Do the transposed convs
        y_cat = self.up_cat(x_cat)
        _y = self.up(x)
        assert _y.shape == y_cat.shape
        
        # Cat the inputs
        _x = torch.cat([y_cat, _y], dim=1)
        
        # Perform the convolutions
        return self.conv_xn(_x)
    
class UpNetHeadBase(DecBase):
    def __init__(self, 
                 in_channels: Union[int,Sequence[int]],
                 num_classses: int, 
                 in_index: Optional[Union[int,Sequence[int]]]=None,
                 in_resize_factors: Optional[Union[int,Sequence[int]]]=None,
                 align_corners: bool=False,
                 dropout_rat_cls_seg:float=0.,
                 nb_up_blocks:int=2,
                 upsample_facs: Union[int, Sequence[int]]=2,
                 bilinear:Union[bool, Sequence[bool]]=True,
                 conv_per_up_blk:Union[int,Sequence[int]]=2,
                 res_con:Union[bool,Sequence[bool]]=True,
                 res_con_interv:Optional[Union[int,Sequence[int]]]=1,
                 skip_first_res_con:Union[bool,Sequence[bool]]=False,
                 recurrent:Union[bool,Sequence[bool]]=False,
                 recursion_steps:Union[int,Sequence[int]]=4,
                 inp_transform:str='resize_concat') -> None:
        
        # Number of up layers in the UNet architecture
        assert nb_up_blocks>0
        self.nb_up_blocks = nb_up_blocks
        
        # Upsampling ratio of each Up layer
        upsample_facs = self.get_attr_list(upsample_facs, cond=lambda a: a>0)
        self.upsample_facs = upsample_facs
        
        # If to use bilinear interp or transposed conv for each Up layer
        bilinear = self.get_attr_list(bilinear)
        self.bilinear_ups = bilinear
        
        # Number of convloutional blocks per each up layer
        conv_per_up_blk = self.get_attr_list(conv_per_up_blk, cond=lambda a: a>0)
        self.conv_per_up_blk = conv_per_up_blk
        
        # If to use residual connections for the conv layers of the Up blocks
        res_con = self.get_attr_list(res_con)
        self.res_con = res_con
        
        # interval between residual connections for the conv layers of the Up blocks (summation)
        res_con_interv = self.get_attr_list(res_con_interv, cond=lambda a: True if a is None else a>0)
        self.res_con_interv = res_con_interv
        
        # If to skip the first residual connection for the conv layers of the Up blocks
        skip_first_res_con = self.get_attr_list(skip_first_res_con)
        self.skip_first_res_con = skip_first_res_con
        
        # If to use recursive conv blocks
        recurrent = self.get_attr_list(recurrent)
        self.recurrent = recurrent
        
        # If recursive conv blocks are used how many recursion steps
        recursion_steps = self.get_attr_list(recursion_steps)
        self.recursion_steps = recursion_steps
        
        # Set kernel size to 3
        self.kernel_sz = 3
        
        ups = self._init_up_layers(in_channels=in_channels)
        assert self.last_out_ch >= num_classses, \
            f"Too many ch size reduction, input to seg_cls: {self.last_out_ch}, but num class: {num_classses} "
        
        super().__init__(in_channels=in_channels,
                         cls_in_channels=self.last_out_ch,  # Assigned in "_init_up_layers"
                         num_classses=num_classses, 
                         in_index=in_index,
                         input_transform=inp_transform,
                         in_resize_factors=in_resize_factors,
                         align_corners=align_corners,
                         dropout_rat=dropout_rat_cls_seg,
                         out_upsample_fac=None,  # Not used
                         bilinear=True)  # Not used
        
        self.ups = ups
    
    @abstractmethod            
    def _init_up_layers(self, in_channels, *args, **kwargs):
        """
            Assign the following fields: self.tot_upsample_fac:int, self.last_out_ch:int 
            Returns ups:nn.ModuleList
        """
        pass
        
    def get_attr_list(self, attr, cond=None):
        if isinstance(attr, list):
            assert len(attr) == self.nb_up_blocks
        else:
            attr = [attr]*self.nb_up_blocks
            
        if cond is not None:
            for a in attr:
                assert cond(a)
        return attr
    
    @abstractmethod
    def compute_feats(self, x):
        pass
    
class ResNetHead(UpNetHeadBase):
    def __init__(self, 
                 in_channels: Union[int,Sequence[int]],
                 num_classses: int, 
                 in_index: Optional[Union[int,Sequence[int]]]=None,
                 in_resize_factors: Optional[Union[int,Sequence[int]]]=None,
                 align_corners: bool=False,
                 dropout_rat_cls_seg:float=0.,
                 nb_up_blocks:int=2,
                 upsample_facs: Union[int, Sequence[int]]=2,
                 bilinear:Union[bool, Sequence[bool]]=True,
                 conv_per_up_blk:Union[int,Sequence[int]]=2,
                 res_con:Union[bool,Sequence[bool]]=True,
                 res_con_interv:Optional[Union[int,Sequence[int]]]=1,
                 skip_first_res_con:Union[bool,Sequence[bool]]=False,
                 recurrent:Union[bool,Sequence[bool]]=False,
                 recursion_steps:Union[int,Sequence[int]]=4,
                 ) -> None:
        
        super().__init__(in_channels,
                         num_classses, 
                         in_index,
                         in_resize_factors,
                         align_corners,
                         dropout_rat_cls_seg,
                         nb_up_blocks,
                         upsample_facs,
                         bilinear,
                         conv_per_up_blk,
                         res_con,
                         res_con_interv,
                         skip_first_res_con,
                         recurrent,
                         recursion_steps,
                         inp_transform='resize_concat')
                
    def _init_up_layers(self, in_channels, *args, **kwargs)->nn.ModuleList:
        tot_upsample_fac = 1
        modules = []
        last_out_ch = sum(in_channels) if isinstance(in_channels, list) else in_channels
        for i in range(self.nb_up_blocks):
            f = self.upsample_facs[i]
            tot_upsample_fac = tot_upsample_fac*f
            modules.append(UpRes(in_channels=last_out_ch, 
                              out_channels=last_out_ch//f, 
                              bilinear=self.bilinear_ups[i], 
                              fact=f, 
                              nb_convs=self.conv_per_up_blk[i],
                              kernel_size=self.kernel_sz,
                              res_con=self.res_con[i],
                              res_con_interv=self.res_con_interv[i],
                              skip_first_res_con=self.skip_first_res_con[i],
                              recurrent=self.recurrent[i],
                              recursion_steps=self.recursion_steps[i]))
            last_out_ch = last_out_ch // f
            
        self.tot_upsample_fac = tot_upsample_fac
        self.last_out_ch = last_out_ch 
        return nn.ModuleList(modules)
        
        
    def compute_feats(self, x):
        for i, up in enumerate(self.ups):
            x = up(x)
        return x
    
    
    
class UNetHead(UpNetHeadBase):
    def __init__(self, 
                 in_channels: Sequence[int],
                 num_classses: int, 
                 in_index: Optional[Union[int,Sequence[int]]]=None,
                 in_resize_factors: Optional[Union[int,Sequence[int]]]=None,
                 align_corners: bool=False,
                 dropout_rat_cls_seg:float=0.,
                 nb_up_blocks:int=2,
                 upsample_facs: Union[int, Sequence[int]]=2,
                 bilinear:Union[bool, Sequence[bool]]=True,
                 conv_per_up_blk:Union[int,Sequence[int]]=2,
                 res_con:Union[bool,Sequence[bool]]=True,
                 res_con_interv:Optional[Union[int,Sequence[int]]]=1,
                 skip_first_res_con:Union[bool,Sequence[bool]]=False,
                 recurrent:Union[bool,Sequence[bool]]=False,
                 recursion_steps:Union[int,Sequence[int]]=4,
                 resnet_cat_inp_upscaling:Union[bool,Sequence[bool]]=True,
                 ) -> None:
        
        for in_ch in in_channels[1:]:
            assert in_ch == in_channels[0], "All inputs must have the same nb of channels"
            
        self.resnet_cat_inp_upscaling = [resnet_cat_inp_upscaling]*nb_up_blocks
        
        super().__init__(in_channels,
                         num_classses, 
                         in_index,
                         in_resize_factors,
                         align_corners,
                         dropout_rat_cls_seg,
                         nb_up_blocks,
                         upsample_facs,
                         bilinear,
                         conv_per_up_blk,
                         res_con,
                         res_con_interv,
                         skip_first_res_con,
                         recurrent,
                         recursion_steps,
                         inp_transform='multiple_select')
        
        self.unet_init_conv = NConv(in_channels = in_channels[0],
                                    out_channels=in_channels[0],
                                    mid_channels=in_channels[0],
                                    kernel_sz=self.kernel_sz,
                                    nb_convs=self.conv_per_up_blk[0],
                                    # batch_norm=True,  # default True
                                    # non_linearity=non_linearity,  # default ReLU
                                    res_con=res_con,
                                    res_con_interv=res_con_interv,
                                    skip_first_res_con=skip_first_res_con,
                                    recurrent=recurrent,
                                    recursion_steps=recursion_steps)
                
    def _init_up_layers(self, in_channels, *args, **kwargs):
        tot_upsample_fac = 1
        last_out_ch = in_channels[0]
        modules = []
        
        for i in range(self.nb_up_blocks):
            f = self.upsample_facs[i]
            tot_upsample_fac = tot_upsample_fac*f
            modules.append(UpUNet(in_channels=last_out_ch, 
                                  in_channels_cat= in_channels[i],
                                  out_channels=last_out_ch//f, 
                                  bilinear=self.bilinear_ups[i], 
                                  fact=f, 
                                  fact_cat_inp=f**(i+1),
                                  nb_convs=self.conv_per_up_blk[i],
                                  kernel_size=self.kernel_sz,
                                  res_con=self.res_con[i],
                                  res_con_interv=self.res_con_interv[i],
                                  skip_first_res_con=self.skip_first_res_con[i],
                                  recurrent=self.recurrent[i],
                                  recursion_steps=self.recursion_steps[i],
                                  resnet_cat_inp_upscaling=self.resnet_cat_inp_upscaling[i]))
            last_out_ch = last_out_ch // f
            
        self.tot_upsample_fac = tot_upsample_fac
        self.last_out_ch = last_out_ch 
        return  nn.ModuleList(modules)
        
        
    def compute_feats(self, x):
        x_up = self.unet_init_conv(x[0])
        x_up = self.ups[0](x_up, x[1])
        
        offset = 2
        for i, up in enumerate(self.ups[1:]):
            x_up = up(x_up, x[i+offset])
        return x_up
    
        
        


