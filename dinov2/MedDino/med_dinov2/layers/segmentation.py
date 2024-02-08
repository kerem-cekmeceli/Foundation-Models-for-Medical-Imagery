import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from mmseg.ops import resize
from typing import Sequence, Union, Optional


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
                self.up = nn.ConvTranspose2d(num_classses , num_classses, kernel_size=out_upsample_fac, stride=out_upsample_fac)
        
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
    
    
class ConvUNet(DecBase):
    def __init__(self, 
                 in_channels: Union[int,Sequence[int]],
                 num_classses: int, 
                 cls_in_channels: Optional[int]=None,
                 in_index: Optional[Union[int,Sequence[int]]]=None,
                 input_transform: Optional[str]=None,
                 in_resize_factors: Optional[Union[int,Sequence[int]]]=None,
                 align_corners: bool=False,
                 dropout_rat:float=0.) -> None:
        
        super().__init__(in_channels=in_channels, 
                         num_classses=num_classses, 
                         cls_in_channels=cls_in_channels, 
                         in_index=in_index, 
                         input_transform=input_transform, 
                         in_resize_factors=in_resize_factors, 
                         align_corners=align_corners, 
                         dropout_rat=dropout_rat)
        
        
    
        
        
        

