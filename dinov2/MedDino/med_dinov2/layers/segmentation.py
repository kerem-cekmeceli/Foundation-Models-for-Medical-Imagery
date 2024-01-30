import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from mmseg.ops import resize



class SegHeadBase(nn.Module, ABC):
    def __init__(self, embedding_sz, num_classses, 
                 n_concat,
                 interp_fact,
                 input_transform,
                 in_index,
                 resize_factors,
                 align_corners) -> None:
        super().__init__()
        
        self.embedding_sz=embedding_sz
        self.num_classes=num_classses
        self.interp_fact=interp_fact
        
        if input_transform == 'resize_concat':
            self.n_concat=n_concat
        else:
            self.n_concat=1
        
        self.input_dim=embedding_sz*self.n_concat
        
        self.input_transform = input_transform
        self.in_index=in_index
        self.resize_factors=resize_factors
        self.align_corners=align_corners
    
    @abstractmethod
    def compute_logits(self, x):
        pass
    
    def _transform_inputs(self, 
                          inputs : list[torch.Tensor]):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        
        if self.in_index is None:
            self.in_index = [i for i in range(len(inputs))]

        if self.input_transform == "resize_concat":
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
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
        
    
    def forward(self, inputs):
        # Concat list of inputs 
        x = self._transform_inputs(inputs)
        
        # segmentation method (assigns logits to each patch)
        logits = self.compute_logits(x)  # [B, N_class, H0, W0]
        
        if self.interp_fact is None:
            return logits
        else:
            # Interpolate to get pixel logits frmo patch logits
            pix_logits = resize(input=logits,
                                scale_factor=self.interp_fact,
                                mode='bilinear',
                                align_corners=self.align_corners)
            # [B, N_class, H, W]
            
            return pix_logits
    
class ConvHead(SegHeadBase):
    
    def __init__(self, 
                 embedding_sz, 
                 num_classses, 
                 n_concat,
                 interp_fact,
                 input_transform='resize_concat',
                 in_index=None,
                 resize_factors=None,
                 align_corners=False,
                 batch_norm=True, dropout_rat=0.) -> None:
        super().__init__(embedding_sz=embedding_sz, 
                         num_classses=num_classses, 
                         n_concat=n_concat,
                         input_transform=input_transform, 
                         in_index=in_index,
                         resize_factors=resize_factors,
                         align_corners=align_corners,
                         interp_fact=interp_fact,)
        
        if batch_norm:
            self.batch_norm = nn.SyncBatchNorm(self.input_dim)  #@TODO change not sure
        
        if dropout_rat>0:
            self.dropout = nn.Dropout2d(dropout_rat)  # Randomly zero out entire channels
        else:
            self.dropout = None
        
        self.conv_seg = nn.Conv2d(self.input_dim, num_classses, kernel_size=1)
        
        
    def compute_logits(self, x):
        # Batch norm
        x = self.batch_norm(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Conv
        x = self.conv_seg(x)
        return x
    
        
        
        

