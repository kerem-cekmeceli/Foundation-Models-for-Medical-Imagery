import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from mmseg.ops import resize


class SegHeadBase(nn.Module, ABC):
    def __init__(self, embedding_sz, num_classses, 
                 input_transform='resize_concat',
                 in_index=None) -> None:
        super().__init__()
        
        self.embedding_sz = embedding_sz
        self.num_classes = num_classses
        self.input_transform = input_transform
        self.in_index=in_index
    
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
        x = self._transform_inputs(inputs)
        logits = self.compute_logits(x)
        return logits
    
class ConvHead(SegHeadBase):
    
    def __init__(self, embedding_sz, num_classses, 
                 input_transform='resize_concat',
                 in_index=None,
                 batch_norm=True, dropout_rat=0.) -> None:
        super().__init__(embedding_sz, num_classses, 
                         input_transform, in_index)
        
        if batch_norm:
            self.batch_norm = nn.SyncBatchNorm(embedding_sz)
        
        if dropout_rat>0:
            self.dropout = nn.Dropout2d(dropout_rat)  # Randomly zero out entire channels
        else:
            self.dropout = None
        
        self.conv_seg = nn.Conv2d(embedding_sz, num_classses, kernel_size=1)
        
        
    def compute_logits(self, x):
        # Batch norm
        x = self.bn(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Conv
        x = self.conv_seg(x)
        return x
    
        
        
        

