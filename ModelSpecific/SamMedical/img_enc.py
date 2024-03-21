import torch
from torch import nn
from OrigModels.SAM.segment_anything import sam_model_registry

from OrigModels.SAM.segment_anything.modeling.image_encoder import PatchEmbed
from typing import Optional, Union, Tuple, Sequence

class ImageEncoderViTFeats(nn.Module):
    def __init__(self, 
                 img_size:int,
                 patch_embed:PatchEmbed,
                 pos_embed: Optional[nn.Parameter],
                 blocks:nn.ModuleList,
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.n_blocks = len(self.blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
            
        return x.permute(0, 3, 1, 2)
    
    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        assert max(blocks_to_take)<total_block_len
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        # output: [B, C, H, W] C = 1280 ViTH and 256 after the neck (if used)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
        
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  
        # norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """_summary_

        Args:
            x (torch.Tensor): [B, nc(r, G, B), h, w]
            n (Union[int, Sequence], optional): Layers or n last layers to take. Defaults to 1.
            reshape (bool, optional): If True reshapes the output patches respecting the W, H order in Defaults to True.
            return_class_token (bool, optional): also returns the cls token among the pathch tokens. Defaults to False.
            norm (bool, optional): Applies normalization to the ooutputs before returning. Defaults to True.

        Returns:
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]: 
        """
    
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        # outputs is a list of length n
        
        # if norm:
        #     outputs = [self.norm(out) for out in outputs]
        
        class_tokens = [out[:, 0] for out in outputs]  # list of cls_token outputs (of length n)
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]  # Discard register tokens
        
        
        return tuple(outputs)
    
    

def get_sam_vit_backbone(bb_size, sam_checkpoint=None):
    if bb_size == 'base':
        model_type = 'vit_b'
    elif bb_size == 'large':
        model_type = 'vit_l'
    elif bb_size == 'huge':
        model_type = 'vit_h'
    else:
        ValueError(f'Model size {bb_size} is not supported for SAM')
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    bb_model = ImageEncoderViTFeats(img_size=sam.image_encoder.img_size,
                                    patch_embed=sam.image_encoder.patch_embed,
                                    pos_embed=sam.image_encoder.pos_embed,
                                    blocks=sam.image_encoder.blocks)
    return bb_model
                        
    
        