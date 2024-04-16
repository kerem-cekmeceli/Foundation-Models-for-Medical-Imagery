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
                 neck:Optional[nn.Sequential]=None
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.n_blocks = len(self.blocks)
        self.neck=neck
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        
        x = x.permute(0, 3, 1, 2) 
        
        if self.neck is not None:
            x = self.neck(x)
            
        return x
        
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        
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
                output.append(x.permute(0, 3, 1, 2).contiguous() if self.neck is None else self.neck(x.permute(0, 3, 1, 2).contiguous()))
        # output: [B, C, H, W] C = 1280 ViTH and 256 after the neck (if used)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        
        # if norm:
        #     outputs = [self.norm(out) for out in outputs]
        
        return tuple(output)
    
    
def get_sam_vit_backbone(bb_size, sam_checkpoint=None, apply_neck=False):
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
                                    blocks=sam.image_encoder.blocks,
                                    neck=sam.image_encoder.neck if apply_neck else None)
    return bb_model
                        
    
        