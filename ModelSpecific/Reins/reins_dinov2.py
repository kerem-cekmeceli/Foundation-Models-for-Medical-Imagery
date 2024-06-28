from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from OrigModels.DinoV2.dinov2.models.vision_transformer import DinoVisionTransformer
import torch
from torch import nn
from typing import Union, Sequence

@BACKBONES.register_module()
class ReinsDinoVisionTransformer(nn.Model):
    def __init__(
        self,
        dino_bb:DinoVisionTransformer,
    ):
        super().__init__()
        self.dino_bb = dino_bb
        self.reins: Reins = Reins(num_layers=len(self.dino_bb.blocks),
                                  embed_dims=self.dino_bb.embed_dim,
                                  patch_size=self.dino_bb.patch_size)
        
        for p in self.dino_bb.parameters():
            p.requires_grad = False
        self.dino_bb.eval()

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)
    
    def get_intermediate_layers(self, 
                                x: torch.Tensor,
                                n: Union[int, Sequence] = 1, 
                                reshape: bool = True,
                                return_class_token: bool = False,
                                norm=True,):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            # Dino ViT block
            x = blk(x)
            
            # Apply Rein
            x = self.reins.forward(
                x,
                i,
                batch_first=True,
                has_cls_token=True,
            )
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        
        # Normalize the outputs
        if norm:
            outputs = [self.norm(out) for out in outputs]
            
        class_tokens = [out[:, 0] for out in outputs]  # list of cls_token outputs (of length n)
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]  # Discard register tokens
        
        # Each oup is of shape: [B, N, embed_dim] --reshape--> [B, embed_dim, h_patches, w_patches]
        if reshape:
            B, _, h, w = x.shape 
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        
        for name, param in self.named_parameters():
            if "reins" not in name:
                param.requires_grad = False
                
            

    # def state_dict(self, destination, prefix, keep_vars):
    #     state = super().state_dict(destination, prefix, keep_vars)
    #     keys = [k for k in state.keys() if "rein" not in k]
    #     for key in keys:
    #         state.pop(key)
    #         if key in destination:
    #             destination.pop(key)
    #     return state