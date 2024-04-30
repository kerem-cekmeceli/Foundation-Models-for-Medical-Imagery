import torch
from OrigModels.MAE.models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

def get_mae_bb(bb_size, checkpoint=None, enc_only=True):
    if bb_size=='base':
        model = mae_vit_base_patch16()
    elif bb_size=='large':
        model = mae_vit_large_patch16()
    elif bb_size=='huge':
        model = mae_vit_huge_patch14()
    else:
        ValueError(f'Undefined model size: {bb_size}')
        
    if checkpoint is not None:
        cp = torch.load(checkpoint)
        model.load_state_dict(cp['model'], strict=False)
        
    if enc_only:
        no_grads = ["decoder_embed", "mask_token", "decoder_blocks", "decoder_norm"]
        for layer in no_grads:
            attr = getattr(model, layer)
            if isinstance(attr, torch.nn.Parameter):
                attr.requires_grad=False
            else:
                for param in attr.parameters():
                    param.requires_grad=False
    return model
        
        
        