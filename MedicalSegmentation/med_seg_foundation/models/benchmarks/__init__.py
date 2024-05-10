from .unet import UNet
from .SwinUnet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
implemented_models = [UNet.__name__, SwinTransformerSys.__name__]