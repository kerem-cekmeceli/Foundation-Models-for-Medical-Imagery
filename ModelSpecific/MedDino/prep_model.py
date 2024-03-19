# import os
# import sys
# par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# REPO_PATH = par_dir
# sys.path.insert(1, REPO_PATH)
# print(REPO_PATH)


import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

from OrigModels.DinoV2.dinov2.eval.segmentation import models

import os
import cv2
from matplotlib import pyplot as plt

from mmseg.models import build_backbone
from datetime import datetime

from pathlib import Path
import mmcv
import numpy as np

# from OrigDino.dinov2.hub.backbones import dinov2_vits14
from OrigModels.DinoV2.dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

import urllib
from mmcv.runner import load_checkpoint

from PIL import Image
from torchvision import transforms

import numpy as np
from OrigModels.DinoV2.dinov2.eval.segmentation.utils import colormaps as colormaps

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


# Same exists in dinov2/hub/utils
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        # computesd nb_pads necessary on the left and right for last 2 dim (W, H) in reverse order
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_segmenter(cfg, backbone_model, seg_checkpoint=None):
    model = init_segmentor(cfg, checkpoint=seg_checkpoint)
    # Returns the oup of last 4 layers
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        # The hook is executed before the forward pass, allowing you to inspect or modify the input data.
        # Takes in 2 args, module and input
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
        # Center padding ensures image is a multiple of patch size (input to forward method = dict{x, masks=masks})
    if seg_checkpoint is None:
        model.init_weights()
    return model

def get_bb_name(backbone_sz, ret_arch=False):
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    
    backbone_arch = backbone_archs[backbone_sz]
    backbone_name = f"dinov2_{backbone_arch}"
    
    if ret_arch:
        return backbone_name, backbone_arch
    else:
        return backbone_name

def get_dino_backbone(backbone_name, backbone_cp=None):
    if isinstance(backbone_cp, Path):
        backbone_cp = str(backbone_cp)
        
    # Local checkpoint
    if backbone_cp is not None:
        print(f"Attempting to load the local checkpoint from {backbone_cp}")
        if not Path(backbone_cp).is_file():
            raise Exception(f"File: {backbone_cp} does not exist")
        
        # Instantiate an empty DinoVisionTransformer model for the selected size
        # backbone_model = torch.hub.load('./', backbone_name, source='local', pretrained=False)
        # backbone_model = dinov2_vits14(pretrained=False)
        if backbone_name == 'dinov2_vits14':
            backbone_model = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1,)
        elif backbone_name == 'dinov2_vitb14':
            backbone_model = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1,)
        elif backbone_name == 'dinov2_vitl14':
            backbone_model = vit_large(patch_size=14, img_size=518, block_chunks=0, init_values=1,)
        elif backbone_name == 'dinov2_vitg14':
            backbone_model = vit_giant2(patch_size=14, img_size=518, block_chunks=0, init_values=1,)
        else:
            raise ValueError(f'Unknown backbone name {backbone_name}')

        # Load the checkpoint using torch.load
        checkpoint = torch.load(backbone_cp)

        if 'model_state_dict' in checkpoint:
            # If your model is saved as part of the checkpoint, the entire model is saved (including optimizer, etc.)
            backbone_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If only the model parameters are saved
            backbone_model.load_state_dict(checkpoint)
            
    # Download the checkpoint from github
    else:
        print("Downloading the checkpoint from github")
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    
    return backbone_model

def get_backone_patch_embed_sizes(backbone_name):
    if backbone_name == "dinov2_vits14":
        return 14, 384 
    elif backbone_name == "dinov2_vitb14":
        return 14, 768
    elif backbone_name == "dinov2_vitl14":
        return 14, 1024
    elif backbone_name == "dinov2_vitg14":
        return 14, 1536
    else:
        ValueError(f"backbone name  {backbone_name} is undefined")
        
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    
DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
def get_seg_head_config(backbone_name, head_dataset, head_type, head_sclae_cnt=3, cfg_fld_path=None):
    # Get the conf file for the segmentation head
    if cfg_fld_path is None:
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_config.py"

        cfg_str = load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        
    else:
        backbone_arc = backbone_name.split('_')[-1]
        cfg = f'{cfg_fld_path}/{backbone_arc}/{backbone_name}_{head_dataset}_{head_type}_config.py'
        cfg = mmcv.Config.fromfile(cfg)
    
    if head_type == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:head_sclae_cnt]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])
        
    cfg.backbone_name = backbone_name
    cfg.head_dataset = head_dataset
    cfg.head_type = head_type
    return cfg

def get_seg_model(backbone_model, cfg, cp_fld_path=None, eval=True):
    backbone_name = cfg.backbone_name 
    head_dataset = cfg.head_dataset
    head_type = cfg.head_type
    
    # Load the checkpoint to the segmentation head
    if cp_fld_path is None:
        # Instantiate the empty segmentation head for the selected backbone
        model = create_segmenter(cfg, backbone_model=backbone_model)
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"
        load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    else:
        # load_checkpoint(model, './checkpoints_segHead/dinov2_vits14_voc2012_ms_head.pth', map_location="cpu")
        cp_seg = f'{cp_fld_path}/{backbone_name}_{head_dataset}_{head_type}_head.pth'
        model = create_segmenter(cfg, backbone_model=backbone_model, seg_checkpoint=cp_seg)
        
    model.cuda()
    if eval:
        model.eval()
    return model

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")
    
def load_pil_img_rgb(local_pth=None, url=None):
    assert local_pth is not None or url is not None, 'Need to provide an url or local path to img'
    
    if local_pth is None:
        image = load_image_from_url(url)
        # image.save('./oup_imgs/orig_dino.png')
    else:
        assert Path(local_pth).is_file(), f"File does not exist: {local_pth}"
        image = Image.open(local_pth).convert("RGB")
    return image

def prep_img_tensor(imgs, pad_patch_sz=None, resized_shape=None, norm=True, device='cuda:0'):
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    trf_indv = []
    trf_indv.append(transforms.ToTensor())  # HWC -> CHW in [0, 1]
    
    if resized_shape is not None:
        # img = cv2.resize(img, resized_shape)
        trf_indv.append(transforms.Resize(resized_shape))
    
    trf_indv = transforms.Compose(trf_indv)
    
    img_tensors = []
    for i, img in enumerate(imgs): 
        if isinstance(img, (str, Path)):
            img = Image.open(img)
            
        img = img.convert("RGB")  # RGBA -> RGB  (if single ch -> replicates the same vals for rgb)
        img_tens = trf_indv(img)  # 
        if resized_shape is None:
            if i==0:
                sz = img_tens.shape
            else:
                assert img_tens.shape == sz, \
                    f"Image sizes are not the same ! From 0 to {i-1} shape: {sz}, {i} shape: {img_tens.shape}"
        img_tensors.append(img_tens)
    img_tensors = torch.stack(img_tensors, dim=0)
        
    trf = []
    trf.append(lambda x: 255.0 * x) # scale by 255
    
    if norm:
        trf.append(transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),))
    
    img_tensors = transforms.Compose(trf)(img_tensors).to(device)
    
    if pad_patch_sz is not None:
        assert isinstance(pad_patch_sz, int)
        img_tensors = CenterPadding(pad_patch_sz)(img_tensors)
        
    return img_tensors  # [B, C, H, W] (cuda tensor)

def conv_to_numpy_img(tensor):
    """Converts torch img tensor to numpy array and sets the channel dim as the last

    Args:
        tensor (torch.Tensor): [B, C, H, W]

    Returns:
        np.array: [B, H, W, C]
    """
    img = tensor.cpu().numpy()  # [B, C, H, W]
    img = img.transpose([0, 2, 3, 1])  # [B, H, W, C]
    return img

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    # segmentation_values = colormap_array[segmentation_logits]
    return Image.fromarray(segmentation_values)

def pca_patches(patch_feats, n, scaling=True):
    """Converts the patch features tensosr to numpy array and 

    Args:
        patch_feats (_type_): _description_
        n (_type_): _description_
        scaling (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    reshaped = False
    if len(patch_feats.shape) == 4:
        reshaped = True
        B, ed, h0, w0 = patch_feats.shape
        pf = patch_feats.reshape(B, ed, -1)  # [B, ed, N]
        pf = pf.transpose([0, 2, 1])  # [B, N, ed]
    else:
        pf = np.copy(patch_feats)  # [B, N, ed]
    
    B, N, ed = pf.shape
    pf = pf.reshape([-1, ed])  # patch feats of all patches of all imgs  [B*N, ed]
    
    pf_red = PCA(n_components=n).fit_transform(pf)# Reduced to n PCs  [B*N, n]
    
    if scaling:
        # Scale in range [0, 1]
        pf_red = minmax_scale(pf_red)
    
    if reshaped:  # B, h0, w0, n
        pf_red = pf_red.reshape((B, h0, w0, n)).transpose([0, 3, 1, 2])  # [B, n, h0, w0]
    else:
        pf_red = pf_red.reshape(B, N, n)  # to unsqueeze if Batch dim is 1
    
    return pf_red

def get_pca_patches_plotable(pca_patches):
    """
    Takes in PCA applied patches in range [0, 1] and dim [B, n, h0, w0]
    returns plotable the feature values [B, h0, w0, n] uint8 in range [0, 255]

    Args:
        pca_patches (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Plot the BG PCA result
    pca_plt = pca_patches.transpose([0, 2, 3, 1])  # [B, h0, w0, n]
    pca_plt = pca_plt*255
    pca_plt = pca_plt.astype('uint8')
    return pca_plt

def get_pca_res(patch_tks, bg_thres=0.6, rescale_fac=None):
    res = {}

    # Get the 1-PCs for each img individually to isolate BG
    pca_bgs = []
    for i in range(patch_tks.shape[0]):
        pca_bg = pca_patches(np.expand_dims(patch_tks[i], 0), n=1)  # [B, n, h0, w0]
        pca_bgs.append(pca_bg)
    pca_bg = np.concatenate(pca_bgs, 0)

    # Plot the BG PCA raw result
    res['pca_bg'] = get_pca_patches_plotable(pca_bg)

    # Thresshold the BG PCA to isolate FG objects
    bg_thres = 0.6
    pca_bg_thres = pca_bg.copy()
    fg_mask = pca_bg_thres > bg_thres
    pca_bg_thres[np.logical_not(fg_mask)]=0  # Set the bg to 0

    # Plot the thresholded BG PCA result
    res['pca_bg_thres'] = get_pca_patches_plotable(pca_bg_thres)

    # Extract FG patches
    B, ed, h0, w0 = patch_tks.shape
    input_patches_flat = patch_tks.transpose([0, 2, 3, 1]).reshape(-1, ed)  # [B*h0*w0, ed]
    fg_mask_flat = fg_mask.squeeze(1).ravel()
    obj_patches = np.expand_dims(input_patches_flat[fg_mask_flat, :], 0)  # [1, all_fg_patches, ed]

    # Apply 3-PCA on all FG patches across images
    pca_fg = np.squeeze(pca_patches(obj_patches, n=3), 0)  # [all_fg_patches, 3]

    # Construct the patch grid
    oup = np.zeros([B*h0*w0, 3])
    oup[fg_mask_flat, :] = pca_fg
    oup = oup.reshape([B, h0, w0, 3]).transpose([0, 3, 1, 2])
    res['pca_fg'] = get_pca_patches_plotable(oup)
    
    if rescale_fac is not None:
        assert isinstance(rescale_fac, int) and rescale_fac>0
        def rescale_pca(key):
            B, H0, W0, C = res[key].shape  # Of the patches
            pcas_upscaled = res[key].transpose([1, 2, 3, 0]).reshape(H0, W0, C*B)
            H = int(H0*rescale_fac)
            W = int(W0*rescale_fac)
            pcas_upscaled = cv2.resize(pcas_upscaled, [H, W], interpolation=cv2.INTER_AREA)
            pcas_upscaled = pcas_upscaled.reshape([H, W, C, B]).transpose(3, 0, 1, 2)
            return pcas_upscaled
        
        for key in res.keys():
            res[key] = rescale_pca(key)  
    
    return res

def time_str():
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the date and time as a string
    formatted_time_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    return formatted_time_str

def plot_batch_im(input, fld_pth, file_suffix='', show=False, save=True, 
                  title_prefix=None, suptitle=None):
    assert show or save is not None, "Should either save, show or do both"
    
    # Define a static variable
    if not hasattr(plot_batch_im, "static_variable"):
        plot_batch_im.cnt = 1        
    
    # Plot subplots (square)
    plt.figure()
    M = math.ceil(math.sqrt(input.shape[0]))
    for i in range(0, input.shape[0]):
        plt.subplot(M, M, i+1)
        if title_prefix is not None:
            plt.title(title_prefix+f"{i}")
        plt.imshow(input[i])
        plt.axis('off')
    if suptitle is not None:    
        plt.suptitle(suptitle)
    plt.tight_layout()
    
    if save:
        pth = fld_pth + '/' + time_str() + file_suffix
        if Path(pth).is_file():
            pth += f'_{plot_batch_im.cnt}'
            plot_batch_im.cnt += 1
        else:
            plot_batch_im.cnt = 1
            
        plt.savefig(pth)
    if show:
        plt.show()
    


# Extras
###############################################################################

def assign_fwd_frozen_patch_tokens(cfg, model, backbone_model):
    model.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    
    if hasattr(backbone_model, "patch_size"):
        # The hook is executed before the forward pass, allowing you to inspect or modify the input data.
        # Takes in 2 args, module and input
        model.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
        # Center padding ensures image is a multiple of patch size (input to forward method = dict{x, masks=masks})
        
def check_config(cfg):
    if isinstance(cfg, str):
        if not Path(cfg).is_file():
            raise TypeError(f"The file {cfg} does not exists.")
        cfg = mmcv.Config.fromfile(cfg)
    elif not isinstance(cfg, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(cfg)))   
    return cfg

def get_frozen_bb_ret_patch_tks(cfg, backbone_model, device='cuda:0'):
    """ 
        Model that is frozen and that returns the patch token outputs
    """
    cfg = check_config(cfg)  
        
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.test_cfg = cfg.get('test_cfg')
    
    bb_model_frozen = build_backbone(cfg.model)
    
    bb_model_frozen.forward = partial(
        backbone_model.get_last_layer_w_attn_scores,
        # n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    
    if hasattr(backbone_model, "patch_size"):
        bb_model_frozen.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
        
    bb_model_frozen.cfg = cfg
    
    bb_model_frozen.dummy_param = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
    
    bb_model_frozen.to(device)
    bb_model_frozen.eval()  # Eval mode -> no training
        
    return bb_model_frozen

def reshape_norm_tensor() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),  # HWC -> CHW in [0, 1]
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

# def pca_patch_feats(patch_feats, n):
#     reshaped = False
#     if len(patch_feats.shape) == 4:
#         reshaped = True
#         B, ed, h0, w0 = patch_feats.shape
#         pf = patch_feats.reshape(B, ed, -1)  # [B, ed, N]
#         pf = pf.permute([0, 2, 1])  # [B, N, ed]
#     else:
#         pf = patch_feats.clone()
           
    
#     # Center the data (subtract mean)
#     mean = torch.mean(pf, dim=-2, keepdims=True)  # keepdims=True
#     patch_feats_c = pf - mean
    
#     # Calculate the covar mtx of the patch features
#     cov_patches = torch.matmul(patch_feats_c.transpose(-2, -1), patch_feats_c) / (pf.size(-2) - 1)
    
#     # Perform SVD on the covariance matrix
#     U, _, _ = torch.svd(cov_patches)

#     # Extract the top principal components
#     principal_components = U[..., :n]
    
#     # Project the original data onto the principal components
#     pca_result = torch.matmul(pf, principal_components)  # []
    
#     if reshaped:
#         # pf = pf.reshape(B, h0, w0, -1).permute(0, 3, 1, 2)
#         pca_result = pca_result.reshape(B, h0, w0, n).permute(0, 3, 1, 2)
    
#     return pca_result
    
# pca_patches = pca_patch_feats(img_feats[0], n=1)

# # min/max scaling
# pca_patches = (pca_patches - pca_patches.min(dim=-3, keepdim=True)[0]) / (pca_patches.max(dim=-3, keepdim=True)[0] - pca_patches.min(dim=-3, keepdim=True)[0])
