import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

import dinov2.eval.segmentation.models

import os
from matplotlib import pyplot as plt

from mmseg.models import build_backbone

from pathlib import Path
import mmcv
import numpy as np

import urllib
from mmcv.runner import load_checkpoint
import cv2


import urllib
from PIL import Image
from torchvision import transforms

from prep_model import get_bb_name, get_dino_backbone,\
        get_seg_had_config, get_seg_model, prep_img_tensor, \
        conv_to_numpy_img, get_pca_res, plot_batch_im

save_plots=False

# Load the pre-trained backbone
BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")
bb_checkpoint_path = './checkpoints_backbone/dinov2_vits14_pretrain.pth'

backbone_name = get_bb_name(BACKBONE_SIZE)
backbone_model = get_dino_backbone(backbone_name, backbone_cp=bb_checkpoint_path)

# Open images from the file list
pth = './oup_imgs/Dogs/'
# pth_img = '/usr/bmicnas02/data-biwi-01/foundation_models/da_data/brain/hcp2/images/test/0050.png'
filelist = [pth+f'dog{i}.png' for i in range(1, 4)]

resized_shape = [448, 448]  
img_tensors = prep_img_tensor(filelist, resized_shape=resized_shape, 
                              pad_patch_sz=backbone_model.patch_size)

# Calculate img patch tokens all at once (Batch Dim)
with torch.no_grad():
    patch_tks = backbone_model.get_intermediate_layers(img_tensors, reshape=True)[0]
patch_tks = patch_tks.detach().cpu().numpy()

# Orig images resized and padded (without normalization)
imgs_transformed = conv_to_numpy_img(prep_img_tensor(filelist, resized_shape=resized_shape, 
                                                     pad_patch_sz=backbone_model.patch_size, 
                                                     norm=False)).astype('uint8')

# Get the PCA applied patch tokens (resized to img sz)
pca_res = get_pca_res(patch_tks, rescale_fac=backbone_model.patch_size)
    

# Concat the PCA result next to the original image
orig_w_pca_res = np.concatenate([imgs_transformed, pca_res['pca_fg']], axis=-2) # Width axis

# Plot the PCA results
pth = './oup_imgs/PCAs/'
plot_batch_im(orig_w_pca_res, pth, show=True, save=save_plots)


# Get the patch tokens and self attentions from the last layer
with torch.no_grad():
    patch_tk, attn = backbone_model.get_last_layer_w_attn_scores(img_tensors, reshape=True)

# Normalize the attention scores with min-max scaling in range [0, 1] (indep for each head)  
attn = (attn - attn.min(dim=-3, keepdim=True)[0]) / (attn.max(dim=-3, keepdim=True)[0] - attn.min(dim=-3, keepdim=True)[0])

# Convert to numpy array
attn = attn.detach().cpu().numpy()

# Create the self attention plots    
fld_pth = './oup_imgs/self_attentions/'
suptitle = "Self-attention for the CLS token query"
plot_batch_im(attn[0], fld_pth, title_prefix='Head', suptitle=suptitle, show=True, save=save_plots)


# Load the pre-trained segmentation head Linear Boosted (+ms)
HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
HEAD_TYPE = "ms" # in ("ms, "linear")

cfg = get_seg_had_config(backbone_name, HEAD_DATASET, HEAD_TYPE, head_sclae_cnt=HEAD_SCALE_COUNT, cfg_fld_path='./configs')
model = get_seg_model(backbone_model, cfg, cp_fld_path='./checkpoints_segHead', eval=True) 


# Semantic segmentation with Boosted Linear head (+ms)
img_file = filelist[0]
segmentation_logits = inference_segmentor(model, img_file)[0]
# segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)

# Save the res of MS seg head
show_result_pyplot(model=model, img=img_file, 
                #    palette=DATASET_COLORMAPS[HEAD_DATASET], 
                   result=[segmentation_logits], 
                   out_file=f'./oup_imgs/seg_out_{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.png',
                   block=False)
plt.close()


# Pre-trained backbone with ViT adapter + mask2former seg head
import dinov2.eval.segmentation_m2f.models.segmentors

BACKBONE_SIZE = "giant"
HEAD_DATASET = "ade20k" 
HEAD_TYPE = "m2f" 

backbone_name = get_bb_name(BACKBONE_SIZE)
cfg = get_seg_had_config(backbone_name, HEAD_DATASET, HEAD_TYPE, head_sclae_cnt=3, cfg_fld_path=None)

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

# Load the M2F model with ViT adapter
local_seg_model = True
# Load the checkpoint to the segmentation model
if not local_seg_model:
    # Instantiate the empty segmentation model
    model = init_segmentor(cfg)
    CHECKPOINT_URL = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.pth"    
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
else:
    cp = f'./checkpoints_seg_model_mask2former/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.pth'
    model = init_segmentor(cfg, checkpoint=None)
    checkpoint = load_checkpoint(model, cp, map_location="cpu")
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    
    
model.cuda()
model.eval()

# img_array = np.array(image)[:, :, ::-1] # BGR  WHY ???
segmentation_logits = inference_segmentor(model, img_file)[0]
# segmented_image = render_segmentation(segmentation_logits, "ade20k")

# Save the res of m2f seg head with dino plugged into a ViT adapter
show_result_pyplot(model=model, img=img_file, result=[segmentation_logits], 
                   out_file=f'./oup_imgs/seg_out_{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.png',
                   block=False)
plt.close()
