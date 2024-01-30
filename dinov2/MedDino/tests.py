import sys
import pathlib
 
dino_main_pth = pathlib.Path(__file__).parent.parent
orig_dino_pth = dino_main_pth / 'OrigDino'
sys.path.insert(1, dino_main_pth.as_posix())
sys.path.insert(2, orig_dino_pth.as_posix())

import torch
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmcv.runner import load_checkpoint
import cv2
import numpy as np
from matplotlib import pyplot as plt

from prep_model import get_bb_name, get_dino_backbone,\
        get_seg_head_config, get_seg_model, prep_img_tensor, \
        conv_to_numpy_img, get_pca_res, plot_batch_im
from OrigDino.dinov2.eval.segmentation import models

from mmseg.datasets.pipelines import Compose

from MedDino.med_dinov2.data.transforms import ElasticTransformation
import mmcv


# import os
# import math
# import itertools
# import urllib
# from functools import partial
# from pathlib import Path
from PIL import Image
from mmseg.apis.inference import LoadImage
# from torchvision import transforms      
# import torch.nn.functional as F
# import mmcv
# from mmseg.models import build_backbone
# from mmseg.apis.inference import LoadImage
# from mmcv.parallel import collate, scatter
# from mmseg.datasets.pipelines import Compose


pth = dino_main_pth / 'oup_imgs/orig.png'
im = Image.open(pth).convert('RGB') # RGB
# im = cv2.imread(str(pth)) # BGR
# im = pth

def put_in_res_dict(img, mask=None):
        # if pil img : Convert PIL img (RGB) --> ndarray (BGR)
        get_nd_arr = lambda img: np.array(img).copy() if isinstance(img, Image.Image) else img
        # if ndarray convert RGB to BGR
        rgb_2_bgr = lambda img: img[..., ::-1].copy() if isinstance(img, np.ndarray) else img
        
        img = rgb_2_bgr(get_nd_arr(img))
        result = dict(img=mmcv.imread(img, flag='color', channel_order='bgr'))
        
        if mask is not None:
                mask = get_nd_arr(mask)
                result['seg_fields']=mmcv.imread(mask, flag='grayscale')
                
        return result

def rm_from_res_dict(results):
        img = results['img']
        if 'seg_fields' in results.keys():
                mask = results['seg_fields']
                return [img, mask]
        return [img]


transforms = []

# Put in dict and convert to BGR
transforms.append(put_in_res_dict)

# Elastic deformations
# transforms.append(dict(type='ElasticTransformation', data_aug_ratio=1.))

# # random translation, rotation and scaling
# transforms.append(dict(type='StructuralAug', data_aug_ratio=1.))

# # random brightness, contrast(mode 0), saturation, hue, contrast(mode 1) | Img only
# transforms.append(dict(type='PhotoMetricDistortion'))   

# BGR->RGB and Normalize with mean and std given in the paper | Img only
# img_transform.append(dict(type='Normalize', 
#                           mean=[123.675, 116.28, 103.53],  #RGB
#                           std=[58.395, 57.12, 57.375],  #RGB
#                           to_rgb=True))

# transforms.append(dict(type='CentralPad',
#                           size_divisor=14,
#                           pad_val=0, seg_pad_val=0))

# transforms.append(dict(type='Resize2',
#                         scale_factor=3., #HW
#                         keep_ratio=True))

# mmseg/datasets/pipelines/transforms  
# conv the img keys to torch.Tensor with [HWC] -> [CHW]
transforms.append(dict(type='ImageToTensor', keys=['img']))

# Remove the dict and keep the tensor
transforms.append(rm_from_res_dict)

transforms = Compose(transforms)

out = transforms(im)

im_t = out[0]

plt.figure()
plt.imshow(im_t.detach().cpu().numpy().transpose([1, 2, 0])[..., ::-1])
plt.show()
print()