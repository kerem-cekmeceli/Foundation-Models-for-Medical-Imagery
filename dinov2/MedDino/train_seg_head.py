
import sys
from pathlib import Path
 
dino_main_pth = Path(__file__).parent.parent
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

from MedDino.med_dinov2.models.segmentor import Segmentor
from MedDino.med_dinov2.layers.segmentation import ConvHead
from MedDino.med_dinov2.data.datasets import SegmentationDataset
from torch.utils.data import DataLoader
from MedDino.med_dinov2.tools.main_fcts import train
from MedDino.med_dinov2.losses.losses import mIoULoss

from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR

# import os
# import math
# import itertools
# import urllib
# from functools import partial
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms      
# import torch.nn.functional as F
# import mmcv
# from mmseg.models import build_backbone
# from mmseg.apis.inference import LoadImage
# from mmcv.parallel import collate, scatter
# from mmseg.datasets.pipelines import Compose


# Load the pre-trained backbone
backbone_sz = "small" # in ("small", "base", "large" or "giant")
backbone_name = get_bb_name(backbone_sz)
bb_checkpoint_path = dino_main_pth/f'Checkpoints/Orig/backbone/{backbone_name}_pretrain.pth'
backbone = get_dino_backbone(backbone_name, backbone_cp=bb_checkpoint_path)

# Initialize the segmentation decode head
num_classses = 15
conv_head = ConvHead(embedding_sz=backbone.embed_dim, 
                     num_classses=num_classses)

# Initialize the segmentor
model = Segmentor(backbone=backbone,
                  decode_head=conv_head,
                  train_backbone=False)

# Define data the transforms @TODO
img_transform = None
mask_transform = None

# Get the data loader
data_root_pth = dino_main_pth.parent / 'DataSample'
train_dataset = SegmentationDataset(img_dir=data_root_pth/'images/test',
                                    mask_dir=data_root_pth/'labels/test',
                                    num_classes=num_classses,
                                    file_extension='.png',
                                    mask_suffix='_labelTrainIds',
                                    img_transform=img_transform,
                                    mask_transform=mask_transform)
batch_sz = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_sz,
                              shuffle=False, pin_memory=True)

# Optimizer
optm = torch.optim.AdamW(model.parameters(), 
                         lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))

# LR scheduler
total_epochs = 10
warmup_iters = total_epochs//2
scheduler1 = LinearLR(optm, total_iters=warmup_iters)
scheduler2 = PolynomialLR(optm, power=1.0)
scheduler = SequentialLR(optm, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])

# Loss function
mIoU = mIoULoss(n_classes=num_classses)

# Training loop
device = 'cuda:0'
res = train(model=model, train_loader=train_dataloader, loss_fn=mIoU, 
            optimizer=optm, n_epochs=total_epochs, device=device)


print()
# Open images from the file list
# '/usr/bmicnas02/data-biwi-01/foundation_models/da_data/brain/hcp2/images/test/0050.png'
# pth = './oup_imgs/Dogs/'
# pth = '/usr/bmicnas02/data-biwi-01/foundation_models/da_data/brain/hcp2/images/test/'
# pth = dino_main_pth.parent / 'DataSample/images/test'
# filelist = [str(pth/str(i).zfill(4)+'.png') for i in range(48, 52)]







# # Save the res of m2f seg head with dino plugged into a ViT adapter
# show_result_pyplot(model=model, img=img_file, result=[segmentation_logits], 
#                 out_file=(dino_main_pth/f'oup_imgs/seg_out_{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.png').as_posix(),
#                 block=False)
# plt.close()





################################

# from med_dinov2.data.dataloader import get_file_list, \
#     color_map_from_imgs


# fld_pth = (dino_main_pth.parent / 'DataSample/labels/test').as_posix()

# file_ls = get_file_list(fld_pth, start_idx=0, num_img=256, 
#                         extension='.png', file_suffix='_labelTrainIds')
# color_map = color_map_from_imgs(fld_pth, file_ls, device='cuda:0')
