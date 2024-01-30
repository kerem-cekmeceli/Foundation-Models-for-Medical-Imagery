
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
        conv_to_numpy_img, get_pca_res, plot_batch_im, time_str
from OrigDino.dinov2.eval.segmentation import models

from MedDino.med_dinov2.models.segmentor import Segmentor
from MedDino.med_dinov2.layers.segmentation import ConvHead
from MedDino.med_dinov2.data.datasets import SegmentationDataset
from torch.utils.data import DataLoader
from MedDino.med_dinov2.tools.main_fcts import train, test
from MedDino.med_dinov2.losses.losses import mIoULoss

from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from mmseg.datasets.pipelines import Compose
from torchinfo import summary

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

cluster_paths = True


# Load the pre-trained backbone
backbone_sz = "small" # in ("small", "base", "large" or "giant")
backbone_name = get_bb_name(backbone_sz)
bb_checkpoint_path = dino_main_pth/f'Checkpoints/Orig/backbone/{backbone_name}_pretrain.pth'
backbone = get_dino_backbone(backbone_name, backbone_cp=bb_checkpoint_path)

print("Dino backbone")
summary(backbone)

# Initialize the segmentation decode head
num_classses = 15
n_concat = 4
device = 'cuda:0'
conv_head = ConvHead(embedding_sz=backbone.embed_dim, 
                     num_classses=num_classses,
                     n_concat=n_concat,
                     interp_fact=backbone.patch_size)
conv_head.to(device)

print("Convolutional decode head")
summary(conv_head)

# Initialize the segmentor
model = Segmentor(backbone=backbone,
                  decode_head=conv_head,
                  train_backbone=False)
model.to(device)

# Print model info
print("Segmentor model")
summary(model)

# Define data augmentations
img_scale_fac = 1
augmentations = []
augmentations.append(dict(type='ElasticTransformation', data_aug_ratio=0.25))
augmentations.append(dict(type='StructuralAug', data_aug_ratio=0.25))
augmentations.append(dict(type='PhotoMetricDistortion'))
augmentations.append(dict(type='Resize2',
                          scale_factor=float(img_scale_fac), #HW
                          keep_ratio=True))

augmentations.append(dict(type='Normalize', 
                          mean=[123.675, 116.28, 103.53],  #RGB
                          std=[58.395, 57.12, 57.375],  #RGB
                          to_rgb=True))
augmentations.append(dict(type='CentralPad',
                          size_divisor=backbone.patch_size,
                          pad_val=0, seg_pad_val=0))

# Get the data loader
if cluster_paths:
        data_root_pth = Path('/usr/bmicnas02/data-biwi-01/foundation_models/da_data/brain/hcp1') 
else:            
        data_root_pth = dino_main_pth.parent.parent / 'DataFoundationModels/HPC1'

train_dataset = SegmentationDataset(img_dir=data_root_pth/'images/train-filtered',
                                    mask_dir=data_root_pth/'labels/train-filtered',
                                    num_classes=num_classses,
                                    file_extension='.png',
                                    mask_suffix='_labelTrainIds',
                                    augmentations=augmentations,
                                    )
val_dataset = SegmentationDataset(img_dir=data_root_pth/'images/val-filtered',
                                  mask_dir=data_root_pth/'labels/val-filtered',
                                  num_classes=num_classses,
                                  file_extension='.png',
                                  mask_suffix='_labelTrainIds',
                                  augmentations=augmentations,
                                  )
test_dataset = SegmentationDataset(img_dir=data_root_pth/'images/test-filtered',
                                   mask_dir=data_root_pth/'labels/test-filtered',
                                   num_classes=num_classses,
                                   file_extension='.png',
                                   mask_suffix='_labelTrainIds',
                                   augmentations=augmentations,
                                   )
batch_sz = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_sz,
                              shuffle=True, pin_memory=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_sz,
                              shuffle=False, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_sz,
                              shuffle=False, pin_memory=True)

# Optimizer
optm = torch.optim.AdamW(model.parameters(), 
                         lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))

# LR scheduler
total_epochs = 2
warmup_iters = 0#total_epochs//10
scheduler1 = LinearLR(optm, start_factor=1/3, end_factor=1.0, total_iters=warmup_iters)
scheduler2 = PolynomialLR(optm, power=1.0)
scheduler = SequentialLR(optm, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])

# Loss function
mIoU = mIoULoss(n_classes=num_classses)

# Training loop
if cluster_paths:
        models_pth = Path(f'scratch_net/biwidl210/kcekmeceli/DataFoundationModels/trained_models/conv_head/{time_str()}.pth') 
else:   
        models_pth = dino_main_pth / f'Checkpoints/MedDino/conv_head/{time_str()}.pth'
(train_loss, val_loss) = train(model=model, train_loader=train_dataloader, 
                               val_loader=val_dataloader, loss_fn=mIoU, 
                               optimizer=optm, scheduler=scheduler,
                               n_epochs=total_epochs, device=device,
                               save_best_val_path=models_pth)

# Prints the test set mIoU loss
test(model=model, val_loader=val_dataloader, loss_fn=mIoU, device=device)


print('***END***')

