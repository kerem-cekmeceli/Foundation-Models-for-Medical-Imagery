
import sys
from pathlib import Path
 
dino_main_pth = Path(__file__).parent.parent
orig_dino_pth = dino_main_pth / 'OrigDino'
sys.path.insert(1, dino_main_pth.as_posix())
sys.path.insert(2, orig_dino_pth.as_posix())

import torch
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

from prep_model import get_bb_name, get_dino_backbone, time_str
from OrigDino.dinov2.eval.segmentation import models

from MedDino.med_dinov2.models.segmentor import Segmentor
from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ConvUNet
from MedDino.med_dinov2.data.datasets import SegmentationDataset
from torch.utils.data import DataLoader
from MedDino.med_dinov2.tools.main_fcts import train, test
from MedDino.med_dinov2.metrics.metrics import mIoU, DiceScore

from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torchinfo import summary
from torch.nn import CrossEntropyLoss
import wandb
from MedDino.med_dinov2.tools.checkpointer import Checkpointer
from mmseg.models.decode_heads import *


cluster_paths = True
save_checkpoints = True
log_the_run = True

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# dec_head = ConvHeadLinear(in_channels=[backbone.embed_dim]*n_concat, 
#                           num_classses=num_classses,
#                           out_upsample_fac=backbone.patch_size,
#                           bilinear=False)

# dec_head = FCNHead(num_convs=3,
#                    kernel_size=3,
#                    concat_input=True,
#                    dilation=1,
#                    in_channels=[backbone.embed_dim]*n_concat,  # input channels
#                    channels=backbone.embed_dim,  # Conv channels
#                    num_classes=num_classses,  # output channels
#                    dropout_ratio=0.1,
#                    conv_cfg=dict(type='Conv2d'), # None = conv2d
#                    norm_cfg=dict(type='BN'),
#                    act_cfg=dict(type='ReLU'),
#                    in_index=[i for i in range(n_concat)],
#                    input_transform='resize_concat',
#                    init_cfg=dict(
#                        type='Normal', std=0.01, override=dict(name='conv_seg')))


dec_head = ConvUNet(in_channels=[backbone.embed_dim]*n_concat,
                    num_classses=num_classses, 
                    # in_index=None,
                    # in_resize_factors=None,
                    # align_corners=False,
                    # dropout_rat_cls_seg=0.,
                    nb_up_blocks=4,
                    upsample_facs=2,
                    bilinear=True,
                    conv_per_up_blk=2,
                    res_con=False,
                    res_con_interv=1)

dec_head.to(device)

print("Convolutional decode head")
summary(dec_head)


# Initialize the segmentor
train_backbone=False
model = Segmentor(backbone=backbone,
                  decode_head=dec_head,
                  train_backbone=train_backbone,
                  reshape_dec_oup=True)
model.to(device)

# Print model info
print("Segmentor model")
summary(model)

# Define data augmentations
img_scale_fac = 1  # Try without first
central_crop = True
augmentations = []
augmentations.append(dict(type='ElasticTransformation', data_aug_ratio=0.25))
augmentations.append(dict(type='StructuralAug', data_aug_ratio=0.25))
augmentations.append(dict(type='PhotoMetricDistortion'))

if img_scale_fac > 1:
    augmentations.append(dict(type='Resize2',
                            scale_factor=float(img_scale_fac), #HW
                            keep_ratio=True))

augmentations.append(dict(type='Normalize', 
                          mean=[123.675, 116.28, 103.53],  #RGB
                          std=[58.395, 57.12, 57.375],  #RGB
                          to_rgb=True))
if central_crop:
    augmentations.append(dict(type='CentralCrop',  
                              size_divisor=backbone.patch_size))
else:
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
val_dataset = SegmentationDataset(img_dir=data_root_pth/'images/val',
                                  mask_dir=data_root_pth/'labels/val',
                                  num_classes=num_classses,
                                  file_extension='.png',
                                  mask_suffix='_labelTrainIds',
                                  augmentations=augmentations,
                                  )
test_dataset = SegmentationDataset(img_dir=data_root_pth/'images/test',
                                   mask_dir=data_root_pth/'labels/test',
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
optm_cfg = dict(lr = 0.001,
                wd = 0.0001,
                betas = (0.9, 0.999))
optm = torch.optim.AdamW(model.parameters(), 
                         lr=optm_cfg['lr'], weight_decay=optm_cfg['wd'], betas=optm_cfg['betas'])

# LR scheduler
nb_epochs = 50#100
warmup_iters = 20
lr_cfg = dict(linear_lr = dict(start_factor=1/3, end_factor=1.0, total_iters=warmup_iters),
              polynomial_lr = dict(power=1.0, total_iters=nb_epochs*2-warmup_iters))
scheduler1 = LinearLR(optm, **lr_cfg['linear_lr'])
scheduler2 = PolynomialLR(optm, **lr_cfg['polynomial_lr'])
scheduler = SequentialLR(optm, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
#@TODO: check tuning strategies for LR

# Loss function
loss = CrossEntropyLoss()

# Metrics
bg_channel = 0
metrics=dict(mIoU=mIoU(n_class=num_classses, 
                       prob_inputs=False, # Decoder does not return probas explicitly
                       soft=True,
                       bg_ch_to_rm=bg_channel,  # bg channel to be removed 
                       reduction='mean'), # average over batches and classes
             dice=DiceScore(n_class=num_classses, 
                            prob_inputs=False,  # Decoder does not return probas explicitly
                            soft=True,
                            bg_ch_to_rm=bg_channel,
                            reduction='mean',
                            k=1, 
                            epsilon=1e-6,))

# Init the logger (wandb)
wnadb_config = dict(backbone_name=backbone_name,
                    backbone_last_n_concat=n_concat,
                    decode_head=dec_head.__class__.__name__,
                    train_backbone=train_backbone,
                    dataset=str(data_root_pth),
                    num_classes=num_classses,
                    augmentations=augmentations,
                    nb_epochs=nb_epochs,
                    lr_cfg=lr_cfg)


wandb_log_path = dino_main_pth / 'Logs'
wandb_log_path.mkdir(parents=True, exist_ok=True)
wandb_group_name = 'SEG_bb_' + backbone_sz + '_frozen' if not train_backbone else '_with_train'
log_mode = 'online' if log_the_run else 'disabled'
logger = wandb.init(project='FoundationModels_MedDino',
                    group=wandb_group_name,
                    config=wnadb_config,
                    dir=wandb_log_path,
                    name=time_str(),
                    mode=log_mode)

seg_res_log_itv = nb_epochs//4  # Log seg reult every xxx epochs
seg_res_nb_patient = 1
seg_log_per_batch = 3

SLICE_PER_PATIENT = 256
first_n_batch_to_seg_log = math.ceil(SLICE_PER_PATIENT/batch_sz*seg_res_nb_patient)


# Init checkpointer
n_best = 2
models_pth = dino_main_pth / f'Checkpoints/MedDino/conv_head'
models_pth.mkdir(parents=True, exist_ok=True)
cp = Checkpointer(save_pth=models_pth, 
                  monitor=['val_loss', 'val_dice', 'val_mIoU'],
                  minimize=[True, False, False], 
                  n_best=n_best, 
                  name_prefix=time_str())

if not save_checkpoints:
    cp = None
    
# Training loop
train(model=model, train_loader=train_dataloader, 
      val_loader=val_dataloader, loss_fn=loss, 
      optimizer=optm, scheduler=scheduler,
      n_epochs=nb_epochs, device=device, logger=logger,
      checkpointer=cp, metrics=metrics,
      seg_val_intv=seg_res_log_itv,
      first_n_batch_to_seg_log=first_n_batch_to_seg_log,
      seg_log_per_batch=seg_log_per_batch)

# Test hard decision predicted seg classes per pix
test(model=model, test_loader=test_dataloader, loss_fn=loss, 
     device=device, logger=logger, metrics=metrics, soft=False,
     first_n_batch_to_seg_log=first_n_batch_to_seg_log, 
     seg_log_per_batch=seg_log_per_batch)

#@TODO log input pred gt randomly every n iter (dice scores per class)
#@TODO add types and comments
#@TODO write readme.md
#@TODO maybe torchlighning Friday

#@TODO ASK
# validate and test at the original resolution, also for training ???
# x --> scale up resize --> model --> scale down resize  --> accuracy 
# Which decoder architecture 
# which segmentations to log (test, how frequent, val ?)



#finish logging
wandb.finish()
print('***END***')
