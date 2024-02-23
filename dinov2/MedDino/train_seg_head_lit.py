
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

from prep_model import get_bb_name, get_dino_backbone, time_str, get_backone_patch_embed_sizes
from OrigDino.dinov2.eval.segmentation import models

from MedDino.med_dinov2.models.segmentor import Segmentor
# from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ConvUNet
# from mmseg.models.decode_heads import *
from MedDino.med_dinov2.data.datasets import SegmentationDataset
from torch.utils.data import DataLoader
from MedDino.med_dinov2.tools.main_fcts import train, test
from MedDino.med_dinov2.eval.metrics import mIoU, DiceScore
from MedDino.med_dinov2.eval.losses import FocalLoss, DiceScore, CompositionLoss

from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torchinfo import summary
from torch.nn import CrossEntropyLoss
import wandb
from MedDino.med_dinov2.tools.checkpointer import Checkpointer
from MedDino.med_dinov2.eval.losses import * 
from MedDino.med_dinov2.models.lit_segmentor import LitSegmentor
import os
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


cluster_paths = True
save_checkpoints = True
log_the_run = True

# Set the precision
precision = 'highest' if cluster_paths else 'high' 
torch.set_float32_matmul_precision(precision)

# Backbone config
train_backbone = True
backbone_sz = "small" # in ("small", "base", "large" or "giant")

backbone_name = get_bb_name(backbone_sz)
bb_checkpoint_path = dino_main_pth/f'Checkpoints/Orig/backbone/{backbone_name}_pretrain.pth'

dino_bb_cfg = dict(backbone_name=backbone_name, backbone_cp=bb_checkpoint_path)
patch_sz, embed_dim = get_backone_patch_embed_sizes(backbone_name)

# Select dataset
dataset = 'hcp2' # 'hcp2'

if dataset=='hcp1':
    data_path_suffix = 'brain/hcp1'
    num_classses = 15
    bg_channel_loss = None
    bg_channel_metric = 0
elif dataset=='hcp2':
    data_path_suffix = 'brain/hcp2'
    num_classses = 15
    bg_channel_loss = None
    bg_channel_metric = 0
    
elif dataset=='cardiac_acdc':
    data_path_suffix = 'cardiac/acdc'
    num_classses = 2
    bg_channel_loss = None
    bg_channel_metric = 0
elif dataset=='cardiac_rvsc':
    data_path_suffix = 'cardiac/rvsc'
    num_classses = 2
    bg_channel_loss = None
    bg_channel_metric = 0
    
elif dataset=='prostate_nci':
    data_path_suffix = 'prostate/nci'
    num_classses = 3
    bg_channel_loss = None
    bg_channel_metric = 0
elif dataset=='prostate_usz':
    data_path_suffix = 'prostate/pirad_erc'
    num_classses = 3
    bg_channel_loss = None
    bg_channel_metric = 0
    
else:
    ValueError(f'Dataset: {dataset} is not defined')


# Decoder config
n_concat = 4
dec_head_cfg_conv_lin = dict(in_channels=[embed_dim]*n_concat, 
                             num_classses=num_classses,
                             out_upsample_fac=patch_sz,
                             bilinear=True)

dec_head_cfg_fcn = dict(num_convs=3,
                        kernel_size=3,
                        concat_input=True,
                        dilation=1,
                        in_channels=[embed_dim]*n_concat,  # input channels
                        channels=embed_dim,  # Conv channels
                        num_classes=num_classses,  # output channels
                        dropout_ratio=0.1,
                        conv_cfg=dict(type='Conv2d'), # None = conv2d
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU'),
                        in_index=[i for i in range(n_concat)],
                        input_transform='resize_concat',
                        init_cfg=dict(
                            type='Normal', std=0.01, override=dict(name='conv_seg')))

dec_head_cfg_unet = dict(in_channels=[embed_dim]*n_concat,
                        num_classses=num_classses,
                        # in_index=None,
                        # in_resize_factors=None,
                        # align_corners=False,
                        dropout_rat_cls_seg=0.1,
                        nb_up_blocks=4,
                        upsample_facs=2,
                        bilinear=False,
                        conv_per_up_blk=2,
                        res_con=True,
                        res_con_interv=1
                        )

decs_dict = dict(lin=dict(name='ConvHeadLinear', params=dec_head_cfg_conv_lin),
                 fcn=dict(name='FCNHead', params=dec_head_cfg_fcn),
                 unet=dict(name='ConvUNet', params=dec_head_cfg_unet))

# Choose the decode head config
dec_head_cfg = decs_dict['unet']


# Training hyperparameters
nb_epochs = 90
warmup_iters = 20

# Config the batch size for training
batch_sz = 16

# Optimizer Config
optm_cfg = dict(name='AdamW',
                params=dict(lr = 0.001,
                            weight_decay = 0.0001,
                            betas = (0.9, 0.999)))

# LR scheduler config
scheduler_configs = []
scheduler_configs.append(\
    dict(name='LinearLR',
         params=dict(start_factor=1/3, end_factor=1.0, total_iters=warmup_iters)))
scheduler_configs.append(\
    dict(name='PolynomialLR',
         params=dict(power=1.0, total_iters=nb_epochs-warmup_iters)))

scheduler_cfg = dict(name='SequentialLR',
                     params=dict(scheduler_configs=scheduler_configs,
                                  milestones=[warmup_iters]),
                    )

# Loss Config
epsilon = 1  # smoothing factor 
k=1  # power

# CE Loss
loss_cfg_ce = dict()

# Dice Loss
loss_cfg_dice = dict(prob_inputs=False, 
                    bg_ch_to_rm=bg_channel_loss, # not removing results in better segmentation
                    reduction='mean',
                    epsilon=epsilon,
                    k=k)

# CE-Dice Loss
loss_cfg_dice_ce=dict(loss1=dict(name='CE',
                                 params=loss_cfg_ce),
                      loss2=dict(name='Dice', 
                                 params=loss_cfg_dice),
                      comp_rat=0.5)

# Focal Loss
loss_cfg_focal = dict(gamma=2,
                      alpha=None)

# Focal-Dice Loss
loss_cfg_comp_foc_dice=dict(loss1=dict(name='Focal',
                                 params=loss_cfg_focal),
                            loss2=dict(name='Dice', 
                                        params=loss_cfg_dice),
                            comp_rat=20/21)

loss_cfgs_dict = dict(ce=dict(name='CrossEntropyLoss', params=loss_cfg_ce),
                      dice=dict(name='DiceLoss', params=loss_cfg_dice),
                      dice_ce=dict(name='CompositionLoss', params=loss_cfg_dice_ce),
                      focal=dict(name='FocalLoss', params=loss_cfg_focal),
                      focal_dixce=dict(name='CompositionLoss', params=loss_cfg_comp_foc_dice))

loss_cfg = loss_cfgs_dict['ce']


# Metrics
epsilon = 1  # smoothing factor 
k=1  # power

SLICE_PER_PATIENT = 256
assert SLICE_PER_PATIENT % batch_sz == 0, \
    f'batch size must be a multiple of slice/patient but got {batch_sz} and {SLICE_PER_PATIENT}'

miou_cfg=dict(prob_inputs=False, # Decoder does not return probas explicitly
              soft=False,
              bg_ch_to_rm=bg_channel_metric,  # bg channel to be removed 
              reduction='mean',
              vol_batch_sz=SLICE_PER_PATIENT,
              epsilon=epsilon)

dice_cfg=dict(prob_inputs=False,  # Decoder does not return probas explicitly
             soft=False,
             bg_ch_to_rm=bg_channel_metric,
             reduction='mean',
             k=k, 
             epsilon=epsilon,
             vol_batch_sz=SLICE_PER_PATIENT)

metric_cfgs=[dict(name='mIoU', params=miou_cfg), 
             dict(name='dice', params=dice_cfg)]


# Seg result logging cfg
seg_res_log_itv = max(nb_epochs//5, 1)   # Log seg reult every N epochs
seg_res_nb_patient = 1  # Process minibatches for N number of patients
seg_log_per_batch = 4  # Log N samples from each minibatch
assert seg_log_per_batch<=batch_sz
first_n_batch_to_seg_log = math.ceil(SLICE_PER_PATIENT/batch_sz*seg_res_nb_patient)

sp = seg_log_per_batch+1
# maximal separation from each other and from edges (from edges is prioritized)
log_idxs = torch.arange(batch_sz//sp, 
                        batch_sz//sp*sp, 
                        batch_sz//sp)
log_idxs = log_idxs + (batch_sz%sp)//2
log_idxs = log_idxs.tolist()

# Init the segmentor model
segmentor_cfg = dict(backbone=dino_bb_cfg,
                     decode_head=dec_head_cfg,
                     loss_config=loss_cfg, 
                     optimizer_config=optm_cfg,
                     schedulers_config=scheduler_cfg,
                     metric_configs=metric_cfgs,
                     train_backbone=train_backbone,
                     reshape_dec_oup=True,
                     align_corners=False,
                     val_metrics_over_vol=True, # Also report metrics over vol
                     first_n_batch_to_seg_log=first_n_batch_to_seg_log,
                     minibatch_log_idxs=log_idxs,
                     seg_val_intv=seg_res_log_itv)

model = LitSegmentor(**segmentor_cfg)

# Print model info
summary(model)


# Define data augmentations
img_scale_fac = 1  # Keep at 1 
central_crop = True
augmentations = []
train_augmentations = []
train_augmentations.append(dict(type='ElasticTransformation', data_aug_ratio=0.25))
train_augmentations.append(dict(type='StructuralAug', data_aug_ratio=0.25))
train_augmentations.append(dict(type='PhotoMetricDistortion'))

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
                              size_divisor=patch_sz))
else:
    augmentations.append(dict(type='CentralPad',  
                            size_divisor=patch_sz,
                            pad_val=0, seg_pad_val=0))
    
train_augmentations = train_augmentations + augmentations

# Get the data loader
if cluster_paths:
    data_root_pth = Path('/usr/bmicnas02/data-biwi-01/foundation_models/da_data') 
else:            
    data_root_pth = dino_main_pth.parent.parent / 'DataFoundationModels'
    
data_root_pth = data_root_pth / data_path_suffix

train_fld = 'train-filtered' if 'hcp' in dataset else 'train'

train_dataset = SegmentationDataset(img_dir=data_root_pth/'images'/train_fld,
                                    mask_dir=data_root_pth/'labels'/train_fld,
                                    num_classes=num_classses,
                                    file_extension='.png',
                                    mask_suffix='_labelTrainIds',
                                    augmentations=train_augmentations,
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

num_workers = 16#min(os.cpu_count()//2, 10)
persistent_workers=True
drop_last=True
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_sz,
                              shuffle=True, pin_memory=True, num_workers=num_workers,
                              persistent_workers=persistent_workers, drop_last=drop_last)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_sz,
                              shuffle=False, pin_memory=True, num_workers=num_workers,
                              persistent_workers=persistent_workers, drop_last=drop_last)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_sz,
                              shuffle=False, pin_memory=True, num_workers=num_workers,
                              persistent_workers=persistent_workers, drop_last=drop_last)


# Init the logger (wandb)
loss_name = loss_cfg['name'] if not loss_cfg['name']=='CompositionLoss' else \
                f'{loss_cfg["params"]["loss1"]["name"]}{loss_cfg["params"]["loss2"]["name"]}Loss'
run_name = f'{data_root_pth.stem}_{model.model.decode_head.__class__.__name__}_{loss_name}'

wnadb_config = dict(backbone_name=backbone_name,
                    backbone_last_n_concat=n_concat,
                    decode_head=model.model.decode_head.__class__.__name__,
                    dec_head_cfg=dec_head_cfg,
                    segmentor_cfg=segmentor_cfg,
                    dataset=str(data_root_pth),
                    batch_sz=batch_sz,
                    num_classes=num_classses,
                    augmentations=augmentations,
                    nb_epochs=nb_epochs,
                    scheduler_cfg=scheduler_cfg,
                    optm_cfg=optm_cfg,
                    loss_cfg=loss_cfg,
                    metrics_cfg=metric_cfgs,
                    timestamp=time_str(),
                    torch_precision=precision)


wandb_log_path = dino_main_pth / 'Logs'
wandb_log_path.mkdir(parents=True, exist_ok=True)
wandb_group_name = 'SEG_bb_' + backbone_sz + '_frozen' if not segmentor_cfg['train_backbone'] else '_with_train'
log_mode = 'online' if log_the_run else 'disabled'

logger = WandbLogger(project='FoundationModels_MedDino',
                    group=wandb_group_name,
                    config=wnadb_config,
                    dir=wandb_log_path,
                    name=run_name,
                    mode=log_mode,
                    settings=wandb.Settings(_service_wait=30),  # Can increase timeout
                    tags=[dataset])

checkpointers = []
n_best = 2 if save_checkpoints else 0
models_pth = dino_main_pth / f'Checkpoints/MedDino/{model.model.decode_head.__class__.__name__}'
models_pth.mkdir(parents=True, exist_ok=True)
time_s = time_str()
checkpointers.append(ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_loss", mode='min', filename=time_s+'-{epoch}-{val_loss:.2f}'))
checkpointers.append(ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_dice", mode='max', filename=time_s+'-{epoch}-{val_dice:.2f}'))
checkpointers.append(ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_mIoU", mode='max', filename=time_s+'-{epoch}-{val_mIoU:.2f}'))


trainer = L.Trainer(max_epochs=nb_epochs, logger=logger, log_every_n_steps=100, num_sanity_val_steps=0,
                    enable_checkpointing=True, callbacks=checkpointers)

# Train the model
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Load the best checkpoint (highest val_dice)
logs = trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=checkpointers[0].best_model_path)

#@TODO  multi GPU (performance optm) (metrics per volume are problematic)
#@TODO add types and comments
#@TODO write readme.md
#@TODO use the original hdf5 files 

print('Done !')
#finish logging
wandb.finish()
print('***END***')
