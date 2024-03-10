
import sys
from pathlib import Path
 
dino_main_pth = Path(__file__).parent.parent
orig_dino_pth = dino_main_pth / 'OrigDino'
sys.path.insert(1, dino_main_pth.as_posix())
sys.path.insert(2, orig_dino_pth.as_posix())

import torch
import math

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

from prep_model import get_bb_name, time_str, get_backone_patch_embed_sizes #, get_dino_backbone
# from OrigDino.dinov2.eval.segmentation import models

# from MedDino.med_dinov2.models.segmentor import Segmentor
# from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ConvUNet
# from mmseg.models.decode_heads import *
from MedDino.med_dinov2.data.datasets import SegmentationDataset, SegmentationDatasetHDF5, VolDataModule
from torch.utils.data import DataLoader
# from MedDino.med_dinov2.tools.main_fcts import train, test
# from MedDino.med_dinov2.eval.metrics import mIoU, DiceScore
# from MedDino.med_dinov2.eval.losses import FocalLoss, DiceScore, CompositionLoss

# from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torchinfo import summary
# from torch.nn import CrossEntropyLoss
import wandb
# from MedDino.med_dinov2.tools.checkpointer import Checkpointer
from MedDino.med_dinov2.eval.losses import * 
from MedDino.med_dinov2.models.lit_segmentor import LitSegmentor
import os
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything


cluster_paths = True
save_checkpoints = True
log_the_run = True

gpus=torch.cuda.device_count()
strategy='ddp' if gpus>1 else 'auto'

seed = 42

# Set the BB
train_backbone = False
backbone_sz = "small" # in ("small", "base", "large" or "giant")

# Select dataset
dataset = 'hcp1' # 'hcp2' , cardiac_acdc, cardiac_rvsc, prostate_nci, prostate_usz, abide_caltech, abide_stanford
hdf5_data = True

# Select the dec head
dec_head_key = 'lin'  # 'lin', 'fcn', 'psp', 'da', 'resnet', 'unet'

# Select loss
loss_cfg_key = 'ce'  # 'ce', 'dice', 'dice_ce', 'focal', 'focal_dice'

# Training hyperparameters
nb_epochs = 2
warmup_iters = max(1, int(nb_epochs*0.2))  # try *0.25

# Config the batch size and lr for training
lr = 0.5e-4  # 0.5e-4
weigh_loss_bg = False  # False is better

# Test checkpoint
test_checkpoint_key = 'val_dice'  # 'val_loss', 'val_dice', 'val_mIoU'

# Dataloader workers
# num_workers_dataloader = min(os.cpu_count(), torch.cuda.device_count()*8)
num_workers_dataloader=3

# Set the precision
precision = 'highest' if cluster_paths else 'high'  # medium
torch.set_float32_matmul_precision(precision)


########################################################################################################################

# Set seeds for numpy, torch and python.random
seed_everything(seed, workers=True)

# Backbone config
backbone_name = get_bb_name(backbone_sz)
bb_checkpoint_path = dino_main_pth/f'Checkpoints/Orig/backbone/{backbone_name}_pretrain.pth'

dino_bb_cfg = dict(backbone_name=backbone_name, backbone_cp=bb_checkpoint_path)
patch_sz, embed_dim = get_backone_patch_embed_sizes(backbone_name)

# Dataset parameters
if dataset=='hcp1':
    batch_sz = 8//gpus  # [4, 8, 16, ...]
    if not hdf5_data:
        data_path_suffix = 'brain/hcp1'
    else:
        data_path_suffix = 'brain/hcp'
        hdf5_train_name = 'data_T1_original_depth_256_from_0_to_20.hdf5'
        hdf5_val_name = 'data_T1_original_depth_256_from_20_to_25.hdf5'
        hdf5_test_name = 'data_T1_original_depth_256_from_50_to_70.hdf5'
    num_classses = 15
    vol_depth = 256
    ignore_idx_loss = None
    ignore_idx_metric = 0
    
elif dataset=='hcp2':
    batch_sz = 8//gpus  # [4, 8, 16, ...]
    if not hdf5_data:
        data_path_suffix = 'brain/hcp2'
    else:
        data_path_suffix = 'brain/hcp'
        hdf5_train_name = 'data_T2_original_depth_256_from_0_to_20.hdf5'
        hdf5_val_name = 'data_T2_original_depth_256_from_20_to_25.hdf5'
        hdf5_test_name = 'data_T2_original_depth_256_from_50_to_70.hdf5'
    num_classses = 15
    vol_depth = 256
    ignore_idx_loss = None
    ignore_idx_metric = 0
    
elif dataset=='abide_caltech':
    batch_sz = 8//gpus  # [4, 8, 16, ...]
    if not hdf5_data:
        data_path_suffix = 'brain/abide_caltech'
    else:
        data_path_suffix = 'brain/abide/caltech'
        hdf5_train_name = 'data_T1_original_depth_256_from_0_to_10.hdf5'
        hdf5_val_name = 'data_T1_original_depth_256_from_10_to_15.hdf5'
        hdf5_test_name = 'data_T1_original_depth_256_from_16_to_36.hdf5'
    num_classses = 15
    vol_depth = 256
    ignore_idx_loss = None
    ignore_idx_metric = 0
    
elif dataset=='abide_stanford':
    batch_sz = 6//gpus  # [4, 8, 16, ...]
    if not hdf5_data:
        data_path_suffix = 'brain/abide_stanford'
    else:
        data_path_suffix = 'brain/abide/stanford'
        hdf5_train_name = 'data_T1_original_depth_132_from_0_to_10.hdf5'
        hdf5_val_name = 'data_T1_original_depth_132_from_10_to_15.hdf5'
        hdf5_test_name = 'data_T1_original_depth_132_from_16_to_36.hdf5'
    num_classses = 15
    vol_depth = 132
    ignore_idx_loss = None
    ignore_idx_metric = 0
    
elif dataset=='cardiac_acdc':
    data_path_suffix = 'cardiac/acdc'
    num_classses = 2
    vol_depth = 256 #@TODO VERIFY
    ignore_idx_loss = None
    ignore_idx_metric = 0
    if hdf5_data:
        ValueError('HDF5 path is not defined yet')      
         
elif dataset=='cardiac_rvsc':
    data_path_suffix = 'cardiac/rvsc'
    num_classses = 2
    vol_depth = 256 #@TODO VERIFY
    ignore_idx_loss = None
    ignore_idx_metric = 0
    if hdf5_data:
        ValueError('HDF5 path is not defined yet')
    
elif dataset=='prostate_nci':
    batch_sz = 10//gpus  # [4, 8, 16, ...]
    data_path_suffix = 'prostate/nci'
    num_classses = 3
    vol_depth = 20  #@TODO VERIFY
    ignore_idx_loss = None
    ignore_idx_metric = 0
    if hdf5_data:
        ValueError('HDF5 path is not defined yet')
        
elif dataset=='prostate_usz':
    batch_sz = 10//gpus  # [4, 8, 16, ...]
    data_path_suffix = 'prostate/pirad_erc'
    num_classses = 3
    vol_depth = 21 #@TODO VERIFY
    ignore_idx_loss = None
    ignore_idx_metric = 0
    if hdf5_data:
        ValueError('HDF5 path is not defined yet')
    
else:
    ValueError(f'Dataset: {dataset} is not defined')


# Decoder config
n_concat = 4
# Linear classification of each patch + upsampling to pixel dim
dec_head_cfg_conv_lin = dict(in_channels=[embed_dim]*n_concat, 
                             num_classses=num_classses,
                             out_upsample_fac=patch_sz,
                             bilinear=True)

# https://arxiv.org/abs/1411.4038
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

# https://arxiv.org/abs/1612.01105
dec_head_cfg_psp = dict(pool_scales=(1, 2, 3, 6),
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

# https://arxiv.org/abs/1809.02983
dec_head_cfg_da = dict(pam_channels=embed_dim,
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

# ResNet-like with recurrent convs
dec_head_cfg_resnet = dict(in_channels=[embed_dim]*n_concat,
                        num_classses=num_classses,
                        # in_index=None,
                        # in_resize_factors=None,
                        # align_corners=False,
                        dropout_rat_cls_seg=0.1,
                        nb_up_blocks=4,
                        upsample_facs_ch=2,
                        upsample_facs_wh=2,
                        bilinear=False,
                        conv_per_up_blk=2,
                        res_con=True,
                        res_con_interv=None, # None = Largest possible (better)
                        skip_first_res_con=False, 
                        recurrent=True,
                        recursion_steps=2)

# https://arxiv.org/abs/1505.04597 (unet papaer)
n_concat=5
input_group_cat_nb = 2
n_concat *= input_group_cat_nb
dec_head_cfg_unet = dict(in_channels=[embed_dim]*n_concat,
                        num_classses=num_classses,
                        # in_index=None,
                        # in_resize_factors=None,
                        # align_corners=False,
                        dropout_rat_cls_seg=0.1,
                        nb_up_blocks=4,
                        upsample_facs_ch=2,
                        upsample_facs_wh=2,
                        bilinear=False,
                        conv_per_up_blk=2, # 5
                        res_con=True,
                        res_con_interv=None, # None = Largest possible (better)
                        skip_first_res_con=False, 
                        recurrent=True,
                        recursion_steps=2, # 3
                        resnet_cat_inp_upscaling=True,
                        input_group_cat_nb=input_group_cat_nb)

decs_dict = dict(lin=dict(name='ConvHeadLinear', params=dec_head_cfg_conv_lin),
                 fcn=dict(name='FCNHead', params=dec_head_cfg_fcn),
                 psp=dict(name='PSPHead', params=dec_head_cfg_psp),
                 da=dict(name='DAHead', params=dec_head_cfg_da),
                 resnet=dict(name='ResNetHead', params=dec_head_cfg_resnet),
                 unet=dict(name='UNetHead', params=dec_head_cfg_unet))

# Choose the decode head config
assert dec_head_key in decs_dict.keys()
dec_head_cfg = decs_dict[dec_head_key]


# Optimizer Config
optm_cfg = dict(name='AdamW',
                params=dict(lr = lr,
                            weight_decay = 0.5e-4,   # 0.5e-4  | 1e-2
                            betas = (0.9, 0.999)))

# LR scheduler config
scheduler_configs = []
scheduler_configs.append(\
    dict(name='LinearLR',
         params=dict(start_factor=1/3, end_factor=1.0, total_iters=warmup_iters)))
scheduler_configs.append(\
    dict(name='PolynomialLR',
         params=dict(power=1.0, total_iters=(nb_epochs-warmup_iters)*2)))

scheduler_cfg = dict(name='SequentialLR',
                     params=dict(scheduler_configs=scheduler_configs,
                                  milestones=[warmup_iters]),
                    )

# Loss Config
epsilon = 1  # smoothing factor 
k=1  # power
weight = [0.1] + [1.]*(num_classses-1)
weight = torch.Tensor(weight)

# CE Loss
loss_cfg_ce = dict(ignore_index=ignore_idx_loss if ignore_idx_loss is not None else -100,
                   weight=weight if weigh_loss_bg else None)

# Dice Loss
loss_cfg_dice = dict(prob_inputs=False, 
                    bg_ch_to_rm=ignore_idx_loss, # not removing results in better segmentation
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
loss_cfg_focal = dict(bg_ch_to_rm=ignore_idx_loss,
                      gamma=2,
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
                      focal_dice=dict(name='CompositionLoss', params=loss_cfg_comp_foc_dice))

assert loss_cfg_key in loss_cfgs_dict.keys()
loss_cfg = loss_cfgs_dict[loss_cfg_key] 


# Metrics
epsilon = 1  # smoothing factor 
k=1  # power

assert vol_depth % batch_sz == 0, \
    f'batch size must be a multiple of slice/patient but got {batch_sz} and {vol_depth}'

miou_cfg=dict(prob_inputs=False, # Decoder does not return probas explicitly
              soft=False,
              bg_ch_to_rm=ignore_idx_metric,  # bg channel to be removed 
              reduction='mean',
              vol_batch_sz=vol_depth,
              epsilon=epsilon)

dice_cfg=dict(prob_inputs=False,  # Decoder does not return probas explicitly
             soft=False,
             bg_ch_to_rm=ignore_idx_metric,
             reduction='mean',
             k=k, 
             epsilon=epsilon,
             vol_batch_sz=vol_depth)

metric_cfgs=[dict(name='mIoU', params=miou_cfg), 
             dict(name='dice', params=dice_cfg)]


# Seg result logging cfg
seg_res_log_itv = max(nb_epochs//5, 1)   # Log seg reult every N epochs
seg_res_nb_patient = 1  # Process minibatches for N number of patients
seg_log_per_batch = 4  # Log N samples from each minibatch
seg_log_nb_batches = 16
assert seg_log_per_batch<=batch_sz

# Which samles in the minibatch to log
sp = seg_log_per_batch+1
multp = batch_sz//sp
# maximal separation from each other and from edges (from edges is prioritized)
if multp>0:
    log_idxs = torch.arange(multp, 
                            multp*sp, 
                            multp)
    log_idxs = log_idxs + (batch_sz%sp)//2
else:
    log_idxs = torch.arange(0, 
                            batch_sz, 
                            1)
log_idxs = log_idxs.tolist()

# batch indexes to log
step = max(vol_depth//batch_sz//seg_log_nb_batches, 1)
seg_log_batch_idxs = torch.arange(0+step-1, min(seg_log_nb_batches*step, vol_depth//batch_sz*seg_res_nb_patient), step).tolist()
assert len(seg_log_batch_idxs)==seg_log_nb_batches

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
                     seg_log_batch_idxs=seg_log_batch_idxs,
                     minibatch_log_idxs=log_idxs,
                     seg_val_intv=seg_res_log_itv,
                     sync_dist_train=gpus>1,
                     sync_dist_val=gpus>1,
                     sync_dist_test=gpus>1)

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
    if hdf5_data:
        data_root_pth = data_root_pth / 'hdf5'
    
data_root_pth = data_root_pth / data_path_suffix

if not hdf5_data:
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
    
else:
    train_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_train_name, 
                                            num_classes=num_classses, 
                                            augmentations=train_augmentations)
    val_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_val_name, 
                                            num_classes=num_classses, 
                                            augmentations=augmentations)
    test_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_test_name, 
                                            num_classes=num_classses, 
                                            augmentations=augmentations)

                                            
                                             
persistent_workers=True
pin_memory=True
drop_last=True

train_dataloader_cfg = dict(batch_size=batch_sz, shuffle=True, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                            persistent_workers=persistent_workers, drop_last=drop_last)
val_dataloader_cfg = dict(batch_size=batch_sz, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                          persistent_workers=persistent_workers, drop_last=drop_last, 
                          shuffle=False if gpus==1 else None)
test_dataloader_cfg = dict(batch_size=batch_sz, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                           persistent_workers=persistent_workers, drop_last=drop_last, 
                           shuffle=False)  # if gpus==1 else None

# train_dataloader = DataLoader(dataset=train_dataset, **train_dataloader_cfg)
# val_dataloader = DataLoader(dataset=val_dataset,  **val_dataloader_cfg)
# test_dataloader = DataLoader(dataset=test_dataset, **test_dataloader_cfg)

data_module = VolDataModule(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, 
                            train_dataloader_cfg=train_dataloader_cfg, val_dataloader_cfg=val_dataloader_cfg, test_dataloader_cfg=test_dataloader_cfg,
                            vol_depth=vol_depth, num_gpus=gpus)

# Trainer config (loggable components)
trainer_cfg = dict(accelerator='gpu', devices=gpus, sync_batchnorm=True, strategy=strategy,
                   max_epochs=nb_epochs, log_every_n_steps=100, num_sanity_val_steps=0,
                   enable_checkpointing=True, 
                   gradient_clip_val=0, gradient_clip_algorithm='norm',  # Gradient clipping by norm/value
                   accumulate_grad_batches=1) #  runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN.


# Trainer cfg for testing
trainer_cfg_test = dict(accelerator='gpu', devices=1, 
                        log_every_n_steps=100, num_sanity_val_steps=0,) #  runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN.


# Init the logger (wandb)
loss_name = loss_cfg['name'] if not loss_cfg['name']=='CompositionLoss' else \
                f'{loss_cfg["params"]["loss1"]["name"]}{loss_cfg["params"]["loss2"]["name"]}Loss'
dec_head_name = model.model.decode_head.__class__.__name__
bb_train_str_short = 'bbT' if segmentor_cfg['train_backbone'] else 'NbbT'
run_name = f'{dataset}_{backbone_name}_{bb_train_str_short}_{dec_head_key}_{loss_cfg_key}'
data_type = 'hdf5' if hdf5_data else 'png'

wnadb_config = dict(backbone_name=backbone_name,
                    backbone_last_n_concat=n_concat,
                    decode_head=dec_head_name,
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
                    torch_precision=precision,
                    train_dataloader_cfg=train_dataloader_cfg,
                    val_dataloader_cfg=val_dataloader_cfg,
                    test_dataloader_cfg=test_dataloader_cfg,
                    trainer_cfg=trainer_cfg,
                    nb_gpus=gpus,
                    strategy=strategy,
                    data_type=data_type)

wandb_log_path = dino_main_pth / 'Logs'
wandb_log_path.mkdir(parents=True, exist_ok=True)

bb_train_str = 'train_bb_YES' if segmentor_cfg['train_backbone'] else 'train_bb_NO'
log_mode = 'online' if log_the_run else 'disabled'
tags = [dataset, loss_name, bb_train_str, dec_head_name]
tags.extend(backbone_name.split('_'))
tags.append(data_type)

logger = WandbLogger(project='FoundationModels_MedDino',
                     group=backbone_name,
                     config=wnadb_config,
                     dir=wandb_log_path,
                     name=run_name,
                     mode=log_mode,
                     settings=wandb.Settings(_service_wait=300),  # Can increase timeout
                     tags=tags)

# log gradients, parameter histogram and model topology
logger.watch(model, log="all")

n_best = 1 if save_checkpoints else 0
models_pth = dino_main_pth / f'Checkpoints/MedDino/{model.model.decode_head.__class__.__name__}'
models_pth.mkdir(parents=True, exist_ok=True)
time_s = time_str()
checkpointers = dict(val_loss = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_loss", mode='min', filename=time_s+'-{epoch}-{val_loss:.2f}'),
                     val_dice = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_dice", mode='max', filename=time_s+'-{epoch}-{val_dice:.2f}'),
                     val_mIoU = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_mIoU", mode='max', filename=time_s+'-{epoch}-{val_mIoU:.2f}'))

# Create the trainer object
trainer = L.Trainer(logger=logger, callbacks=list(checkpointers.values()), **trainer_cfg)

# Train the model
# model is saved only on the main process when using distributed training
trainer.fit(model=model, datamodule=data_module)#train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

torch.distributed.destroy_process_group()
if trainer.global_rank == 0:
    trainer = L.Trainer(logger=logger, **trainer_cfg_test)
    # Load the best checkpoint (highest val_dice)
    model = LitSegmentor.load_from_checkpoint(checkpoint_path=checkpointers[test_checkpoint_key].best_model_path, **segmentor_cfg)
    logs = trainer.test(model=model, datamodule=data_module)  # dataloaders=test_dataloader,

print('Done !')
#finish logging
wandb.finish()
print('***END***')


#@TODO  multi GPU (performance optm) (metrics per volume are problematic)
#@TODO add types and comments
#@TODO write readme.md
#@TODO use the original hdf5 files 



# Try training with plain-vanilla SGD (no momentum nor weight decay).
# Start with a low learning rate. Can you train successfully, even if slowly?
# If so, try increasing the learning rate and possibly turning on momentum.

# lr = 0.1
# optimizer = optim.Adam(model.parameters(), lr = lr)
# Although it often trains faster, Adam can be unstable sometimes.

# Also, as a general rule, the learning rate with which Adam can train
# stably tends to be numerically significantly smaller than those that
# work with SGD. I would suggest starting with lr = 1.e-6 and increasing
# it until you either get successful, if slow, training, or unstable training.
