
import sys
from pathlib import Path
 
med_seg_path = Path(__file__).parent
main_pth = med_seg_path.parent
med_seg_mod_pth = med_seg_path / 'med_seg_foundation'

orig_models_pth = main_pth / 'OrigModels' 
dino_mod_pth = orig_models_pth / 'DinoV2' 
sam_mod_pth = orig_models_pth / 'SAM' 

sys.path.insert(0, str(main_pth))
sys.path.insert(1, str(med_seg_mod_pth))
sys.path.insert(2, str(dino_mod_pth))
sys.path.insert(3, str(sam_mod_pth))

import torch
# import math
from enum import Enum
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

from ModelSpecific.DinoMedical.prep_model import time_str
# from prep_model import get_bb_name, time_str, get_backone_patch_embed_sizes #, get_dino_backbone
# from OrigDino.dinov2.eval.segmentation import models

# from MedDino.med_dinov2.models.segmentor import Segmentor
# from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ConvUNet
# from mmseg.models.decode_heads import *

from med_seg_foundation.data.datasets import VolDataModule # SegmentationDataset, SegmentationDatasetHDF5,
from torch.utils.data import DataLoader
# from MedDino.med_dinov2.tools.main_fcts import train, test
# from MedDino.med_dinov2.eval.metrics import mIoU, DiceScore
# from MedDino.med_dinov2.eval.losses import FocalLoss, DiceScore, CompositionLoss

# from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torchinfo import summary
# from torch.nn import CrossEntropyLoss
import wandb
# from MedDino.med_dinov2.tools.checkpointer import Checkpointer
from med_seg_foundation.eval.losses import * 
from med_seg_foundation.models.lit_segmentor import LitSegmentor
import os
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything

from med_seg_foundation.tools.configs import *
# from torch.utils.data import Sampler
from MedicalSegmentation.med_seg_foundation.models.segmentor import Segmentor
from MedicalSegmentation.med_seg_foundation.models.unet import UNet

cluster_paths = True
save_checkpoints = True
log_the_run = True

class ModelType(Enum):
    SEGMENTOR=1
    UNET=2

# Select model type
model_type = ModelType.UNET

if model_type == ModelType.SEGMENTOR:
    # Set the BB
    backbone = 'sam'  # sam, resnet
    train_backbone = True
    backbone_sz = "base" # in ("small", "base", "large" or "giant")
    
    # Select the dec head
    dec_head_key = 'unet'  # 'lin', 'fcn', 'psp', 'da', 'segformer', 'resnet', 'unet'
    

# Select dataset
dataset = 'hcp2' # 'hcp1', 'hcp2', abide_caltech, abide_stanford, prostate_nci, prostate_usz, cardiac_acdc, cardiac_rvsc, 
hdf5_data = True

# Select loss
loss_cfg_key = 'ce'  # 'ce', 'dice', 'dice_ce', 'focal', 'focal_dice'

# Training hyperparameters
nb_epochs = 100 if cluster_paths else 2
warmup_iters = max(1, int(nb_epochs*0.2))  # try *0.25

# Config the batch size and lr for training
lr = 0.5e-4  # 0.5e-4
weigh_loss_bg = False  # False is better

# Test checkpoint
test_checkpoint_key = 'val_dice_vol'  # 'val_loss', 'val_dice_vol', 'val_mIoU_vol'

# Dataloader workers
# num_workers_dataloader = min(os.cpu_count(), torch.cuda.device_count()*8)
num_workers_dataloader=3

brain_datasets = ['hcp1', 'hcp2', 'abide_caltech']
prostate_datasets = ['prostate_nci', 'prostate_usz']

if cluster_paths:
    if dataset in brain_datasets:
        test_datasets = brain_datasets
    elif dataset in prostate_datasets:
        test_datasets = prostate_datasets
    else:
        test_datasets = [dataset] 
else:
    test_datasets = [dataset]

####################################################################################################
seed = 42

gpus=torch.cuda.device_count()
strategy='ddp' if gpus>1 else 'auto'

print(f'{gpus} available GPUs')
for gpu_i in range(gpus):
    print(f'GPU{gpu_i}: {torch.cuda.get_device_name(gpu_i)}')

# Get device 
current_dev_idx = torch.cuda.current_device()
device = torch.cuda.device(current_dev_idx)

# Get device properties
props = torch.cuda.get_device_properties(device)
print(f'Current device is GPU{current_dev_idx}: {torch.cuda.get_device_name(current_dev_idx)}')
# Convert to gigabytes (GB) for readability
total_memory_gb = props.total_memory / (1024**3)
formatted_total_memory_gb = "{:.2f}".format(total_memory_gb)
print("Total VRAM:", formatted_total_memory_gb, "GB")

if props.major >= 7:
    print("GPU has tensor cores (Volta architecture or newer).")
    tensor_cores = True
else:
    print("GPU does not have tensor cores.")
    tensor_cores = False

# Set the precision
precision = 'highest' if not tensor_cores else 'high'  # medium
torch.set_float32_matmul_precision(precision)


########################################################################################################################


# Set seeds for numpy, torch and python.random
seed_everything(seed, workers=True)

# Data attributes
dataset_attrs = get_data_attrs(name=dataset, use_hdf5=hdf5_data)
batch_sz = get_batch_sz(dataset_attrs, gpus)

if model_type==ModelType.SEGMENTOR:
    # Backbone config
    bb_cfg = get_bb_cfg(bb_name=backbone, bb_size=backbone_sz, train_bb=train_backbone, 
                        dec_name=dec_head_key, main_pth=main_pth, pretrained=True)

    # Decoder config
    dec_head_cfg, n_in_ch = get_dec_cfg(dec_name=dec_head_key, dataset_attrs=dataset_attrs)


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
loss_cfg = get_loss_cfg(loss_key=loss_cfg_key, data_attr=dataset_attrs)

# Metrics
metric_cfgs = get_metric_cfgs(data_attr=dataset_attrs)

# Log indexes for segmentation for the minibatch
log_idxs = get_minibatch_log_idxs(batch_sz=batch_sz)

# Log indexes for segmentation for the batches
seg_log_batch_idxs = get_batch_log_idxs(batch_sz=batch_sz, data_attr=dataset_attrs)

 # Log seg val reult every N epochs during training
seg_res_log_itv = max(nb_epochs//5, 1)  

# Init the segmentor model
if model_type == ModelType.SEGMENTOR:
    segmentor_cfg = dict(name=Segmentor.__name__,
                         params=dict(backbone=bb_cfg,
                                     decode_head=dec_head_cfg,
                                     reshape_dec_oup=True,
                                     align_corners=False))
    
elif model_type == ModelType.UNET:
    segmentor_cfg = dict(name=UNet.__name__,
                         params=dict(n_channels=3, 
                                     n_classes=dataset_attrs['num_classses'], 
                                     bilinear=False))
    
else:
    ValueError(f'Model type: {model_type} is not found')

segmentor_cfg_lit = dict(segmentor=segmentor_cfg,
                         loss_config=loss_cfg, 
                         optimizer_config=optm_cfg,
                         schedulers_config=scheduler_cfg,
                         metric_configs=metric_cfgs,
                         val_metrics_over_vol=True, # Also report metrics over vol
                         seg_log_batch_idxs=seg_log_batch_idxs,
                         minibatch_log_idxs=log_idxs,
                         seg_val_intv=seg_res_log_itv,
                         sync_dist_train=gpus>1,
                         sync_dist_val=gpus>1,
                         sync_dist_test=gpus>1)

model = LitSegmentor(**segmentor_cfg_lit)

# Print model info
summary(model)

# Get augmentations
augmentations = get_augmentations()

# Get data pre-processing
if model_type==ModelType.SEGMENTOR:
    processings = model.segmentor.backbone.get_pre_processing_cfg_list()
else:
    processings = [dict(type='Normalize', 
                               mean=[123.675, 116.28, 103.53],  #RGB
                               std=[58.395, 57.12, 57.375],  #RGB
                               to_rgb=True)]

# Get the data loader
if cluster_paths:
    data_root_pth = Path('/usr/bmicnas02/data-biwi-01/foundation_models/da_data') 
else:      
    data_root_pth = main_pth.parent / 'DataFoundationModels'
    if hdf5_data:
        data_root_pth = data_root_pth / 'hdf5'
        
# Get datasets
train_dataset, val_dataset, dataset_name_testing = get_datasets(data_root_pth=data_root_pth, 
                                                        hdf5_data=hdf5_data, 
                                                        data_attr=dataset_attrs, 
                                                        train_procs=augmentations+processings, 
                                                        val_test_procs=processings)
                                            
# Dataloader configs                                          
persistent_workers=True
pin_memory=True
drop_last=False

train_dataloader_cfg = dict(batch_size=batch_sz, shuffle=True, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                            persistent_workers=persistent_workers, drop_last=drop_last)
val_dataloader_cfg = dict(batch_size=batch_sz, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                          persistent_workers=persistent_workers, drop_last=drop_last, 
                          shuffle=False if gpus==1 else None)

data_module = VolDataModule(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=dataset_name_testing, 
                            train_dataloader_cfg=train_dataloader_cfg, val_dataloader_cfg=val_dataloader_cfg, 
                            vol_depth=dataset_attrs['vol_depth'], num_gpus=gpus)

# Trainer config (loggable components)
trainer_cfg = dict(accelerator='gpu', devices=gpus, sync_batchnorm=True, strategy=strategy,
                   max_epochs=nb_epochs, log_every_n_steps=100, num_sanity_val_steps=0,
                   enable_checkpointing=True, 
                   gradient_clip_val=0, gradient_clip_algorithm='norm',  # Gradient clipping by norm/value
                   accumulate_grad_batches=1) #  runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN.

# Init the logger (wandb)
loss_name = loss_cfg['name'] if not loss_cfg['name']=='CompositionLoss' else \
                f'{loss_cfg["params"]["loss1"]["name"]}{loss_cfg["params"]["loss2"]["name"]}Loss'
data_type = 'hdf5' if hdf5_data else 'png'

# Tags
tags = [dataset, loss_name, data_type]

if model_type==ModelType.SEGMENTOR:
    dec_head_name = model.segmentor.decode_head.__class__.__name__
    backbone_name = model.segmentor.backbone.name
    bb_train_str_short = 'bbT' if train_backbone else 'NbbT'
    wnadb_config_add = dict(dec_head_name = dec_head_name,
                            dec_head_cfg=dec_head_cfg,
                            backbone_name = backbone_name,
                            bb_cfg=bb_cfg,
                            backbone_n_in_ch=n_in_ch,
                            )
    run_name = f'{dataset}_{backbone_name}_{bb_train_str_short}_{dec_head_key}_{loss_cfg_key}'
    bb_train_str = 'train_bb_YES' if train_backbone else 'train_bb_NO'
    tags.append(bb_train_str)
    tags.append(dec_head_name)
    tags.extend(backbone_name.split('_'))
    group_name = backbone_name
    
    
elif model_type==ModelType.UNET:
    group_name = UNet.__name__ + 'Model'
    run_name = f'{dataset}_{group_name}_{loss_cfg_key}'
    
else:
    ValueError(f'Model type: {model_type} is not found')


wnadb_config = dict(segmentor_cfg_lit=segmentor_cfg_lit,
                    dataset=str(data_root_pth),
                    batch_sz=batch_sz,
                    num_classes=dataset_attrs['num_classses'],
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
                    test_datasets=test_datasets,
                    trainer_cfg=trainer_cfg,
                    nb_gpus=gpus,
                    precision=precision,
                    strategy=strategy,
                    data_type=data_type)

wandb_log_path = main_pth / 'Logs'
wandb_log_path.mkdir(parents=True, exist_ok=True)

log_mode = 'online' if log_the_run else 'disabled'
logger = WandbLogger(project='FoundationModels_MedDino',
                     group=group_name,
                     config=wnadb_config,
                     dir=wandb_log_path,
                     name=run_name,
                     mode=log_mode,
                     settings=wandb.Settings(_service_wait=300),  # Can increase timeout
                     tags=tags)

# log gradients, parameter histogram and model topology
logger.watch(model, log="all")

n_best = 1 if save_checkpoints else 0
models_pth = main_pth / f'Checkpoints/{group_name}'
if model_type == ModelType.SEGMENTOR:
    models_pth = models_pth / model.segmentor.decode_head.__class__.__name__
    
models_pth.mkdir(parents=True, exist_ok=True)
time_s = time_str()
checkpointers = dict(val_loss = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_loss", mode='min', filename=time_s+'-{epoch}-{val_loss:.2f}'),
                     val_dice_vol = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_dice_vol", mode='max', filename=time_s+'-{epoch}-{val_dice:.2f}'),
                     val_mIoU_vol = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_mIoU_vol", mode='max', filename=time_s+'-{epoch}-{val_mIoU:.2f}'))

# Create the trainer object
trainer = L.Trainer(logger=logger, callbacks=list(checkpointers.values()), **trainer_cfg)

# Train the model
# model is saved only on the main process when using distributed training
trainer.fit(model=model, datamodule=data_module)#train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Test and Validate on all the indicated datasets on a single GPU
if gpus>1:
    torch.distributed.destroy_process_group()
if trainer.global_rank == 0:
    # Trainer cfg for testing
    trainer_cfg_test = dict(accelerator='gpu', devices=1, 
                            log_every_n_steps=100, num_sanity_val_steps=0,)
    trainer_testing = L.Trainer(logger=logger, **trainer_cfg_test)
    
    test_model_cfg = dict(backbone=bb_cfg,
                          decode_head=dec_head_cfg,  # WRONG SHOULD BE CHANGED nb class changes
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
                          sync_dist_train=False,
                          sync_dist_val=False,
                          sync_dist_test=False)
    
    # Load the best checkpoint (highest val_dice_vol)
    model = LitSegmentor.load_from_checkpoint(checkpoint_path=checkpointers[test_checkpoint_key].best_model_path, **test_model_cfg)
    
    for i, dataset_name_testing in enumerate(test_datasets):
        print(f'Testing for dataset: {dataset_name_testing} {i+1}/{len(test_datasets)}')
        
        # Set the test_dataset name
        model._test_dataset_name = dataset_name_testing
        
        # Get attributes and the batch size for the given dataset
        dataset_attrs = get_data_attrs(name=dataset_name_testing, use_hdf5=hdf5_data)
        batch_sz = get_batch_sz(dataset_attrs, num_gpu=gpus)  # use the same batch size as ddp to ensure it will fit in the memory
        
        # Dataloader config for val and test on singl GPU
        test_dataloader_cfg = dict(batch_size=batch_sz, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                                persistent_workers=persistent_workers, drop_last=drop_last, 
                                shuffle=False)  
        
        # Get datasets
        _, val_dataset, dataset_name_testing = get_datasets(data_root_pth=data_root_pth, 
                                                                hdf5_data=hdf5_data, 
                                                                data_attr=dataset_attrs, 
                                                                train_procs=augmentations+processings, 
                                                                val_test_procs=processings)
        
        val_dataloader = DataLoader(dataset=val_dataset, **test_dataloader_cfg)
        test_dataloader = DataLoader(dataset=dataset_name_testing, **test_dataloader_cfg)
        
        # Validate on single GPU
        trainer_testing.validate(model=model, dataloaders=val_dataloader)
        
        # Test on single GPU
        trainer_testing.test(model=model, dataloaders=test_dataloader) 
        
    # Finish logging
    wandb.finish()
        
print('***END***')


#@TODO  multi GPU (performance optm) (metrics per volume are problematic)
#@TODO add types and comments
#@TODO write readme.md

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
