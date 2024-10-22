
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
from ModelSpecific.DinoMedical.prep_model import time_str
from torch.utils.data import DataLoader
from MedicalSegmentation.med_seg_foundation.data.datasets import VolDataModule
from torchinfo import summary
import wandb
from med_seg_foundation.utils.losses import * 
from MedicalSegmentation.med_seg_foundation.trainer.lit_trainer import LitTrainer
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from med_seg_foundation.configs import *
import socket
from MedicalSegmentation.med_seg_foundation.utils.tools import log_class_rel_freqs, get_ckp_path
import os

cluster_mode = 'KeremPC' != socket.gethostname()

cluster_paths = cluster_mode
save_checkpoints = cluster_mode
log_the_run = cluster_mode

# Select model type
model_type = ModelType.SEGMENTOR  # SEGMENTOR, UNET, SWINUNET, R2ATTNUNET

# Fully Test Time Adaptations (Entropy Minimization)
ftta = False

# Self training (Vanilla)
self_training = False

pseudo_label_update_intv=1
pseudo_lab_confidence_thres=0.9  # 0.9
nb_labeled_vol = 3 if self_training else None ## 3

if model_type == ModelType.SEGMENTOR: # Do not TOUCH !
    # Set the BB
    backbone = 'dino'  # dino, dinoReg, sam, medsam, mae, resnet
    train_backbone = False and not ftta and not self_training
    train_finetune = True and not ftta and not self_training # For reins and reins Lora
    
    backbone_sz = "base" if cluster_mode else "base" # in ("small", "base", "large" "huge" "giant")
    
    # Choose the FineTuning  # ladderR, ladderD, rein, reinL
    if backbone in ['dino', 'dinoReg']:
        fine_tune = 'reinL' 
    elif backbone=='sam':
        fine_tune = 'rein' 
    elif backbone=='medsam':
        fine_tune = 'ladderD' 
    elif backbone=='mae':
        fine_tune = ''
    else:
        fine_tune = ''
        # raise ValueError(f'Best FT is not determined for {backbone} backbone yet !') 
                            
    backbone = f'{fine_tune}_{backbone}' if fine_tune != '' else backbone
    
    if 'rein' in fine_tune or 'ladder' in fine_tune:
        train_backbone = False
    
    if self_training:    
        backbone_to_load = backbone
    
    # Select the dec head
        # 'lin', 'fcn', 'psp', 'da', 'segformer', 'resnet', 'unet', 'unetS', 
        #'sam_mask_dec', 'hsam_mask_dec', 'hq_sam_mask_dec', 'hq_hsam_mask_dec'
    dec_head_key = 'unetS' if backbone=='resnet' else  'hq_hsam_mask_dec' # 'unetS'#  'hq_hsam_mask_dec'

# Select dataset
# 'hcp1', 'hcp2', abide_caltech, abide_stanford, 
# prostate_nci, prostate_usz, 
# cardiac_acdc, cardiac_rvsc, 
# spine_mrspinesegv, spine_verse
# BraTS_T1, BraTS_FLAIR

# Domain adaptation
if ftta or self_training:
    sd_dataset = 'abide_stanford'#'BraTS_FLAIR'  # To be loaded from saved checkpoints  spine_mrspinesegv  
    da_dataset = 'abide_caltech'#'BraTS_T1'
    dataset = da_dataset
    rcs_enabled = False
    
    assert sd_dataset != da_dataset
    
    if ftta:
        # Select loss
        loss_cfg_key = 'entropy_min'
        
        # Config the batch size
        batch_sz = 4 #32 ? 
        
        # Test checkpoint
        test_checkpoint_key = 'last'
        
        # Nb epochs
        nb_epochs = 2
        
    if self_training:
        # Select loss
        loss_cfg_key = 'ce'
        
        # Config the batch size
        batch_sz = 4 
        
        # Test checkpoint
        test_checkpoint_key = 'val_dice_vol'
        
        # Nb epochs
        nb_epochs = 20
 
# Source domain training    
else:
    dataset = 'prostate_usz'  #if cluster_paths else 'prostate_usz'
    rcs_enabled = True

    # Select loss
    loss_cfg_key = 'ce'  #'ce'  # 'ce', 'dice', 'dice_ce', 'focal', 'focal_dice'
    
    # Config the batch size
    batch_sz = 4#8 
    
    # Test checkpoint
    test_checkpoint_key = 'val_dice_vol'  # 'val_loss', 'val_dice_vol', 'val_mIoU_vol'
    
    # Nb epochs
    nb_epochs=120
    if model_type==ModelType.SEGMENTOR:
        if backbone in ["sam", "medsam"]:
            if dataset in ['hcp1', 'hcp2']:
                nb_epochs=120
            elif dataset in ['spine_verse']:
                if backbone=="sam":
                    nb_epochs=50
                else:
                    nb_epochs=85
        else:
            if dataset in ['spine_verse']:
                if backbone_sz != 'small':
                    nb_epochs=90
                else:
                    nb_epochs=100
    if 'BraTS' in dataset:
        nb_epochs=80


if not cluster_paths:
    nb_epochs = 2

####################################################################################################

# Dataloader workers
# num_workers_dataloader = min(os.cpu_count(), torch.cuda.device_count()*8)
num_workers_dataloader=3

brain_datasets = ['hcp1', 'hcp2', 'abide_caltech', 'abide_stanford']
prostate_datasets = ['prostate_nci', 'prostate_usz'] #if backbone != 'medsam' else ['prostate_usz']
spine_datasets = ['spine_mrspinesegv', 'spine_verse']
brats_datasets = ['BraTS_T1', 'BraTS_FLAIR']

if dataset in brain_datasets:
    test_datasets = brain_datasets
elif dataset in prostate_datasets:
    test_datasets = prostate_datasets
elif dataset in spine_datasets:
    test_datasets = spine_datasets
elif dataset in brats_datasets:
    test_datasets = brats_datasets
else:
    test_datasets = [dataset] 
    
    
if ftta:
    # Verification for fully test time adaptations
    if sd_dataset in brain_datasets:
        assert da_dataset in brain_datasets
    elif sd_dataset in prostate_datasets:
        assert da_dataset in prostate_datasets
    elif sd_dataset in spine_datasets:
        assert da_dataset in spine_datasets
    elif sd_dataset in brats_datasets:
        assert da_dataset in brats_datasets
    else:
        raise ValueError(f'Could not find a group of datasets for SD and TD: {sd_dataset}, {da_dataset}')

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

batch_sz = batch_sz // gpus  if strategy == 'ddp' else batch_sz

# Data attributes
dataset_attrs = get_data_attrs(name=dataset, use_hdf5=None, rcs_enabled=rcs_enabled)

# Init the segmentor model
if model_type == ModelType.SEGMENTOR:
    kwargs = dict(backbone=backbone,
                  backbone_sz=backbone_sz,
                  train_backbone=train_backbone,
                  dec_head_key=dec_head_key,
                  main_pth=main_pth)
    if 'rein' in backbone or 'ladder' in backbone:
        kwargs['train_finetune'] = train_finetune
else:
    kwargs=dict()
    
segmentor_cfg_lit = get_lit_trainer_cfg(batch_sz=batch_sz, nb_epochs=nb_epochs, loss_cfg_key=loss_cfg_key, 
                                          dataset_attrs=dataset_attrs, gpus=gpus, model_type=model_type, 
                                          ftta=ftta, self_training=self_training, 
                                          pseudo_label_update_intv=pseudo_label_update_intv,
                                          pseudo_lab_confidence_thres=pseudo_lab_confidence_thres, **kwargs)

if not ftta and not self_training:
    model = LitTrainer(**segmentor_cfg_lit)
else:
    if cluster_paths:
        # search_dir_ = '/scratch_net/biwidl210_second/kcekmeceli/Checkpoints'
        search_dir_ = '/usr/bmicnas02/data-biwi-01/foundation_models/AllCheckpoints'
        dirs = os.listdir(search_dir_)
        search_dir = [os.path.join(search_dir_, dir) for dir in dirs if os.path.isdir(os.path.join(search_dir_, dir))]
    else:
        search_dir = str(main_pth / 'Checkpoints')
    
    if model_type==ModelType.SEGMENTOR:
        sd_model_ckp_pth = get_ckp_path(search_dir=search_dir, dataset=sd_dataset, model_type=model_type,
                                        bb_size=backbone_sz, backbone=backbone_to_load, dec_name=dec_head_key)
    else:
        sd_model_ckp_pth = get_ckp_path(search_dir=search_dir, dataset=sd_dataset, model_type=model_type)
    
    model = LitTrainer.load_from_checkpoint(checkpoint_path=sd_model_ckp_pth, **segmentor_cfg_lit, strict=False)

# Get augmentations
augmentations = get_augmentations()

# Get data pre-processing
if model_type==ModelType.SEGMENTOR:
    processings = model.segmentor.backbone.get_pre_processing_cfg_list()
else:
    processings = [dict(type='Normalize', 
                        mean=[0., 0., 0.],  #RGB
                        std=[255.0, 255.0, 255.0],  #RGB
                        to_rgb=True)]

# Get the data loader
if cluster_paths:
    data_root_pth = Path('/usr/bmicnas02/data-biwi-01/foundation_models/da_data') 
else:      
    data_root_pth = main_pth.parent / 'DataFoundationModels'
    if dataset_attrs['format']=='hdf5':
        data_root_pth = data_root_pth / 'hdf5'
        
# Get datasets
train_dataset, val_dataset, _ = get_datasets(data_root_pth=data_root_pth, 
                                                        data_attr=dataset_attrs, 
                                                        train_procs=augmentations+processings, 
                                                        val_test_procs=processings,
                                                        ftta=ftta, nb_labeled_vol=nb_labeled_vol)    
                             
# Dataloader configs                                          
persistent_workers=True
pin_memory=True
drop_last=False

train_dataloader_cfg = dict(batch_size=batch_sz, shuffle=True, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                            persistent_workers=persistent_workers, drop_last=drop_last)
val_dataloader_cfg = dict(batch_size=batch_sz, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                          persistent_workers=persistent_workers, drop_last=drop_last, 
                          shuffle=False if gpus==1 else None)

data_module = VolDataModule(train_dataset=train_dataset, val_dataset=val_dataset, #test_dataset=None, 
                            train_dataloader_cfg=train_dataloader_cfg, val_dataloader_cfg=val_dataloader_cfg, 
                            vol_depth=None, num_gpus=gpus) 

# Trainer config (loggable components)
trainer_cfg = dict(accelerator='gpu', devices=gpus, sync_batchnorm=True, strategy=strategy,
                   max_epochs=nb_epochs, log_every_n_steps=100, num_sanity_val_steps=0,
                   enable_checkpointing=True, 
                   gradient_clip_val=0, gradient_clip_algorithm='norm',  # Gradient clipping by norm/value
                   accumulate_grad_batches=1) #  runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN.

# Init the logger (wandb)
loss_name = segmentor_cfg_lit['loss_config']['name'] if not segmentor_cfg_lit['loss_config']['name']=='CompositionLoss' else \
                f'{segmentor_cfg_lit["loss_config"]["params"]["loss1"]["name"]}{segmentor_cfg_lit["loss_config"]["params"]["loss2"]["name"]}Loss'

# Tags
tags = [dataset, loss_name, dataset_attrs['format']]
if ftta or self_training:
    DA_info = dict(sd_dataset=sd_dataset,
                   da_dataset=da_dataset,
                   sd_model_ckp_pth=sd_model_ckp_pth)
    if ftta:
        name_prefix = 'FTTA'
        tags.append('FTTA')
    elif self_training:
        name_prefix = 'ST'
        tags.append('SelfTraining')
        DA_info['nb_labeled_vol'] = nb_labeled_vol
        DA_info['pseudo_label_update_intv'] = pseudo_label_update_intv
        DA_info['pseudo_lab_confidence_thres'] = pseudo_lab_confidence_thres
        
    else:
        raise ValueError()
    
    
    
wandb_run_dataset = dataset if (not ftta and not self_training) else f'{name_prefix}_{sd_dataset}_to_{da_dataset}'

if model_type==ModelType.SEGMENTOR:
    dec_head_name = model.segmentor.decode_head.__class__.__name__
    backbone_name = model.segmentor.backbone.name
    bb_train_str_short = 'bbT' if train_backbone else 'NbbT'
    if 'rein' in fine_tune or 'ladder' in  fine_tune:
        ft_train_str_short = 'ftT' if train_finetune else 'NftT'
        run_name = f'{wandb_run_dataset}_{backbone_name}_{bb_train_str_short}_{ft_train_str_short}_{dec_head_key}_{loss_cfg_key}'
    else:
        run_name = f'{wandb_run_dataset}_{backbone_name}_{bb_train_str_short}_{dec_head_key}_{loss_cfg_key}'
    wnadb_config_add = dict(dec_head_name = dec_head_name,
                            backbone_name = backbone_name,
                            )
    
    bb_train_str = 'train_bb_YES' if train_backbone else 'train_bb_NO'
    if 'rein' in fine_tune or 'ladder' in  fine_tune:
        ft_train_str = 'train_ft_YES' if train_finetune else 'train_ft_NO'
        tags.append(ft_train_str)
    tags.append(bb_train_str)
    tags.append(dec_head_name)
    tags.extend(backbone_name.split('_'))
    group_name = backbone_name
    
else:    
    if model_type==ModelType.UNET:
        group_name = 'UNetModel'
        
    elif model_type==ModelType.SWINUNET:
        group_name = 'SwinUNetModel'
        
    elif model_type==ModelType.R2ATTNUNET:
        group_name = 'AttnR2UNetModel'
        
    else:
        ValueError(f'Model type: {model_type} is not found')
        
    run_name = f'{wandb_run_dataset}_{group_name}_{loss_cfg_key}'


wnadb_config = dict(segmentor_cfg_lit=segmentor_cfg_lit,
                    dataset=dataset_attrs,
                    batch_sz=batch_sz,
                    augmentations=augmentations,
                    nb_epochs=nb_epochs,
                    timestamp=time_str(),
                    torch_precision=precision,
                    train_dataloader_cfg=train_dataloader_cfg,
                    val_dataloader_cfg=val_dataloader_cfg,
                    test_datasets=test_datasets,
                    trainer_cfg=trainer_cfg,
                    nb_gpus=gpus,
                    precision=precision,
                    strategy=strategy,
                    self_training=self_training,
                    FTTA=ftta)
if ftta or self_training:
    wnadb_config['domain_adaptation_cfg'] = DA_info

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
# logger.watch(model, log="all")

n_best = 1 if save_checkpoints else 0

if cluster_paths:
    ckp_pth = Path('/usr/bmicnas02/data-biwi-01/foundation_models/AllCheckpoints')
    # ckp_pth = main_pth
else:
    ckp_pth = main_pth
    
if not ftta and not self_training:
    models_pth = ckp_pth / f'Checkpoints/{dataset}/{group_name}'
else:
    models_pth = ckp_pth / f'Checkpoints/{name_prefix}/SD_{sd_dataset}/TD_{da_dataset}/{group_name}'
    
if model_type == ModelType.SEGMENTOR:
    models_pth = models_pth / model.segmentor.decode_head.__class__.__name__
    
models_pth.mkdir(parents=True, exist_ok=True)
time_s = time_str()
if not ftta:
    checkpointers = dict(val_loss = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_loss", mode='min', filename=time_s+'-{epoch}-{val_loss:.2f}'),
                        val_dice_vol = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_dice_vol", mode='max', filename=time_s+'-{epoch}-{val_dice_vol:.2f}'),
                        )
                        #  val_mIoU_vol = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="val_mIoU_vol", mode='max', filename=time_s+'-{epoch}-{val_mIoU:.2f}'))
else:
    checkpointers = dict(last = ModelCheckpoint(dirpath=models_pth, save_top_k=n_best, monitor="epoch", mode='max', filename=time_s+'-{epoch}-Last'),)

# Create the trainer object
trainer = L.Trainer(logger=logger, callbacks=list(checkpointers.values()), **trainer_cfg)

# Print model info
summary(model)

# Train the model
# model is saved only on the main process when using distributed training
trainer.fit(model=model, datamodule=data_module)#train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer_rank = trainer.global_rank
del trainer
del data_module
del val_dataset

# Test and Validate on all the indicated datasets on a single GPU
if gpus>1:
    torch.distributed.destroy_process_group()
if trainer_rank == 0:
    # Log the relative frequencies
    log_class_rel_freqs(dataset=train_dataset, log_name_key=f'train_{dataset}')
    del train_dataset

    
    # Trainer cfg for testing
    trainer_cfg_test = dict(accelerator='gpu', devices=1, 
                            log_every_n_steps=100, num_sanity_val_steps=0,)
    trainer_testing = L.Trainer(logger=logger, **trainer_cfg_test)
    
    batch_sz = batch_sz * gpus  if strategy == 'ddp' else batch_sz
    test_model_cfg = get_lit_trainer_cfg(batch_sz=batch_sz, nb_epochs=nb_epochs, loss_cfg_key=loss_cfg_key, 
                                          dataset_attrs=dataset_attrs, gpus=1, model_type=model_type, 
                                          ftta=ftta, **kwargs)
    
    # Load the best checkpoint (highest val_dice_vol)
    model = LitTrainer.load_from_checkpoint(checkpoint_path=checkpointers[test_checkpoint_key].best_model_path, **test_model_cfg)
    
    for i, dataset_name_testing in enumerate(test_datasets):
        print(f'Eval/Test for dataset: {dataset_name_testing} {i+1}/{len(test_datasets)}')
        
        # Set the test_dataset name
        model._test_dataset_name = dataset_name_testing
        
        # Get attributes and the batch size for the given dataset
        dataset_attrs = get_data_attrs(name=dataset_name_testing, use_hdf5=None)
        
        # Dataloader config for val and test on singl GPU
        test_dataloader_cfg = dict(batch_size=batch_sz, pin_memory=pin_memory, num_workers=num_workers_dataloader,
                                persistent_workers=persistent_workers, drop_last=drop_last, 
                                shuffle=False)  
        
        # Get datasets
        _, val_dataset, test_dataset = get_datasets(data_root_pth=data_root_pth, 
                                                    data_attr=dataset_attrs, 
                                                    train_procs=augmentations+processings, 
                                                    val_test_procs=processings)
        
        val_dataloader = DataLoader(dataset=val_dataset, **test_dataloader_cfg)
        test_dataloader = DataLoader(dataset=test_dataset, **test_dataloader_cfg)
        
        # Validate on single GPU
        trainer_testing.validate(model=model, dataloaders=val_dataloader)
        
        # Test on single GPU
        trainer_testing.test(model=model, dataloaders=test_dataloader) 
        
        del val_dataloader
        del test_dataloader
        del val_dataset
        del test_dataset
        
    # Finish logging
    wandb.finish()
        
print('***END***')

