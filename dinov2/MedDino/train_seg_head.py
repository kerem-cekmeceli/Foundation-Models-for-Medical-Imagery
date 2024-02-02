
import sys
from pathlib import Path
 
dino_main_pth = Path(__file__).parent.parent
orig_dino_pth = dino_main_pth / 'OrigDino'
sys.path.insert(1, dino_main_pth.as_posix())
sys.path.insert(2, orig_dino_pth.as_posix())

import torch

import cv2
import numpy as np
from matplotlib import pyplot as plt

from prep_model import get_bb_name, get_dino_backbone, time_str
from OrigDino.dinov2.eval.segmentation import models

from MedDino.med_dinov2.models.segmentor import Segmentor
from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear
from MedDino.med_dinov2.data.datasets import SegmentationDataset
from torch.utils.data import DataLoader
from MedDino.med_dinov2.tools.main_fcts import train, test
from MedDino.med_dinov2.metrics.metrics import mIoU, DiceScore

from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torchinfo import summary
from torch.nn import CrossEntropyLoss
import wandb
from MedDino.med_dinov2.tools.checkpointer import Checkpointer


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

dec_head = ConvHeadLinear(embedding_sz=backbone.embed_dim, 
                     num_classses=num_classses,
                     n_concat=n_concat,
                     interp_fact=backbone.patch_size)
dec_head.to(device)

print("Convolutional decode head")
summary(dec_head)

# Initialize the segmentor
train_backbone=False
model = Segmentor(backbone=backbone,
                  decode_head=dec_head,
                  train_backbone=train_backbone)
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
nb_epochs = 3
warmup_iters = 1
lr_cfg = dict(linear_lr = dict(start_factor=1/3, end_factor=1.0, total_iters=warmup_iters),
              polynomial_lr = dict(power=1.0))
scheduler1 = LinearLR(optm, **lr_cfg['linear_lr'])
scheduler2 = PolynomialLR(optm, **lr_cfg['polynomial_lr'])
scheduler = SequentialLR(optm, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
#@TODO: check tuning strategies for LR

# Loss function
loss = CrossEntropyLoss()

# Metrics
metrics=dict(mIoU=mIoU(n_classes=num_classses),
             dice=DiceScore(bg_channel=0, soft=True, 
                            reduction='mean', k=1, epsilon=1e-6,
                            fg_only=True))

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
log_mode = 'offline' if log_the_run else 'disabled'
logger = wandb.init(project='FoundationModels_MedDino',
                    group=wandb_group_name,
                    config=wnadb_config,
                    dir=wandb_log_path,
                    name=time_str(),
                    mode=log_mode)

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
# train(model=model, train_loader=train_dataloader, 
#       val_loader=val_dataloader, loss_fn=loss, 
#       optimizer=optm, scheduler=scheduler,
#       n_epochs=nb_epochs, device=device, logger=logger,
#       checkpointer=cp, metrics=metrics)

# Prints the test set mIoU loss
#@TODO return hard decision predicted seg classes per pix
test(model=model, test_loader=test_dataloader, loss_fn=loss, 
     device=device, logger=logger, metrics=metrics, soft=False)

#@TODO plot gt and prediction side by side

# log input pred gt randomly every n iter (dice scores per class)

#@TODO add types and comments
#@TODO write readme.md
#@TODO maybe torchlighning Friday

#finish logging
wandb.finish()
print('***END***')
