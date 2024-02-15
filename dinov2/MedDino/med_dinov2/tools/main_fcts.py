from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
# from model import Segmentor
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
from typing import Callable, Optional, Union, Dict
from MedDino.med_dinov2.tools.checkpointer import Checkpointer
from MedDino.med_dinov2.tools.plot import show_result
import math
from MedDino.med_dinov2.metrics.metrics import SegScoreBase

def train_batches(model : nn.Module, 
                      train_loader : DataLoader, 
                      loss_fn : Callable, 
                      optimizer : torch.optim.Optimizer, 
                      metrics : Optional[Dict[str, Callable]] = None) -> dict:
    model.train()
    batches = tqdm(train_loader, desc='Train Batches', leave=False)
    tot_batches = len(batches)
    
    if tot_batches<=0:
        raise Exception('No data')
    
    metrics = {} if metrics is None else metrics
    
    # Init the epoch log dict (to be averaged over all the batches)
    log_epoch = dict(loss=0.)
    
    for i_batch, (x_batch, y_batch) in enumerate(batches):
        # Put the data on the selected device
        x_batch = x_batch.to(device=next(model.parameters()).device)
        y_batch = y_batch.to(device=next(model.parameters()).device)
        
        # Forward pass
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Save the values
        log_epoch['loss']+=loss.item()
        
        # Compute the metrics
        for metric_n, metric in metrics.items():
            metric_dict = metric.get_res_dict(y_pred, y_batch)
            for k, v in metric_dict.items():
                key_epoch = metric_n+k
                if not key_epoch in log_epoch.keys():
                    log_epoch[key_epoch] = 0.
                log_epoch[key_epoch] += v.item()
    
    # Average out the epoch logs 
    for key in log_epoch.keys():
        log_epoch[key] /= tot_batches  
               
    return log_epoch

@torch.no_grad()        
def validate_batches(model: nn.Module, 
                     val_loader: DataLoader, 
                     loss_fn: Callable, 
                     metrics: Optional[Dict[str, SegScoreBase]]=None,
                     first_n_batch_to_seg_log=0,
                     seg_log_per_batch=3,
                     metrics_over_vol=False) -> dict:
    model.eval()
    batches = tqdm(val_loader, desc='Eval Batches', leave=False)
    tot_batches = len(batches)
    
    if tot_batches<=0:
        raise Exception('No data')
    
    metrics = {} if metrics is None else metrics
    
    if len(batches)<=0:
        raise Exception('No data')
    
    log_epoch = dict(val_loss=0.)
        
    # Seg result logging preps
    if first_n_batch_to_seg_log>0:
        # get the batch indexes to log
        batch_sz = val_loader.batch_size
        sp = seg_log_per_batch+1
        # maximal separation from each other and from edges (from edges is prioritized)
        log_idxs = torch.arange(batch_sz//sp, 
                                batch_sz//sp*sp, 
                                batch_sz//sp)
        log_idxs = log_idxs + (batch_sz%sp)//2
        log_idxs = log_idxs.cpu().numpy().tolist()
    
    with torch.no_grad():
        for i_batch, (x_batch, y_batch) in enumerate(batches):
            # Put the data on the selected device
            x_batch = x_batch.to(device=next(model.parameters()).device)  # [N, C, H, W]
            y_batch = y_batch.to(device=next(model.parameters()).device)  # [N, num_cls, H, W]
            
            # Forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # save the values        
            log_epoch['val_loss']+=loss.item()
            
            # Compute the metrics
            for metric_n, metric in metrics.items():
                metric_dict = metric.get_res_dict(y_pred, y_batch, depth_idx=i_batch if metrics_over_vol else None)  
                for k, v in metric_dict.items():
                    key_epoch = 'val_'+metric_n+k
                    if not key_epoch in log_epoch.keys():
                        log_epoch[key_epoch] = 0.
                    log_epoch[key_epoch] += v.item()
                
            # save the segmentation result
            if i_batch < first_n_batch_to_seg_log:
                imgs = []
                masks_pred = []
                masks_gt = []
                caption = f'Eval batch: {i_batch+1}/{tot_batches}, samples: '
                for idx in log_idxs:
                    # Note: We can also log a single channel (grayscale) instead of RGB since they are all the same 
                    imgs.append(x_batch[idx].detach().transpose(0, -1).flip(-1))  # CHW -> HWC and BGR -> RGB
                    masks_pred.append(y_pred[idx].detach().argmax(dim=0))
                    masks_gt.append(y_batch[idx].detach().argmax(dim=0))
                    caption += f'{idx+1}/{batch_sz}, '
                    
                # Concat the seg results for the samples from the same batch
                imgs = torch.cat(imgs, axis=1)
                masks_pred = torch.cat(masks_pred, axis=1)
                masks_gt = torch.cat(masks_gt, axis=1)
                    
                log_img = wandb.Image(data_or_path=imgs.cpu().numpy(),
                                    masks={
                                        'predictions': {'mask_data': masks_pred.cpu().numpy(),
                                                        },
                                        'ground_truth': {'mask_data': masks_gt.cpu().numpy()} 
                                        },
                                    caption=caption)
                # Log the seg result        
                log_epoch[f'val_seg_batch{i_batch+1}']=log_img
        
        # Average out the epoch logs 
        for key in log_epoch.keys():
            if not key.startswith('val_seg_batch'):
                if key == 'val_loss' or not metrics_over_vol:
                    log_epoch[key] /= tot_batches
                else:
                    log_epoch[key] /= (tot_batches / (next(iter(metrics.items()))[1].vol_batch_sz / x_batch.size(0)))
       
    return log_epoch


def train(model: nn.Module, 
          train_loader: DataLoader, 
          loss_fn: Callable, 
          optimizer: torch.optim.Optimizer,
          n_epochs: int, 
          logger: wandb.wandb_sdk.wandb_run.Run,
          scheduler: Optional[torch.optim.lr_scheduler.SequentialLR]=None, 
          val_loader: Optional[DataLoader]=None, 
          print_epoch_info: bool=True,
          checkpointer: Optional[Checkpointer]=None, 
          metrics: Optional[Dict[str, SegScoreBase]]=None,
          seg_val_intv=20,
          first_n_batch_to_seg_log=16,
          seg_log_per_batch=3,
          val_metrics_over_vol=False,
          print_per_class=False) -> None:
    
    epochs = tqdm(range(n_epochs), desc='Epochs')
    
    if metrics is not None:
        assert isinstance(metrics, dict)
    
    for epoch in epochs:
        # Init the epoch log dict
        log_epoch = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"])
                
        # Train
        log_train = train_batches(model, train_loader, loss_fn, optimizer, metrics)
        log_epoch = log_epoch | log_train
        
        # Validate    
        if val_loader is not None:
            batch_th = first_n_batch_to_seg_log if (epoch+1)%seg_val_intv==0 else 0
            log_val = validate_batches(model, val_loader, loss_fn, metrics,
                                       batch_th, seg_log_per_batch, val_metrics_over_vol)
            log_epoch = log_epoch | log_val
            
        # Update LR
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch info
        if print_epoch_info:
            epoch_str = ''
            for key, val in log_epoch.items():
                if not key.startswith('val_seg_batch'):
                    if not 'class' in key or print_per_class:
                        if epoch_str != '':
                            epoch_str += ', '
                        epoch_str += f'{key} = {round(val, 5)}'
            tqdm.write(epoch_str)
            
        # Checkpointer
        if checkpointer is not None:
            checkpointer.update(model=model, metrics=log_epoch, epoch=epoch, opt=optimizer, scheduler=scheduler)
            
        # Log the epoch
        logger.log(log_epoch)
        
    # Save the best models:
    if checkpointer is not None:
        checkpointer.save()
        
        
@torch.no_grad()           
def test_batches(model: nn.Module, 
                val_loader: DataLoader, 
                loss_fn: Callable, 
                metrics: Optional[Dict[str, SegScoreBase]]=None, 
                soft: bool=False,
                first_n_batch_to_seg_log=16,
                seg_log_per_batch=3,
                metrics_over_vol=False) -> dict:
    model.eval()
    batches = tqdm(val_loader, desc='Test Batches', leave=False)
    tot_batches = len(batches)
    
    if tot_batches<=0:
        raise Exception('No data')
    
    metrics = {} if metrics is None else metrics
    
    if len(batches)<=0:
        raise Exception('No data')
    
    log_test = dict(test_loss=0.)
    
    # Seg result logging preps
    if first_n_batch_to_seg_log>0:
        # get the batch indexes to log
        batch_sz = val_loader.batch_size
        sp = seg_log_per_batch+1
        # maximal separation from each other and from edges (from edges is prioritized)
        log_idxs = torch.arange(batch_sz//sp, 
                                batch_sz//sp*sp, 
                                batch_sz//sp)
        log_idxs = log_idxs + (batch_sz%sp)//2
        log_idxs = log_idxs.cpu().numpy().tolist()
        
        column_names = [f'sample {i+1}/{batch_sz}' for i in log_idxs]
        column_names = column_names + ['test_loss'] + [f'test_{key}' for key in metrics.keys()]
        # Create the wandb table to log the selected seg results
        log_table = wandb.Table(columns=column_names)  
    else:
        log_table = None
    
    with torch.no_grad():
        for i_batch, (x_batch, y_batch) in enumerate(batches):
            # Put the data on the selected device
            x_batch = x_batch.to(device=next(model.parameters()).device)
            y_batch = y_batch.to(device=next(model.parameters()).device)
            
            # check if you have nans x batch or y batch
            
            # Forward pass
            y_pred = model(x_batch)
            
            # Hard decision
            if not soft:
                n_classes = y_pred.shape[1]
                dtype = y_pred.dtype
                y_pred = F.one_hot(torch.argmax(y_pred, dim=1), n_classes).permute([0, 3, 1, 2]).to(dtype)
            
            loss = loss_fn(y_pred, y_batch)
            
            # Log the segmentation result
            if i_batch < first_n_batch_to_seg_log:
                log_row = []
                for idx in log_idxs:
                    log_img = wandb.Image(data_or_path=x_batch[idx].detach().transpose(0, -1).flip(-1).cpu().numpy(),
                                    masks={
                                        'predictions': {'mask_data': y_pred[idx].detach().argmax(dim=0).cpu().numpy(),
                                                        },
                                        'ground_truth': {'mask_data': y_batch[idx].detach().argmax(dim=0).cpu().numpy()} 
                                        })
                    log_row.append(log_img)
            
            # Save the values        
            log_test['test_loss']+=loss.item()
            
            # Append the row for the table
            if i_batch < first_n_batch_to_seg_log:
                log_row.append(loss.item())
            
            # Compute the metrics
            for metric_n, metric in metrics.items():
                metric_dict = metric.get_res_dict(y_pred, y_batch, depth_idx=i_batch if metrics_over_vol else None)
                for k, v in metric_dict.items():
                    key_test = 'test_'+metric_n+k
                    if not key_test in log_test.keys():
                        log_test[key_test] = 0.
                    log_test[key_test] += v.item()
                
                # Append the row for the table
                if i_batch < first_n_batch_to_seg_log:
                    log_row.append(metric.get_res_dict(y_pred, y_batch)[''])
                
            if i_batch < first_n_batch_to_seg_log:
                log_table.add_data(*log_row)
        
        # Average out the epoch logs 
        for key in log_test.keys():
            if key == 'test_loss' or not metrics_over_vol:
                log_test[key] /= tot_batches
            else:
                log_test[key] /= (tot_batches / (next(iter(metrics.items()))[1].vol_batch_sz / x_batch.size(0)))
            
        # Log the table
        if log_table is not None:
            log_test['test_seg_table'] = log_table
         
    return log_test

    
def test(model: nn.Module,
         test_loader: DataLoader, 
         loss_fn: Callable, 
         logger: wandb.wandb_sdk.wandb_run.Run,
         metrics: Optional[Dict[str, Callable]]=None, 
         soft:bool = False,
         first_n_batch_to_seg_log=16,
         seg_log_per_batch=3,
         metrics_over_vol=False):
    log_test = test_batches(model, test_loader, loss_fn, metrics, soft, 
                            first_n_batch_to_seg_log, seg_log_per_batch,
                            metrics_over_vol)
    
    test_str = ''
    for key, val in log_test.items():
        if not key.startswith('test_seg_table'):
            logger.summary[key] = val
            test_str += f'{key}={round(val, 5)}, '
        else:
            # Log the table
            logger.log({key:val})
    print(test_str)
    

    

###########################################################################

# def train_(model, train_loader, criterion, optimizer, epoch, device):
#     model.train()
#     loop = tqdm(train_loader, total=len(train_loader))
#     running_loss = 0
#     correct = 0

#     for batch_idx, (data, target) in enumerate(loop):
#         # print(batch_idx) 
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(output.data, 1)
#         loop.set_description(f"Epoch {epoch+1}")
#         loop.set_postfix(loss = loss.item())

#     print(f'\nTrain set: Average loss: {running_loss/len(train_loader):.4f}')

# def validation(model, criterion, valid_loader):
#     model.eval()
#     running_loss = 0
#     correct = 0

#     with torch.no_grad():
#         loop = tqdm(valid_loader, total=len(valid_loader))
#         for data, target in loop:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             running_loss += loss.item()
#             _, predicted = torch.max(output.data, 1)

#     print(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.4f}')


# def infer(image_path, model, device, img_transform):
#     # Load and transform the image
#     image = Image.open(image_path).convert("RGB")
#     transformed_image = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

#     # Make sure the model is in evaluation mode
#     model.eval()

#     with torch.no_grad():
#         # Make prediction
#         output = model(transformed_image)

#         # Get the predicted class for each pixel
#         _, predicted = torch.max(output, 1)
    
#     # Move prediction to cpu and convert to numpy array
#     predicted = predicted.squeeze().cpu().numpy()

#     return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), predicted