from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from abc import abstractclassmethod
from torch.nn import CrossEntropyLoss
from MedDino.med_dinov2.eval.metrics import mIoU, DiceScore
from MedDino.med_dinov2.eval.losses import DiceLoss, FocalLoss, CompositionLoss
import lightning as L
from torch.optim.optimizer import Optimizer
from OrigDino.dinov2.hub.utils import CenterPadding
from lightning.pytorch.core.optimizer import LightningOptimizer
from MedDino.med_dinov2.models.segmentor import Segmentor
import torch
import wandb
from typing import Union, Optional, Sequence, Callable, Any
from torchvision.transforms.functional import to_pil_image 
# from tqdm import tqdm
import torch.nn.functional as F
import math
from torchvision.utils import make_grid

class LitBaseModule(L.LightningModule):
    def __init__(self,
                 loss_config:dict,
                 optimizer_config:dict,
                 scheduler_config:Optional[dict]=None,
                 metric_configs:Optional[Union[dict, Sequence[dict]]]=None) -> None:
        
        super().__init__()  
        self.loss_fn = LitBaseModule._get_loss(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        self.metrics = {}
        if metric_configs is not None:
            if isinstance(metric_configs, dict):
                metric_configs = [metric_configs]
            
            for metric_config in metric_configs:
                self.metrics[metric_config['name']] = LitBaseModule._get_metric(metric_config)
       
    
    def _get_loss(loss_config):
        loss_name = loss_config['name']
        loss_params = loss_config.get('params')
        
        loss_params = {} if loss_params is None else loss_params
        
        if loss_name == 'CrossEntropyLoss':
            return CrossEntropyLoss(**loss_params)
        
        elif loss_name == 'DiceLoss':
            return DiceLoss(**loss_params)
        
        elif loss_name == 'FocalLoss':
            return FocalLoss(**loss_params)
        
        elif loss_name == 'CompositionLoss':
            return CompositionLoss(**loss_params)
        
        else:
            raise ValueError(f"Loss '{loss_name}' is not implemented.")
        
    def _get_metric(metric_config):
        metric_name = metric_config['name']
        metric_params = metric_config.get('params')
        
        metric_params = {} if metric_params is None else metric_params
        
        if metric_name == 'mIoU':
            return mIoU(**metric_params)
        
        elif metric_name == 'dice':
            return DiceScore(**metric_params)
        
        else:
            raise ValueError(f"Metric '{metric_name}' is not implemented.")
        
    
    def _get_optimizer(self):
        optimizer_name = self.optimizer_config['name']
        optimizer_params = self.optimizer_config['params']

        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **optimizer_params)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' is not implemented.")

        return optimizer
    
    def _get_scheduler(self, optimizer, scheduler_config=None):
        if scheduler_config is None:
            scheduler_config = self.scheduler_config
            
        assert scheduler_config is not None
            
        scheduler_name = scheduler_config['name']
        scheduler_params = scheduler_config['params']
        
        if scheduler_name == 'SequentialLR':
            schedulers = []
            for scheduler_cfg in scheduler_params['scheduler_configs']:
                schedulers.append(self._get_scheduler(optimizer=optimizer,
                                                      scheduler_config=scheduler_cfg))
                
            scheduler_params = scheduler_params.copy()
            del scheduler_params['scheduler_configs']
            return SequentialLR(optimizer, schedulers, **scheduler_params)
        
        elif scheduler_name == 'LinearLR':
            return LinearLR(optimizer, **scheduler_params)
        
        elif scheduler_name == 'PolynomialLR':
            return PolynomialLR(optimizer, **scheduler_params)
        
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' is not implemented.")
        

class LitSegmentor(LitBaseModule):
    def __init__(self,
                 backbone, 
                 decode_head, 
                 loss_config:dict,
                 optimizer_config:dict,
                 schedulers_config:Optional[dict]=None,
                 metric_configs:Optional[Union[dict, Sequence[dict]]]=None,
                 train_backbone=False, 
                 reshape_dec_oup=False, 
                 align_corners=False,
                 val_metrics_over_vol=True,
                 first_n_batch_to_seg_log=0, # 0 means no logging
                 minibatch_log_idxs=None,
                seg_val_intv=20) -> None:
        super().__init__(loss_config=loss_config,
                         optimizer_config=optimizer_config,
                         scheduler_config=schedulers_config,
                         metric_configs=metric_configs)
        
        self.val_metrics_over_vol = val_metrics_over_vol
        assert first_n_batch_to_seg_log>=0
        self.first_n_batch_to_seg_log = first_n_batch_to_seg_log
        
        self.model = Segmentor(backbone=backbone,
                               decode_head=decode_head,
                               train_backbone=train_backbone,
                               reshape_dec_oup=reshape_dec_oup,
                               align_corners=align_corners)
        
        if first_n_batch_to_seg_log > 0:
            assert len(minibatch_log_idxs)>0
        self.minibatch_log_idxs = [] if minibatch_log_idxs is None else minibatch_log_idxs
        
        # nb epoch interval to log the seg result
        self.seg_val_intv = max(seg_val_intv, 1)
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def configure_optimizers(self):
        ret = dict(optimizer = self._get_optimizer())
        
        if self.scheduler_config is not None:
            ret['lr_scheduler'] = self._get_scheduler(optimizer=ret['optimizer'])
    
        return ret
    
    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        
        assert (x_batch.isnan()==False).all(), f"x_batch nan ratio={torch.count_nonzero(x_batch.isnan()==True)/torch.numel(x_batch)}"
        assert (y_batch.isnan()==False).all(), f"y_batch nan ratio={torch.count_nonzero(y_batch.isnan()==True)/torch.numel(y_batch)}"
        
        # Forward pass
        y_pred = self.model(x_batch)
        assert (y_pred.isnan()==False).all(), f"y_pred nan ratio={torch.count_nonzero(y_pred.isnan()==True)/torch.numel(y_pred)}"
        loss = self.loss_fn(y_pred, y_batch)
        assert (loss.isnan()==False).all(), f"loss nan ratio={torch.count_nonzero(loss.isnan()==True)/torch.numel(loss)}"
                        
        # Log the loss
        self.log('loss', loss, on_epoch=True, on_step=False)
        
        # Log the metrics
        y_pred_det = y_pred.detach()
        y_batch_det = y_batch.detach()
        for metric_n, metric in self.metrics.items():
            metric_dict = metric.get_res_dict(y_pred_det, y_batch_det)
            for k, v in metric_dict.items():
                self.log(metric_n+k, v, on_epoch=True, on_step=False)
                
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        sync_dist=False
        x_batch, y_batch = val_batch
        
        assert (x_batch.isnan()==False).all(), f"x_batch nan ratio={torch.count_nonzero(x_batch.isnan()==True)/torch.numel(x_batch)}"
        assert (y_batch.isnan()==False).all(), f"y_batch nan ratio={torch.count_nonzero(y_batch.isnan()==True)/torch.numel(y_batch)}"
        
        # Forward pass
        y_pred = self.model(x_batch)
        assert (y_pred.isnan()==False).all(), f"y_pred nan ratio={torch.count_nonzero(y_pred.isnan()==True)/torch.numel(y_pred)}"
        loss = self.loss_fn(y_pred, y_batch)
        assert (loss.isnan()==False).all(), f"loss nan ratio={torch.count_nonzero(loss.isnan()==True)/torch.numel(loss)}"
        
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=sync_dist)
                
        # Compute the metrics 
        for metric_n, metric in self.metrics.items():
            metric_dict = metric.get_res_dict(y_pred, y_batch)
            for k, v in metric_dict.items():
                self.log('val_'+metric_n+k, v, on_epoch=True, on_step=False, sync_dist=sync_dist)
                
            if  self.val_metrics_over_vol:
                metric_dict_vol = metric.get_res_dict(y_pred, y_batch, depth_idx=batch_idx )  
                for k, v in metric_dict_vol.items():
                    self.log('val_'+metric_n+k+'_vol', v, on_epoch=True, on_step=False, sync_dist=sync_dist)
        
        # save the segmentation result
        if batch_idx < self.first_n_batch_to_seg_log:
            if (self.current_epoch+1)%self.seg_val_intv==0:
                imgs = []
                masks_pred = []
                masks_gt = []
                caption = f'Epoch={self.current_epoch}, Samples: '
                for idx in self.minibatch_log_idxs:
                    # Note: We can also log a single channel (grayscale) instead of RGB since they are all the same 
                    imgs.append(x_batch[idx]) 
                    masks_pred.append(y_pred[idx].argmax(dim=0, keepdim=True))
                    masks_gt.append(y_batch[idx].argmax(dim=0, keepdim=True))
                    caption += f'{idx+1}, '
                
                # Concat the seg results for the samples from the same batch
                nb_rows = 2 
                rem = len(self.minibatch_log_idxs) % nb_rows
                if rem > 0:
                    imgs.extend([torch.zeros_like(imgs[0])]*rem)
                    masks_pred.extend([torch.zeros_like(masks_pred[0])]*rem)
                    masks_gt.extend([torch.zeros_like(masks_gt[0])]*rem)
                 
                assert len(imgs) % nb_rows == 0
                
                imgs = make_grid(imgs, nrow=nb_rows, padding=0).permute([1, 2, 0]).flip(-1)
                masks_pred = make_grid(masks_pred, nrow=nb_rows, padding=0)[0]
                masks_gt = make_grid(masks_gt, nrow=nb_rows, padding=0)[0]
                    
                log_img = wandb.Image(data_or_path=imgs.cpu().numpy(), # CHW -> HWC and BGR -> RGB
                                      masks={
                                          'predictions': {'mask_data': masks_pred.cpu().numpy(),
                                                          },
                                          'ground_truth': {'mask_data': masks_gt.cpu().numpy()} 
                                          },
                                      caption=caption)
                self.logger.experiment.log({f'val_seg_batch{batch_idx+1}': [log_img],}, commit=False)  
    
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.log('epoch', self.current_epoch, on_epoch=True, on_step=False)
        
    # def on_train_epoch_end(self) -> None:
    #     return super().on_train_epoch_end()
        
    
    def backward(self, loss: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> None:
        return super().backward(loss, *args, **kwargs)
    
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        self.log('lr', optimizer.param_groups[0]["lr"], on_epoch=True, on_step=False)
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def lr_scheduler_step(self, scheduler: Any , metric) -> None:
        return super().lr_scheduler_step(scheduler, metric)
    
    def test_step(self, batch, batch_idx):   
        sync_dist=False
        x_batch, y_batch = batch
             
        # Forward pass
        y_pred = self.model(x_batch)
        
        # Hard decision
        n_classes = y_pred.shape[1]
        y_pred = F.one_hot(torch.argmax(y_pred, dim=1), n_classes).permute([0, 3, 1, 2]).to(y_pred)
        
        loss = self.loss_fn(y_pred, y_batch)
        
        # save the segmentation result
        if batch_idx < self.first_n_batch_to_seg_log:
            if (self.current_epoch+1)%self.seg_val_intv==0:
                imgs = []
                masks_pred = []
                masks_gt = []
                caption = f'Epoch={self.current_epoch}, Samples: '
                for idx in self.minibatch_log_idxs:
                    # Note: We can also log a single channel (grayscale) instead of RGB since they are all the same 
                    imgs.append(x_batch[idx]) 
                    masks_pred.append(y_pred[idx].argmax(dim=0, keepdim=True))
                    masks_gt.append(y_batch[idx].argmax(dim=0, keepdim=True))
                    caption += f'{idx+1}, '
                
                # Concat the seg results for the samples from the same batch
                nb_rows = 2 
                rem = len(self.minibatch_log_idxs) % nb_rows
                if rem > 0:
                    imgs.extend([torch.zeros_like(imgs[0])]*rem)
                    masks_pred.extend([torch.zeros_like(masks_pred[0])]*rem)
                    masks_gt.extend([torch.zeros_like(masks_gt[0])]*rem)
                 
                assert len(imgs) % nb_rows == 0
                
                imgs = make_grid(imgs, nrow=nb_rows, padding=0).permute([1, 2, 0]).flip(-1)
                masks_pred = make_grid(masks_pred, nrow=nb_rows, padding=0)[0]
                masks_gt = make_grid(masks_gt, nrow=nb_rows, padding=0)[0]
                    
                log_img = wandb.Image(data_or_path=imgs.cpu().numpy(), # CHW -> HWC and BGR -> RGB
                                      masks={
                                          'predictions': {'mask_data': masks_pred.cpu().numpy(),
                                                          },
                                          'ground_truth': {'mask_data': masks_gt.cpu().numpy()} 
                                          },
                                      caption=caption)
                self.logger.experiment.log({f'test_seg_batch{batch_idx+1}': [log_img],}, commit=False)
    
        # Log the test loss        
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=sync_dist)
        
        # Compute the metrics 
        for metric_n, metric in self.metrics.items():
            metric_dict = metric.get_res_dict(y_pred, y_batch)
            for k, v in metric_dict.items():
                self.log('test_'+metric_n+k, v, on_epoch=True, on_step=False, sync_dist=sync_dist)
                
            if  self.val_metrics_over_vol:
                metric_dict_vol = metric.get_res_dict(y_pred, y_batch, depth_idx=batch_idx )  
                for k, v in metric_dict_vol.items():
                    self.log('test_'+metric_n+k+'_vol', v, on_epoch=True, on_step=False, sync_dist=sync_dist)
        
    def on_fit_start(self) -> None:
        wandb.define_metric('loss', summary="min")
        wandb.define_metric('val_loss', summary="min")
        wandb.define_metric('test_loss', summary="min")
        
        for metric_name in self.metrics.keys():
            assert 'dice' in metric_name or 'mIoU' in metric_name, "Unknown metric"
            wandb.define_metric(metric_name, summary="max")
            wandb.define_metric('val_'+metric_name, summary="max")
            wandb.define_metric('test_'+metric_name, summary="max")
            
            if self.val_metrics_over_vol:
                wandb.define_metric(metric_name+'_vol', summary="max")
                wandb.define_metric('val_'+metric_name+'_vol', summary="max")
                wandb.define_metric('test_'+metric_name+'_vol', summary="max")
                
        return super().on_fit_start()
                
                