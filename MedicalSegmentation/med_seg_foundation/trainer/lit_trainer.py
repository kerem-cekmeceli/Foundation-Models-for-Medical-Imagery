from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
# from abc import abstractclassmethod
from torch.nn import CrossEntropyLoss
from utils.metrics import mIoU, DiceScore
from utils.losses import DiceLoss, FocalLoss, CompositionLoss
import lightning as L
from models.segmentor import SegmentorBase, SegmentorEncDec, SegmentorModel, implemented_segmentors
import torch
import wandb
from typing import Union, Optional, Sequence, Callable, Any
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning.pytorch.utilities import rank_zero_only

# from torch.optim.optimizer import Optimizer
# from OrigDino.dinov2.hub.utils import CenterPadding
# from lightning.pytorch.core.optimizer import LightningOptimizer

# from torchvision.transforms.functional import to_pil_image 
# from tqdm import tqdm
# import math

class LitBaseTrainer(L.LightningModule):
    def __init__(self,
                 loss_config:dict,
                 optimizer_config:dict,
                 scheduler_config:Optional[dict]=None,
                 metric_configs:Optional[Union[dict, Sequence[dict]]]=None) -> None:
        
        super().__init__()  
        self.loss_fn = LitBaseTrainer._get_loss(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        self.metrics = {}
        if metric_configs is not None:
            if isinstance(metric_configs, dict):
                metric_configs = [metric_configs]
            
            for metric_config in metric_configs:
                self.metrics[metric_config['name']] = LitBaseTrainer._get_metric(metric_config)
       
    
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
        

class LitTrainer(LitBaseTrainer):
    def __init__(self,
                 segmentor,
                 loss_config:dict,
                 optimizer_config:dict,
                 schedulers_config:Optional[dict]=None,
                 metric_configs:Optional[Union[dict, Sequence[dict]]]=None,
                 val_metrics_over_vol=True,
                 seg_log_batch_idxs=None, # 0 means no logging
                 minibatch_log_idxs=None,
                seg_val_intv=20,
                sync_dist_train=True,
                sync_dist_val=True,
                sync_dist_test=True,
                test_dataset_name='') -> None:
        super().__init__(loss_config=loss_config,
                         optimizer_config=optimizer_config,
                         scheduler_config=schedulers_config,
                         metric_configs=metric_configs)
        
        self.val_metrics_over_vol = val_metrics_over_vol
        if seg_log_batch_idxs is None:
            seg_log_batch_idxs = []
        self.seg_log_batch_idxs = seg_log_batch_idxs
        
        self.sync_dist_train = sync_dist_train
        self.sync_dist_val = sync_dist_val
        self.sync_dist_test = sync_dist_test
        
        self._test_dataset_name =test_dataset_name
        
        if isinstance(segmentor, dict):
            # config is given
            segmentor_name = segmentor['name']
            segmentor_params = segmentor['params']
            
            if segmentor_name not in implemented_segmentors:
                ValueError(f"Decode head {segmentor_name} is not supported from config.")
            
            self.segmentor = globals()[segmentor_name](**segmentor_params)
            
        else:
            assert isinstance(segmentor, SegmentorBase)
            # Model is given
            self.segmentor = segmentor
        
        
        if len(seg_log_batch_idxs) == 0:
            assert len(minibatch_log_idxs)>0
        self.minibatch_log_idxs = [] if minibatch_log_idxs is None else minibatch_log_idxs
        
        # nb epoch interval to log the seg result
        self.seg_val_intv = max(seg_val_intv, 1)

    @property
    def test_dataset_name(self):
        return self._test_dataset_name
    
    @test_dataset_name.setter
    def test_dataset_name(self, val):
        self._test_dataset_name = val
        
        
    def forward(self, x):
        return self.segmentor(x)
    
    
    def configure_optimizers(self):
        ret = dict(optimizer = self._get_optimizer())
        
        if self.scheduler_config is not None:
            ret['lr_scheduler'] = self._get_scheduler(optimizer=ret['optimizer'])
    
        return ret
    
    
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.log('epoch', self.current_epoch, on_epoch=True, on_step=False, sync_dist=self.sync_dist_train)
    
    def check_nans(tensor, batch_idx, name=''):
        assert (tensor.isnan()==False).all(), \
            f"{name} nan ratio={torch.count_nonzero(tensor.isnan()==True)/torch.numel(tensor)}, batch_idx={batch_idx}"
        
    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        
        LitTrainer.check_nans(tensor=x_batch, batch_idx=batch_idx, name='Train x_batch')
        LitTrainer.check_nans(tensor=y_batch, batch_idx=batch_idx, name='Train y_batch')
       
        # Forward pass
        y_pred = self.segmentor(x_batch)
        
        if isinstance(y_pred, list):
            assert len(self.segmentor.decode_head.decoder.loss_weights) == len(y_pred)
            loss = torch.zeros(1, device=y_batch.device)
            for yp, w in zip(y_pred, self.segmentor.decode_head.decoder.loss_weights):
                loss += w*self.loss_fn(yp, y_batch)
            y_pred_det = torch.stack(y_pred, dim=0).detach().mean(0)
            
        else:
            LitTrainer.check_nans(tensor=y_pred, batch_idx=batch_idx, name='Train y_pred')
            loss = self.loss_fn(y_pred, y_batch)
            y_pred_det = y_pred.detach()
            
        LitTrainer.check_nans(tensor=loss, batch_idx=batch_idx, name='Train loss')
                        
        # Log the loss
        self.log('loss', loss, on_epoch=True, on_step=False, sync_dist=self.sync_dist_train)
        
        # Log the metrics
        y_batch_det = y_batch.detach()
        for metric_n, metric in self.metrics.items():
            metric_dict = metric.get_res_dict(y_pred_det, y_batch_det)
            for k, v in metric_dict.items():
                self.log(metric_n+k, v, on_epoch=True, on_step=False, sync_dist=self.sync_dist_train)
                
        return loss
    
    # def on_after_backward(self) -> None:
    #     print("on_after_backward enter")
    #     for name, p in self.named_parameters():
    #         if p.requires_grad:
    #             if p.grad is None:
    #                 print(name)
    #     print("on_after_backward exit")
    
    def get_segmentations(self, x_batch, y_batch, y_pred, batch_idx):
        imgs = []
        masks_pred = []
        masks_gt = []
        caption = f'Epoch={self.current_epoch}, Batch:{batch_idx} Samples: '
        for idx in self.minibatch_log_idxs:
            # Note: We can also log a single channel (grayscale) instead of RGB since they are all the same 
            imgs.append(x_batch[idx]) 
            masks_pred.append(y_pred[idx].argmax(dim=0, keepdim=True))
            masks_gt.append(y_batch[idx].unsqueeze(0))
            caption += f'{idx}, '
        
        # Concat the seg results for the samples from the same batch
        nb_rows = 2 
        rem = len(self.minibatch_log_idxs) % nb_rows
        if rem > 0:
            imgs.extend([torch.zeros_like(imgs[0])]*rem)
            masks_pred.extend([torch.zeros_like(masks_pred[0])]*rem)
            masks_gt.extend([torch.zeros_like(masks_gt[0])]*rem)
            
        assert len(imgs) % nb_rows == 0
        
        imgs = make_grid(imgs, nrow=nb_rows, padding=0).permute([1, 2, 0]).flip(-1) # CHW -> HWC and BGR -> RGB
        masks_pred = make_grid(masks_pred, nrow=nb_rows, padding=0)[0]
        masks_gt = make_grid(masks_gt, nrow=nb_rows, padding=0)[0]
            
        log_img = wandb.Image(data_or_path=imgs.cpu().numpy(), 
                                masks={
                                    'predictions': {'mask_data': masks_pred.cpu().numpy(),
                                                    },
                                    'ground_truth': {'mask_data': masks_gt.cpu().numpy()} 
                                    },
                                caption=caption)
        return log_img
    
    @rank_zero_only                 
    def log_seg(self, batch_idx, x_batch, y_batch, y_pred, cap_prefix=''):
        batch_pos = self.seg_log_batch_idxs.index(batch_idx)
        self.logger.experiment.log({f'{cap_prefix}_seg_batch{batch_pos}': \
                [self.get_segmentations(x_batch=x_batch, y_batch=y_batch, y_pred=y_pred, batch_idx=batch_idx)],}, commit=False)
    
    def validation_step(self, val_batch, batch_idx):
        if self.val_metrics_over_vol:
            x_batch, y_batch, n_xyz = val_batch
        else:
            x_batch, y_batch = val_batch
        
        LitTrainer.check_nans(tensor=x_batch, batch_idx=batch_idx, name='Validation x_batch')
        LitTrainer.check_nans(tensor=y_batch, batch_idx=batch_idx, name='Validation y_batch')
    
        
        # Forward pass
        y_pred = self.segmentor(x_batch)
        LitTrainer.check_nans(tensor=y_pred, batch_idx=batch_idx, name='Validation y_pred')
            
        loss = self.loss_fn(y_pred, y_batch)
        LitTrainer.check_nans(tensor=loss, batch_idx=batch_idx, name='Validation loss')
        
        loss_key = 'val_loss' if self.test_dataset_name=='' else f'val_{self.test_dataset_name}_loss'
        self.log(loss_key, loss, on_epoch=True, on_step=False, sync_dist=self.sync_dist_val)
                
        # Compute the metrics 
        for metric_n, metric in self.metrics.items():
            metric_dict = metric.get_res_dict(y_pred, y_batch)
            for k, v in metric_dict.items():
                metric_key = f'val_{metric_n}{k}' if self.test_dataset_name=='' else f'val_{self.test_dataset_name}_{metric_n}{k}'
                self.log(metric_key, v, on_epoch=True, on_step=False, sync_dist=self.sync_dist_val)
                
            if  self.val_metrics_over_vol:
                metric_dicts_vol = metric.get_res_dict(y_pred, y_batch, last_slice=n_xyz['last_slice'])  
                for metric_dict_vol in metric_dicts_vol:
                    for k, v in metric_dict_vol.items():
                        metric_key = f'val_{metric_n}{k}_vol' if self.test_dataset_name=='' else f'val_{self.test_dataset_name}_{metric_n}{k}_vol'
                        self.log(metric_key, v, on_epoch=True, on_step=False, sync_dist=self.sync_dist_val)
        
        # save the segmentation result (during training only)
        if batch_idx in self.seg_log_batch_idxs and self._test_dataset_name=='':
            if (self.current_epoch+1)%self.seg_val_intv==0:
                self.log_seg(batch_idx, x_batch, y_batch, y_pred, 'val')
                
                
        # return y_pred, y_batch, n_xyz

    
    # def on_validation_epoch_end(self, outputs):
    #     if self.val_metrics_over_vol:
    #         y_pred, y_batch, n_xyz = self.all_gather(outputs)
            
    #         if self.trainer.is_global_zero:                        
    #             # Compute the metrics 
    #             for metric_n, metric in self.metrics.items():     
    #                 metric_dicts_vol = metric.get_res_dict(y_pred, y_batch, last_slice=n_xyz['last_slice'])  
    #                 for metric_dict_vol in metric_dicts_vol:
    #                     for k, v in metric_dict_vol.items():
    #                         metric_key = f'val_{metric_n}{k}_vol' if self.test_dataset_name=='' else f'val_{self.test_dataset_name}_{metric_n}{k}_vol'
    #                         self.log(metric_key, v, on_epoch=True, on_step=False, rank_zero_only=True)
        
    
    def backward(self, loss: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> None:
        return super().backward(loss, *args, **kwargs)
    
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        self.log('lr', optimizer.param_groups[0]["lr"], on_epoch=True, on_step=False, sync_dist=self.sync_dist_train)
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def lr_scheduler_step(self, scheduler: Any , metric=None) -> None:
        return super().lr_scheduler_step(scheduler, metric)
    
    def test_step(self, batch, batch_idx):   
        if self.val_metrics_over_vol:
            x_batch, y_batch, n_xyz = batch
        else:
            x_batch, y_batch = batch
            
        LitTrainer.check_nans(tensor=x_batch, batch_idx=batch_idx, name='Test x_batch')
        LitTrainer.check_nans(tensor=y_batch, batch_idx=batch_idx, name='Test y_batch')
             
        # Forward pass
        y_pred = self.segmentor(x_batch)
        LitTrainer.check_nans(tensor=y_pred, batch_idx=batch_idx, name='Test y_pred')
        
        # Hard decision
        n_classes = y_pred.shape[1]
        y_pred = F.one_hot(torch.argmax(y_pred, dim=1), n_classes).permute([0, 3, 1, 2]).to(y_pred)
        
        loss = self.loss_fn(y_pred, y_batch)
        LitTrainer.check_nans(tensor=loss, batch_idx=batch_idx, name='Test loss')
        
        # save the segmentation result
        if batch_idx in self.seg_log_batch_idxs:
            self.log_seg(batch_idx, x_batch, y_batch, y_pred, f'test_{self._test_dataset_name}')

        # Log the test loss        
        self.log(f'test_{self._test_dataset_name}_loss', loss, on_epoch=True, on_step=False, sync_dist=self.sync_dist_test)
        
        # Compute the metrics 
        for metric_n, metric in self.metrics.items():
            metric_dict = metric.get_res_dict(y_pred, y_batch)
            for k, v in metric_dict.items():
                self.log(f'test_{self._test_dataset_name}_{metric_n}{k}', v, on_epoch=True, on_step=False, sync_dist=self.sync_dist_test)
                
            if  self.val_metrics_over_vol:
                metric_dicts_vol = metric.get_res_dict(y_pred, y_batch, last_slice=n_xyz['last_slice'])  
                for metric_dict_vol in metric_dicts_vol:
                    for k, v in metric_dict_vol.items():
                        self.log(f'test_{self._test_dataset_name}_{metric_n}{k}_vol', v, on_epoch=True, on_step=False, sync_dist=self.sync_dist_test)
                    
    @rank_zero_only   
    def wnadb_conf_metrics(self):
        wandb.define_metric('loss', summary="min")
        wandb.define_metric('val_loss', summary="min")
        # wandb.define_metric('test_loss', summary="min")
        
        for metric_name in self.metrics.keys():
            assert 'dice' in metric_name or 'mIoU' in metric_name, "Unknown metric"
            wandb.define_metric(metric_name, summary="max")
            wandb.define_metric(f'val_{metric_name}', summary="max")
            # wandb.define_metric('test_'+metric_name, summary="max")
            
            if self.val_metrics_over_vol:
                wandb.define_metric(metric_name+'_vol', summary="max")
                wandb.define_metric(f'val_{metric_name}_vol', summary="max")
                # wandb.define_metric('test_'+metric_name+'_vol', summary="max")
    
    @rank_zero_only             
    def wnadb_conf_metrics_multi_dataset(self, target):
        assert target in ['val', 'test']
        
        dataset = self._test_dataset_name
        wandb.define_metric(f'{target}_{dataset}_loss', summary="min")
        
        for metric_name in self.metrics.keys():
            assert 'dice' in metric_name or 'mIoU' in metric_name, "Unknown metric"
            wandb.define_metric(f'{target}_{dataset}_{metric_name}', summary="max")
            
            if self.val_metrics_over_vol:
                wandb.define_metric(f'{target}_{dataset}_{metric_name}_vol', summary="max")
                
                
    def on_train_start(self) -> None:
        self.wnadb_conf_metrics()
        return super().on_train_start()
    
    def on_validation_start(self) -> None:
        if self._test_dataset_name != '':
            self.wnadb_conf_metrics_multi_dataset(target='val')
        return super().on_validation_start()
    
    def on_test_start(self) -> None:
        if self._test_dataset_name != '':
            self.wnadb_conf_metrics_multi_dataset(target='test')
        return super().on_test_start()
                
                