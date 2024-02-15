# from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
# from abc import abstractclassmethod
# from torch.nn import CrossEntropyLoss
# from MedDino.med_dinov2.metrics.metrics import mIoU, DiceScore, DiceLoss
# import lightning as L
# from torch.optim.optimizer import Optimizer
# from OrigDino.dinov2.hub.utils import CenterPadding
# from lightning.pytorch.core.optimizer import LightningOptimizer
# from MedDino.med_dinov2.models.segmentor import Segmentor

# from typing import Union, Optional, Sequence, Callable, Any

# class LitBaseModule(L.LightningModule):
#     def __init__(self,
#                  loss_config:dict,
#                  optimizer_config:dict,
#                  scheduler_config:Optional[dict]=None,
#                  metric_configs:Optional[Union[dict, Sequence[dict]]]=None) -> None:
        
#         super().__init__()  
#         self.loss_fn = LitBaseModule._get_loss(loss_config)
#         self.optimizer_config = optimizer_config
#         self.scheduler_config = scheduler_config
        
#         self.metrics = {}
#         if metric_configs is not None:
#             if isinstance(metric_configs, dict):
#                 metric_configs = [metric_configs]
            
#             for metric_config in metric_configs:
#                 self.metrics[metric_config['name']] = LitBaseModule._get_metric(metric_config)
       
    
#     def _get_loss(loss_config):
#         loss_name = loss_config['name']
#         loss_params = loss_config.get('params')
        
#         loss_params = {} if loss_params is None else loss_params
        
#         if loss_name == 'CrossEntropyLoss':
#             return CrossEntropyLoss(**loss_params)
        
#         elif loss_name == 'DiceLoss':
#             return DiceLoss(**loss_params)
        
#         else:
#             raise ValueError(f"Loss '{loss_name}' is not implemented.")
        
#     def _get_metric(metric_config):
#         metric_name = metric_config['name']
#         metric_params = metric_config.get('params')
        
#         metric_params = {} if metric_params is None else metric_params
        
#         if metric_name == 'mIoU':
#             return mIoU(**metric_params)
        
#         elif metric_name == 'DiceScore':
#             return DiceScore(**metric_params)
        
#         else:
#             raise ValueError(f"Metric '{metric_name}' is not implemented.")
        
    
#     def _get_optimizer(self):
#         optimizer_name = self.optimizer_config['name']
#         optimizer_params = self.optimizer_config['params']

#         if optimizer_name == 'SGD':
#             optimizer = torch.optim.SGD(self.parameters(), **optimizer_params)
#         elif optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
#         elif optimizer_name == 'AdamW':
#             optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)
#         else:
#             raise ValueError(f"Optimizer '{optimizer_name}' is not implemented.")

#         return optimizer
    
#     def _get_scheduler(self, optimizer, scheduler_config=None):
#         if scheduler_config is None:
#             scheduler_config = self.scheduler_config
            
#         assert scheduler_config is not None
            
#         scheduler_name = scheduler_config['name']
#         scheduler_params = scheduler_config['params']
        
#         if scheduler_name == 'SequentialLR':
#             schedulers = []
#             for scheduler_cfg in scheduler_params['scheduler_configs']:
#                 schedulers.append(self._get_scheduler(optimizer=optimizer,
#                                                       scheduler_config=scheduler_cfg))
                
#             scheduler_params = scheduler_params.copy().pop('scheduler_configs')
#             return SequentialLR(optimizer, schedulers, **scheduler_params)
        
#         elif scheduler_name == 'LinearLR':
#             return LinearLR(optimizer, **scheduler_params)
        
#         elif scheduler_name == 'PolynomialLR':
#             return PolynomialLR(optimizer, **scheduler_params)
        
#         else:
#             raise ValueError(f"Scheduler '{scheduler_name}' is not implemented.")

# class LitSegmentor(LitBaseModule):
#     def __init__(self,
#                  backbone, 
#                  decode_head, 
#                  optimizer_config:dict,
#                  schedulers_config:Optional[dict]=None,
#                  train_backbone=False, 
#                  reshape_dec_oup=False, 
#                  align_corners=False,
#                  val_metrics_over_vol=False,
#                  first_n_batch_to_seg_log=0) -> None:
#         super().__init__(optimizer_config=optimizer_config,
#                          scheduler_config=schedulers_config)
        
#         self.val_metrics_over_vol = val_metrics_over_vol
#         self.first_n_batch_to_seg_log = first_n_batch_to_seg_log
        
#         self.model = Segmentor(backbone=backbone,
#                                decode_head=decode_head,
#                                train_backbone=train_backbone,
#                                reshape_dec_oup=reshape_dec_oup,
#                                align_corners=align_corners)
#     def forward(self, x):
#         return self.model(x)
    
#     def configure_optimizers(self):
#         ret = dict(optimizer = self._get_optimizer())
        
#         if self.scheduler_config is not None:
#             ret['lr_scheduler'] = self._get_scheduler(optimizer=ret['optimizer'])
    
#         return ret
    
#     def training_step(self, batch, batch_idx):
#         x_batch, y_batch = batch
        
#         # Forward pass
#         y_pred = self.segmentor(x_batch)
#         loss = self.loss_fn(y_pred, y_batch)
        
#         # Log the loss
#         self.log(loss, loss.item(), on_epoch=True)
        
#         # Log the metrics
#         for metric_n, metric in self.metrics.items():
#             metric_dict = metric.get_res_dict(y_pred, y_batch)
#             for k, v in metric_dict.items():
#                 self.log(metric_n+k, v.item(), on_epoch=True)
                
#         return loss
    
#     def validation_step(self, val_batch, batch_idx):
#         x_batch, y_batch = val_batch
        
#         # Forward pass
#         y_pred = self.model(x_batch)
#         loss = self.loss_fn(y_pred, y_batch)
        
#         # Log the metrics
#         for metric_n, metric in self.metrics.items():
#             metric_dict = metric.get_res_dict(y_pred, y_batch, 
#                                               depth_idx=batch_idx if self.val_metrics_over_vol else None)  
#             for k, v in metric_dict.items():
#                 self.log('val_'+metric_n+k, v.item(), on_epoch=True)
            
#         # save the segmentation result
#         if batch_idx < self.first_n_batch_to_seg_log:
#             imgs = []
#             masks_pred = []
#             masks_gt = []
#             caption = f'Eval batch: {batch_idx+1}, samples: '
#             for idx in log_idxs:
#                 # Note: We can also log a single channel (grayscale) instead of RGB since they are all the same 
#                 imgs.append(x_batch[idx].detach().cpu().permute([1, 2, 0]).numpy()[..., ::-1])  
#                 masks_pred.append(y_pred[idx].detach().cpu().argmax(dim=0).numpy())
#                 masks_gt.append(y_batch[idx].detach().cpu().argmax(dim=0).numpy())
#                 caption += f'{idx+1}, '
                
#             # Concat the seg results for the samples from the same batch
#             imgs = np.concatenate(imgs, axis=1)
#             masks_pred = np.concatenate(masks_pred, axis=1)
#             masks_gt = np.concatenate(masks_gt, axis=1)
                
#             log_img = wandb.Image(data_or_path=imgs,
#                                 masks={
#                                     'predictions': {'mask_data': masks_pred,
#                                                     },
#                                     'ground_truth': {'mask_data': masks_gt} 
#                                     },
#                                 caption=caption)
#             # Log the seg result        
#             log_epoch[f'val_seg_batch{batch_idx+1}']=log_img
        
#         return 
    
#     def backward(self, loss: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> None:
#         return super().backward(loss, *args, **kwargs)
    
#     def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
#         self.log('lr', optimizer.param_groups[0]["lr"])
#         return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
#     def lr_scheduler_step(self, scheduler: Any , metric: Any | None) -> None:
#         return super().lr_scheduler_step(scheduler, metric)