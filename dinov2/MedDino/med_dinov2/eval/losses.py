from .metrics import mIoU, DiceScore, ScoreBase
from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch
from abc import ABC, abstractmethod
from torch.nn import CrossEntropyLoss

class mIoULoss(mIoU):
    def __init__(self, 
                 prob_inputs=False, 
                 bg_ch_to_rm=None,
                 reduction='mean',
                 epsilon=1e-6):
        assert reduction in ['mean , sum']
        super().__init__(prob_inputs=prob_inputs, 
                         soft=True,  # loss => must be differentiable => soft
                         bg_ch_to_rm=bg_ch_to_rm,
                         reduction=reduction,
                         ret_per_class_scores=False,
                         vol_batch_sz=None,
                         epsilon=epsilon)
    
    def forward(self, inputs, target_oneHot):
        return 1 - super().forward(inputs, target_oneHot)
    
    
    
class DiceLoss(DiceScore):
    def __init__(self, 
                 prob_inputs=False, 
                 bg_ch_to_rm=None,
                 reduction='mean',
                 k=1, 
                 epsilon=1e-6,):
        assert reduction in ['mean', 'sum']
        super().__init__(prob_inputs=prob_inputs, 
                         soft=True,  # loss => must be differentiable => soft
                         bg_ch_to_rm=bg_ch_to_rm,
                         reduction=reduction,
                         ret_per_class_scores=False,
                         vol_batch_sz=None,
                         k=k,
                         epsilon=epsilon)
    
    def forward(self, inputs, target_oneHot):
        return 1 - super().forward(inputs, target_oneHot)
    
    

class FocalLoss(nn.Module):
    def __init__(self, 
                 bg_ch_to_rm=None,
                 reduction='mean',
                 gamma=2,
                 alpha=None):
        assert reduction in ['mean', 'sum']
        
        super().__init__()
        if bg_ch_to_rm is None:
            bg_ch_to_rm = -100
        self.bg_ch_to_rm = bg_ch_to_rm
        assert reduction in ['sum', 'mean']
        self.reduction = reduction
        
        assert gamma > 0
        self.gamma = gamma
        
        if isinstance(alpha, list):
            alpha = torch.Tensor(alpha)
        self.alpha = alpha        

    def forward(self, mask_pred, mask_gt):
        """mask_pred is log probas with BG removed if necessary"""
        mask_pred = F.log_softmax(mask_pred, dim=1)
        
        ce_loss_weighted = F.nll_loss(input=mask_pred, 
                                      target=mask_gt.argmax(dim=1), 
                                      weight=self.alpha, reduction='none',
                                      ignore_index=self.bg_ch_to_rm) # [N, H, W]
        
        pt = torch.exp(-1*F.nll_loss(input=mask_pred, target=mask_gt.argmax(dim=1), 
                                     reduction='none', ignore_index=self.bg_ch_to_rm))  # [N, H, W]
        
        focal_loss = (1-pt) ** self.gamma * ce_loss_weighted
        
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:
            ValueError(f'Not implemented error for reduction type: {self.reduction}')\
                
                
                
                
class CompositionLoss(torch.nn.Module):
    def __init__(self, loss1, loss2, comp_rat=0.5) -> None:
        super().__init__()
        
        self.loss1 = CompositionLoss.get_loss(loss1)
        self.loss2 = CompositionLoss.get_loss(loss2)
        
        assert comp_rat>=0 and comp_rat<=1
        self.comp_rat = comp_rat
        
    def get_loss(cfg):
        loss_name = cfg['name']
        loss_params = cfg['params']
        
        if loss_name == 'CE':
            return CrossEntropyLoss(**loss_params)
        elif loss_name == 'Dice':
            return DiceLoss(**loss_params)
        elif loss_name == 'Focal':
            return FocalLoss(**loss_params)
        else:
            ValueError(f'Undefined loss name {loss_name}')
        
    def forward(self, x, x_pred):   
        return self.comp_rat * self.loss1(x, x_pred) + (1-self.comp_rat) * self.loss2(x, x_pred)
    
    