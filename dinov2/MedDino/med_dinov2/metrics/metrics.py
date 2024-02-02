from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch


class mIoU(nn.Module):
    def __init__(self, n_classes):
        super(mIoU, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, targets_one_hot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]

        # Numerator Product
        inter = inputs * targets_one_hot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).nansum(2)

        #Denominator 
        union= inputs + targets_one_hot - (inputs*targets_one_hot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).nansum(2)

        iou = inter/union  # N x C

        ## Return average loss over classes and batch
        return iou.mean()  
    
class mIoULoss(mIoU):
    def __init__(self, n_classes):
        super().__init__(n_classes)
    
    def forward(self, inputs, target_oneHot):
        return 1 - super().forward(inputs, target_oneHot)
    
    

class DiceScore(nn.Module):
    def __init__(self, bg_channel=None, soft=True, 
                 reduction='mean', k=1, epsilon=1e-6,
                 fg_only=True) -> None:
        super().__init__()
        
        assert reduction in ['none', 'mean', 'sum'], f'Unrecognised reduction: {reduction}'
        
        self.fg_only=fg_only
        self.epsilon=epsilon
        self.bg_channel=bg_channel
        self.soft = soft
        self.reduction = reduction
        self.k = k
        
    def forward(self, mask_pred, mask_gt):
        if not self.soft:
            n_classes = mask_pred.shape[1]
            dtype = mask_pred.dtype
            mask_pred = F.one_hot(torch.argmax(mask_pred, dim=1), n_classes).permute([0, 3, 1, 2]).to(dtype)
            
        N, C = mask_pred.shape[0:2]
        mask_pred = mask_pred.view(N, C, -1)
        mask_gt = mask_gt.view(N, C, -1)

        assert mask_pred.shape == mask_gt.shape

        inter = torch.sum(mask_gt * mask_pred, dim=-1)
        pred = torch.sum(mask_pred ** self.k, dim=-1)
        gt = torch.sum(mask_gt ** self.k, dim=-1)
        dices = (2 * inter + self.epsilon) / (pred + gt + self.epsilon)


        fg_mask = (torch.arange(mask_pred.shape[1]) != self.bg_channel)
        if self.reduction == 'none':
            if self.fg_only:
                return dices[:, fg_mask, ...]
            return dices, dices[:, fg_mask, ...]
        elif self.reduction == 'mean':
            if self.fg_only:
                return dices[:, fg_mask, ...].nanmean()
            return dices.nanmean(), dices[:, fg_mask, ...].nanmean()
        elif self.reduction == 'sum':
            if self.fg_only:
                return dices[:, fg_mask, ...].nansum()
            return dices.nansum(), dices[:, fg_mask, ...].nansum()
        
    
class DiceLoss(DiceScore):
    def __init__(self, bg_channel=None, soft=True, 
                 reduction='mean', k=1, epsilon=0, fg_only=True) -> None:
        super().__init__(bg_channel, soft, reduction, k, epsilon, fg_only)
        
    def forward(self, mask_pred, mask_gt):
        
        if self.fg_only:
            dice = super().forward(mask_pred, mask_gt)
        else:
            dice, _ = super().forward(mask_pred, mask_gt)
            
        loss = 1 - dice
        return loss


# def dice_score(mask_pred, mask_gt, soft=True, reduction='mean', bg_channel=0, k=1, epsilon=0):

#     if not soft:
#         n_classes = mask_pred.shape[1]
#         mask_pred = F.one_hot(torch.argmax(mask_pred, dim=1), n_classes)

#     N, C = mask_pred.shape[0:2]
#     mask_pred = mask_pred.view(N, C, -1)
#     mask_gt = mask_gt.view(N, C, -1)

#     assert mask_pred.shape == mask_gt.shape

#     inter = torch.sum(mask_gt * mask_pred, dim=-1)
#     pred = torch.sum(mask_pred ** k, dim=-1)
#     gt = torch.sum(mask_gt ** k, dim=-1)
#     dices = (2 * inter + epsilon) / (pred + gt + epsilon)

#     assert reduction in ['none', 'mean', 'sum'], f'Unrecognised reduction: {reduction}'

#     fg_mask = (torch.arange(mask_pred.shape[1]) != bg_channel)
#     if reduction == 'none':
#         return dices, dices[:, fg_mask, ...]
#     elif reduction == 'mean':
#         return dices.nanmean(), dices[:, fg_mask, ...].nanmean()
#     elif reduction == 'sum':
#         return dices.nansum(), dices[:, fg_mask, ...].nansum()

    

# @TODO eval on 3d not per slice
