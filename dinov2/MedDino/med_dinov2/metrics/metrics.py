from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch
from abc import ABC, abstractmethod


class SegScoreBase(nn.Module, ABC):
    def __init__(self, 
                 n_class, 
                 prob_inputs=False, 
                 soft=True, 
                 bg_ch_to_rm=None,
                 reduction='mean',
                 ret_per_class_scores=True,
                 vol_batch_sz=None):
        super(SegScoreBase, self).__init__()
        
        assert n_class>0, f'number of classes should be a positive integer, got {n_class}'
        self.n_class = n_class
        self.prob_inputs=prob_inputs
        self.soft=soft
        
        assert reduction in ['none', 'mean', 'sum'], f'Undefined reduction: {reduction}'
        self.reduction=reduction
        
        if bg_ch_to_rm is not None:
            assert bg_ch_to_rm>=0 and bg_ch_to_rm<n_class, f'BG channel to remove:{bg_ch_to_rm} not in range [0, {n_class-1}]'
        self.bg_ch_to_rm=bg_ch_to_rm
        
        self.ret_per_class_scores = ret_per_class_scores
        
        if vol_batch_sz is not None:
            assert vol_batch_sz > 0
        self.vol_batch_sz = vol_batch_sz
        
    def _verify(self, mask_pred, mask_gt):
        assert mask_pred.shape == mask_gt.shape, f'mask_pred and mask_gt shapes do not match, {mask_pred.shape} != {mask_gt.shape}'
        # assert len(mask_pred.shape)==4, f'wrong mask shape length, should be 4 [N, n_class, H, W], but got: {len(mask_pred.shape)}' 
        assert mask_pred.shape[1]==self.n_class, f'mask prediction, wrong nb of classes, expected: {self.n_class}, got: {mask_pred.shape[1]}'
        assert (mask_gt>=0).all() and (mask_gt[mask_gt>0]==1).all(), 'mask gt can be 0 or 1'
        if self.prob_inputs:
           assert (mask_pred>=0).all() and (mask_pred<=1).all(), 'mask prediction out of bounds [0, 1]'
        
    def _get_probas(self, mask_pred):
        if not self.prob_inputs:
            mask_pred = F.softmax(mask_pred, dim=1)
        if not self.soft:
            dtype = mask_pred.dtype
            mask_pred = F.one_hot(torch.argmax(mask_pred, dim=1), self.n_class).transpose(-1, 1).to(dtype)  # [0, 3, 1, 2]
        return mask_pred  
    
    def _prep_inputs(self, mask_pred, mask_gt):
        # Verify shapes and values
        self._verify(mask_pred, mask_gt)
        
        # Prep mask prediction
        mask_pred = self._get_probas(mask_pred)
        
        # remove bg
        if self.bg_ch_to_rm is not None:
            fg_mask = (torch.arange(mask_pred.shape[1]) != self.bg_ch_to_rm)
            mask_pred = mask_pred[:, fg_mask, ...]  # Indexing, not slicing => returns a copy
            mask_gt = mask_gt[:, fg_mask, ...]
        
        return mask_pred, mask_gt
    
    def _score_reduction(self, score, dim=None):
        assert (score>=0).all(), 'Score has negative components !'
        if self.reduction == 'mean':
            return score.mean(dim=dim)
        elif self.reduction == 'sum':
            return score.sum(dim=dim)
        else:
            return score
        
    def _put_in_res_dict(self, scores):
        '''scores: [N, (C or C-1)]'''
        offset = 0
        if self.bg_ch_to_rm is None:
            assert scores.shape[-1] == self.n_class
        else:
            assert scores.shape[-1] == self.n_class - 1
            if self.bg_ch_to_rm==0:
                offset = 1
                
        res = {'':self._score_reduction(score=scores)}
        if self.ret_per_class_scores:
            scores_per_class = self._score_reduction(scores, dim=0)
            for i in range(scores.shape[-1]):
                res[f'_class{i+offset}'] = scores_per_class[i]
        return res
        
    @abstractmethod
    def _compute_score(self, mask_pred, mask_gt):
        """mask_pred and mask_gt : [N, C, ...] in [0, 1]"""
        pass
    
    
    def get_res_dict(self, inputs, target_oneHot, depth_idx=None):
        # Verify and prep inputs
        mask_pred, mask_gt = self._prep_inputs(inputs, target_oneHot)
        
        # Calc score ever 2D
        if depth_idx is None:
            scores = self._compute_score(mask_pred, mask_gt)
            return self._put_in_res_dict(scores)
        
        # Calc score over 3D 
        else:
            assert self.vol_batch_sz is not None
            
            if depth_idx % self.vol_batch_sz == 0:
                # Reinit the slices for the next volume batch
                self.slices_mask_pred = mask_pred.unsqueeze(dim=-3)
                self.slices_mask_gt = mask_gt.unsqueeze(dim=-3)
                return {}  # Return an empty dict 
                
            elif depth_idx % self.vol_batch_sz == self.vol_batch_sz-1:
                # Compute the scores over the complete volume
                scores = self._compute_score(self.slices_mask_pred, self.slices_mask_gt)
                return self._put_in_res_dict(scores)
            
            else:
                # Concat on depth dimension
                self.slices_mask_pred = torch.concat([self.slices_mask_pred, mask_pred.unsqueeze(dim=-3)], dim=-3)
                self.slices_mask_gt = torch.concat([self.slices_mask_gt, mask_gt.unsqueeze(dim=-3)], dim=-3)
                return {}  # Return an empty dict 
    
    def forward(self, inputs, target_oneHot):
        # Verify and prep inputs
        mask_pred, mask_gt = self._prep_inputs(inputs, target_oneHot)
        
        # Compute the scores
        scores = self._compute_score(mask_pred, mask_gt)  # scores: [N, C]
        
        # Apply the selcted reduction
        score_red = self._score_reduction(scores)
        return score_red

class mIoU(SegScoreBase):
    def __init__(self, 
                 n_class, 
                 prob_inputs=False, 
                 soft=True,
                 bg_ch_to_rm=None,
                 reduction='mean',
                 ret_per_class_scores=True,
                 vol_batch_sz=None):
        super(mIoU, self).__init__(n_class=n_class, 
                                   prob_inputs=prob_inputs, 
                                   soft=soft,  # score => not differentiable => can be hard
                                   bg_ch_to_rm=bg_ch_to_rm,
                                   reduction=reduction,
                                   ret_per_class_scores=ret_per_class_scores,
                                   vol_batch_sz=vol_batch_sz)  
        

    def _compute_score(self, mask_pred, mask_gt):
        # Batch size and num_class (without the ignored one)
        N, C = mask_pred.shape[:2] # [N x n_class x H x W]

        # Numerator Product
        inter = mask_pred * mask_gt
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.reshape(N, C, -1).sum(-1)

        #Denominator 
        union= mask_pred + mask_gt - (mask_pred*mask_gt)
        # Sum over all pixels N x C x H x W => N x C
        union = union.reshape(N, C, -1).sum(-1)
        
        # Score per batch and classes
        iou = inter/union  # N x C

        return iou
    
class mIoULoss(mIoU):
    def __init__(self, 
                 n_class, 
                 prob_inputs=False, 
                 bg_ch_to_rm=None,
                 reduction='mean'):
        super().__init__(n_class=n_class, 
                         prob_inputs=prob_inputs, 
                         soft=True,  # loss => must be differentiable => soft
                         bg_ch_to_rm=bg_ch_to_rm,
                         reduction=reduction,
                         ret_per_class_scores=False,
                         vol_batch_sz=None)
    
    def forward(self, inputs, target_oneHot):
        return 1 - super().forward(inputs, target_oneHot)
    
    

class DiceScore(SegScoreBase):
    def __init__(self, 
                 n_class, 
                 prob_inputs=False, 
                 soft=True,
                 bg_ch_to_rm=None,
                 reduction='mean',
                 k=1, 
                 epsilon=1e-6,
                 vol_batch_sz=None) -> None:
        super().__init__(n_class=n_class, 
                         prob_inputs=prob_inputs, 
                         soft=soft,  # score => not differentiable => can be hard
                         bg_ch_to_rm=bg_ch_to_rm,
                         reduction=reduction,
                         vol_batch_sz=vol_batch_sz)
        assert epsilon>0, f'Epsilon must be positive, got: {epsilon}'
        self.epsilon=epsilon
        assert k>0, f'k must be positive, got: {k}'
        self.k = k
   
        
    def _compute_score(self, mask_pred, mask_gt):
        # Batch size and num_class (without the ignored one)
        N, C = mask_pred.shape[:2] # [N x n_class x H x W]
        
        # Flatten the img dimensions
        mask_pred = mask_pred.reshape(N, C, -1)
        mask_gt = mask_gt.reshape(N, C, -1)

        # Compute the dices over batches x classes
        inter = torch.sum(mask_gt * mask_pred, dim=-1)
        pred = torch.sum(mask_pred ** self.k, dim=-1)
        gt = torch.sum(mask_gt ** self.k, dim=-1)
        dices = (2 * inter + self.epsilon) / (pred + gt + self.epsilon)  # [N, C]

        return dices
    
class DiceLoss(DiceScore):
    def __init__(self, 
                 n_class, 
                 prob_inputs=False, 
                 bg_ch_to_rm=None,
                 reduction='mean'):
        assert reduction in ['mean', 'sum']
        super().__init__(n_class=n_class, 
                         prob_inputs=prob_inputs, 
                         soft=True,  # loss => must be differentiable => soft
                         bg_ch_to_rm=bg_ch_to_rm,
                         reduction=reduction,
                         vol_batch_sz=None)
    
    def forward(self, inputs, target_oneHot):
        return 1 - super().forward(inputs, target_oneHot)


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

