from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch
from abc import ABC, abstractmethod


class ScoreBase(nn.Module, ABC):
    def __init__(self, 
                 prob_inputs=False, 
                 soft=True, 
                 bg_ch_to_rm=None,
                 reduction='mean',
                 ret_per_class_scores=True,
                 vol_batch_sz=None,
                 log_probas=False):
        super(ScoreBase, self).__init__()
        
 
        self.prob_inputs=prob_inputs
        self.soft=soft
        
        assert reduction in ['none', 'mean', 'sum'], f'Undefined reduction: {reduction}'
        self.reduction=reduction
        
        self.bg_ch_to_rm=bg_ch_to_rm
        
        self.ret_per_class_scores = ret_per_class_scores
        
        if vol_batch_sz is not None:
            assert vol_batch_sz > 0
        self.vol_batch_sz = vol_batch_sz
        
        self.log_probas = log_probas
        
    def _verify_proba(self, vec_proba, dim=1):
        # Verify proba range
        assert (vec_proba>=0).all(), f"Prediction probas can't be negative, but got value {vec_proba.min()}"
        assert (vec_proba<=1).all(), f"Prediction probas can't be >1, but got value {vec_proba.max()}"
        thres = 1e-5
        check_tensor = torch.abs(vec_proba.sum(dim=dim)-1)<thres
        assert check_tensor.all(), f"Prediction probas do not add up to 1, {torch.count_nonzero(check_tensor)} elems invalidate the threshold {thres}"
        
    def _verify(self, mask_pred, mask_gt):
        assert mask_pred.shape == mask_gt.shape, f'mask_pred and mask_gt shapes do not match, {mask_pred.shape} != {mask_gt.shape}'
        self._verify_proba(mask_gt)
        assert ((mask_gt==0) + (mask_gt==1)).all(), "mask_gt must be one-hot-encoded"
        
    def _get_probas(self, mask_pred):
        if not self.prob_inputs:
            if mask_pred.shape[1]>2:
                mask_pred = F.softmax(mask_pred, dim=1)    
            else:
                assert mask_pred.shape[1] == 2, "Must have at least 2 classes on channel dim (1)"
                mask_pred = F.sigmoid(mask_pred)
        
        self._verify_proba(mask_pred)
                            
        if not self.soft:
            mask_pred = F.one_hot(torch.argmax(mask_pred, dim=1), mask_pred.size(1)).permute([0, 3, 1, 2]).to(mask_pred)
        return mask_pred  
        
    
    def _prep_inputs(self, mask_pred, mask_gt):
        # One Hot Encode the integer lables
        if mask_pred.shape != mask_gt.shape:
            mask_gt = F.one_hot(mask_gt, mask_pred.size(1)).permute([0, 3, 1, 2]).to(mask_pred)
        
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
        assert (score<=1).all(), 'Score has components greater than 1 !'
        if self.reduction == 'mean':
            return score.mean(dim=dim)
        elif self.reduction == 'sum':
            return score.sum(dim=dim)
        else:
            return score
        
    def _put_in_res_dict(self, scores):
        '''scores: [N, (C or C-1)]'''
        offset = 0
        if not self.bg_ch_to_rm is None and self.bg_ch_to_rm==0:
            offset = 1
        
        if not self.ret_per_class_scores:        
            res = {'':self._score_reduction(score=scores)}
        else:
            scores_per_class = self._score_reduction(scores, dim=0)
            res = {'':self._score_reduction(score=scores_per_class)}
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
            batch_sz = inputs.size(0)
            assert self.vol_batch_sz%batch_sz == 0, 'Batch size must be a multiple of samples/volume'
            
            vol_minibatch_sz = self.vol_batch_sz / batch_sz
            
            if depth_idx % vol_minibatch_sz == 0:
                # Reinit the slices for the next volume batch
                self.slices_mask_pred = [mask_pred.transpose(0, 1)]
                self.slices_mask_gt = [mask_gt.transpose(0, 1)]
                
            else:
                # Concat on depth dimension
                # [N, C, H, W] --> [C, N, H, W]
                self.slices_mask_pred.append(mask_pred.transpose(0, 1))
                self.slices_mask_gt.append(mask_gt.transpose(0, 1))
                
                if depth_idx % vol_minibatch_sz == (vol_minibatch_sz-1):
                    # Compute the scores over the complete volume
                    scores = self._compute_score(torch.cat(self.slices_mask_pred, dim=-3).unsqueeze(0), \
                                                 torch.cat(self.slices_mask_gt, dim=-3).unsqueeze(0))  # [1, C, D, H, W]
                    self.slices_mask_pred.clear()
                    self.slices_mask_gt.clear()
                    return self._put_in_res_dict(scores)
                
            return {}  # Return an empty dict 
    
    def forward(self, inputs, target_oneHot):
        # Verify and prep inputs
        mask_pred, mask_gt = self._prep_inputs(inputs, target_oneHot)
        
        # Compute the scores
        scores = self._compute_score(mask_pred, mask_gt)  # scores: [N, C] or [N]
        
        # Apply the selcted reduction
        score_red = self._score_reduction(scores)
        return score_red
    

class mIoU(ScoreBase):
    def __init__(self, 
                 prob_inputs=False, 
                 soft=False,
                 bg_ch_to_rm=None,
                 reduction='mean',
                 ret_per_class_scores=True,
                 vol_batch_sz=None,
                 epsilon=1e-6):
        super(mIoU, self).__init__(prob_inputs=prob_inputs, 
                                   soft=soft,  # score => not differentiable => can be hard
                                   bg_ch_to_rm=bg_ch_to_rm,
                                   reduction=reduction,
                                   ret_per_class_scores=ret_per_class_scores,
                                   vol_batch_sz=vol_batch_sz)  
        
        assert epsilon>0, f'Epsilon must be positive, got: {epsilon}'
        self.epsilon=epsilon
        

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
        iou = (inter+self.epsilon)/(union+self.epsilon)  # N x C

        return iou
     

class DiceScore(ScoreBase):
    def __init__(self, 
                 prob_inputs=False, 
                 soft=False,
                 bg_ch_to_rm=None,
                 reduction='mean',
                 k=1, 
                 epsilon=1e-6,
                 vol_batch_sz=None,
                 ret_per_class_scores=True) -> None:
        super().__init__(prob_inputs=prob_inputs, 
                         soft=soft,  # score => not differentiable => can be hard
                         bg_ch_to_rm=bg_ch_to_rm,
                         reduction=reduction,
                         ret_per_class_scores=ret_per_class_scores,
                         vol_batch_sz=vol_batch_sz,
                         )
        assert epsilon>0, f'Epsilon must be positive, got: {epsilon}'
        self.epsilon=epsilon
        assert k>0, f'k must be positive, got: {k}'
        self.k = k
   
        
    def _compute_score(self, mask_pred, mask_gt):
        # Batch size and num_class (without the ignored one)
        N, C = mask_pred.shape[:2] # [N x n_class x H x W]

        # Compute the dices over batches x classes
        inter = (mask_gt * mask_pred).reshape(N, C, -1).sum(dim=-1)
        pred = (mask_pred ** self.k).reshape(N, C, -1).sum(dim=-1) if self.k!=1 else mask_pred.reshape(N, C, -1).sum(dim=-1)
        gt = (mask_gt ** self.k).reshape(N, C, -1).sum(dim=-1)  if self.k!=1 else mask_gt.reshape(N, C, -1).sum(dim=-1)
        dices = (2 * inter + self.epsilon) / (pred + gt + self.epsilon)  # [N, C]

        return dices



