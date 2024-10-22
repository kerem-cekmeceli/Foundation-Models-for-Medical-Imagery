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
                 ignore_idxs=None,
                 reduction='mean',
                 ret_per_class_scores=True,
                 EN_vol_scores=False,
                 log_probas=False,
                 weight=None):
        super(ScoreBase, self).__init__()
        
        if weight is not None:
            weight = torch.Tensor(weight)
        self.weight = weight
        self.prob_inputs=prob_inputs
        self.soft=soft
        
        assert reduction in ['none', 'mean', 'sum'], f'Undefined reduction: {reduction}'
        self.reduction=reduction
        
        if ignore_idxs is None:
            ignore_idxs = []
        elif isinstance(ignore_idxs, int):
            ignore_idxs = [ignore_idxs]
        else:
            ignore_idxs = list(ignore_idxs)
        self.ignore_idxs=ignore_idxs
        
        self.ret_per_class_scores = ret_per_class_scores
        
        self.EN_vol_scores = EN_vol_scores  # Enables score to be computed over the volume
        if EN_vol_scores:
            self.slices_mask_pred = []
            self.slices_mask_gt = []
        
        self.log_probas = log_probas
        
    def _verify_proba(self, vec_proba, dim=1):
        # Verify proba range
        assert (vec_proba>=0).all(), f"Prediction probas can't be negative, but got value {vec_proba.min()}"
        assert (vec_proba<=1).all(), f"Prediction probas can't be >1, but got value {vec_proba.max()}"
        thres = 1e-5
        check_tensor = torch.abs(vec_proba.sum(dim=dim)-1)<thres
        assert check_tensor.all(), f"Prediction probas do not add up to 1, {torch.count_nonzero(torch.logical_not(check_tensor))} elems invalidate the threshold {thres}"
        
    def _verify(self, mask_pred, mask_gt):
        assert mask_pred.shape == mask_gt.shape, f'mask_pred and mask_gt shapes do not match, {mask_pred.shape} != {mask_gt.shape}'
        self._verify_proba(mask_gt)
        assert ((mask_gt==0) + (mask_gt==1)).all(), "mask_gt must be one-hot-encoded"
        
    def _get_probas(self, mask_pred):
        if not self.prob_inputs:
            if True: #mask_pred.shape[1]>2:
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
        if self.ignore_idxs:
            fg_mask = torch.ones(mask_pred.shape[1], dtype=torch.bool)
            for i in self.ignore_idxs:
                fg_mask &= (torch.arange(mask_pred.shape[1]) != i)
          
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
        if self.ignore_idxs and (0 in self.ignore_idxs):
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
    
    def _apply_weight(self, scores_per_class):
        # Apply the weights per class
        if self.weight is not None:
            assert scores_per_class.size(-1)==self.weight.size(-1), \
                f"Nb of classes in the weight tensor do not match the data, expected {scores_per_class.size(-1)}, got {self.weight.size(-1)}"
            scores_per_class = scores_per_class * self.weight.unsqueeze(0)
        return scores_per_class
    
    
    def score_over_vol(self, mask_pred, mask_gt, last_slice):
        assert mask_pred.shape[0] == mask_gt.shape[0] == last_slice.shape[0], 'Should have the same batch dim'
        assert len(last_slice.shape) == 1, 'last_slice should be a flat 1D tensor'
        
        # Need to append => All slices are from the same volume
        res = []
        if (last_slice[:-1]==False).all() or len(last_slice[:-1])==0:
            # Concat on depth dimension
            # [N, C, H, W] --> [C, N, H, W]
            self.slices_mask_pred.append(mask_pred.transpose(0, 1))
            self.slices_mask_gt.append(mask_gt.transpose(0, 1))
        
            # Last element is the end of the volume => compute the score
            if last_slice[-1]==True:
                # Compute the scores over the complete volume
                # scores : [1, C]
                scores = self._compute_score(torch.cat(self.slices_mask_pred, dim=-3).unsqueeze(0), \
                                                torch.cat(self.slices_mask_gt, dim=-3).unsqueeze(0))  # [1, C, D, H, W]
                # Apply the weight
                scores = self._apply_weight(scores)
                self.slices_mask_pred.clear()
                self.slices_mask_gt.clear()
                res.append(self._put_in_res_dict(scores))
                
            else:
                # Appending empty dict, res calculation is pending for the completion of the volume
                res.append({})
        
        # Need to seperate the batch so that scores can be computed using slices corresponding to the same 3d volume    
        else:
            thres_idx = torch.nonzero(last_slice)[0] + 1
            assert thres_idx < last_slice.shape[0]
            
            mask_pred_vol_end = mask_pred[:thres_idx]
            mask_gt_vol_end = mask_gt[:thres_idx]
            last_slice_vol_end = last_slice[:thres_idx]
            res.extend(self.score_over_vol(mask_pred=mask_pred_vol_end, mask_gt=mask_gt_vol_end, last_slice=last_slice_vol_end))
            
            mask_pred_rest = mask_pred[thres_idx:]
            mask_gt_rest = mask_gt[thres_idx:]
            last_slice_rest = last_slice[thres_idx:]
            res.extend(self.score_over_vol(mask_pred=mask_pred_rest, mask_gt=mask_gt_rest, last_slice=last_slice_rest))
            
        return res
                
    
    def get_res_dict(self, inputs, target_oneHot, last_slice=None):
        # Verify and prep inputs
        mask_pred, mask_gt = self._prep_inputs(inputs, target_oneHot)
        
        # Calc score ever 2D
        if last_slice is None:
            scores = self._compute_score(mask_pred, mask_gt)
            # Apply the weight
            scores = self._apply_weight(scores)
            return self._put_in_res_dict(scores)
        
        # Calc score over 3D 
        else:
            assert self.EN_vol_scores, 'Computation over the volume should be enabled during intialization'
            return self.score_over_vol(mask_pred=mask_pred, mask_gt=mask_gt, last_slice=last_slice)
    
    def forward(self, inputs, target_oneHot):
        # Verify and prep inputs
        mask_pred, mask_gt = self._prep_inputs(inputs, target_oneHot)
        
        # Compute the scores
        scores = self._compute_score(mask_pred, mask_gt)  # scores: [N, C] or [N]
        
        # Apply the weight
        scores = self._apply_weight(scores)
        
        # Apply the selcted reduction
        score_red = self._score_reduction(scores)
        return score_red
    

class mIoU(ScoreBase):
    def __init__(self, 
                 prob_inputs=False, 
                 soft=False,
                 ignore_idxs=None,
                 reduction='mean',
                 ret_per_class_scores=True,
                 EN_vol_scores=True,
                 epsilon=1e-6,
                 weight=None):
        super(mIoU, self).__init__(prob_inputs=prob_inputs, 
                                   soft=soft,  # score => not differentiable => can be hard
                                   ignore_idxs=ignore_idxs,
                                   reduction=reduction,
                                   ret_per_class_scores=ret_per_class_scores,
                                   EN_vol_scores=EN_vol_scores,
                                   weight=weight)  
        
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
                 ignore_idxs=None,
                 reduction='mean',
                 k=1, 
                 epsilon=1e-6,
                 EN_vol_scores=True,
                 ret_per_class_scores=True,
                 weight=None) -> None:
        super().__init__(prob_inputs=prob_inputs, 
                         soft=soft,  # score => not differentiable => can be hard
                         ignore_idxs=ignore_idxs,
                         reduction=reduction,
                         ret_per_class_scores=ret_per_class_scores,
                         EN_vol_scores=EN_vol_scores,
                         weight=weight)
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



