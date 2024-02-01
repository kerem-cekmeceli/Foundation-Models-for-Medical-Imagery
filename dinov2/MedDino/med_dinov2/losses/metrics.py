from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


class mIoU(nn.Module):
    def __init__(self, n_classes):
        super(mIoU, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1) # [B, n_class, h0, w0]
        
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union  # N x C

        ## Return average loss over classes and batch
        return loss.mean()  # minus since loss is  to be minimized nont maximized
    

    
# @TODO CrossEntropyLoss (Dino)
# @TODO Dice score implement  !! medical  (BG supression)
# @TODO keep miou as a metric for natural img comparison
# @TODO eval on 3d not per slice
