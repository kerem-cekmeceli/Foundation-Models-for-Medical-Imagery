import sys
import pathlib
 
dino_main_pth = pathlib.Path(__file__).parent.parent
orig_dino_pth = dino_main_pth / 'OrigDino'
sys.path.insert(1, dino_main_pth.as_posix())
sys.path.insert(2, orig_dino_pth.as_posix())

import torch
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmcv.runner import load_checkpoint
import cv2
import numpy as np
from matplotlib import pyplot as plt

from prep_model import get_bb_name, get_dino_backbone,\
        get_seg_head_config, get_seg_model, prep_img_tensor, \
        conv_to_numpy_img, get_pca_res, plot_batch_im
from OrigDino.dinov2.eval.segmentation import models

from mmseg.datasets.pipelines import Compose

from MedDino.med_dinov2.data.transforms import ElasticTransformation
import mmcv
import h5py


# import os
# import math
# import itertools
# import urllib
# from functools import partial
# from pathlib import Path
from PIL import Image
from mmseg.apis.inference import LoadImage
# from torchvision import transforms      
# import torch.nn.functional as F
# import mmcv
# from mmseg.models import build_backbone
# from mmseg.apis.inference import LoadImage
# from mmcv.parallel import collate, scatter
# from mmseg.datasets.pipelines import Compose


# seg_log_per_batch = 3
# batch_sz = 21

# sp = seg_log_per_batch+1
# log_idxs = torch.arange(batch_sz//sp, 
#                         batch_sz//sp*sp, 
#                         batch_sz//sp)

# print(log_idxs)

# log_idxs = log_idxs + (batch_sz%sp)//2

# print(log_idxs)

# print()


# pth = dino_main_pth / 'oup_imgs/orig.png'
# im = Image.open(pth).convert('RGB') # RGB
# # im = cv2.imread(str(pth)) # BGR
# # im = pth

# def put_in_res_dict(img, mask=None):
#         # if pil img : Convert PIL img (RGB) --> ndarray (BGR)
#         get_nd_arr = lambda img: np.array(img).copy() if isinstance(img, Image.Image) else img
#         # if ndarray convert RGB to BGR
#         rgb_2_bgr = lambda img: img[..., ::-1].copy() if isinstance(img, np.ndarray) else img
        
#         img = rgb_2_bgr(get_nd_arr(img))
#         result = dict(img=mmcv.imread(img, flag='color', channel_order='bgr'))
        
#         if mask is not None:
#                 mask = get_nd_arr(mask)
#                 result['seg_fields']=mmcv.imread(mask, flag='grayscale')
                
#         return result

# def rm_from_res_dict(results):
#         img = results['img']
#         if 'seg_fields' in results.keys():
#                 mask = results['seg_fields']
#                 return [img, mask]
#         return [img]


# transforms = []

# # Put in dict and convert to BGR
# transforms.append(put_in_res_dict)

# # Elastic deformations
# # transforms.append(dict(type='ElasticTransformation', data_aug_ratio=1.))

# # # random translation, rotation and scaling
# # transforms.append(dict(type='StructuralAug', data_aug_ratio=1.))

# # # random brightness, contrast(mode 0), saturation, hue, contrast(mode 1) | Img only
# # transforms.append(dict(type='PhotoMetricDistortion'))   

# # BGR->RGB and Normalize with mean and std given in the paper | Img only
# # img_transform.append(dict(type='Normalize', 
# #                           mean=[123.675, 116.28, 103.53],  #RGB
# #                           std=[58.395, 57.12, 57.375],  #RGB
# #                           to_rgb=True))

# # transforms.append(dict(type='CentralPad',
# #                           size_divisor=14,
# #                           pad_val=0, seg_pad_val=0))

# # transforms.append(dict(type='Resize2',
# #                         scale_factor=3., #HW
# #                         keep_ratio=True))

# transforms.append(dict(type='CentralCrop',  
#                               size_divisor=14))

# # mmseg/datasets/pipelines/transforms  
# # conv the img keys to torch.Tensor with [HWC] -> [CHW]
# transforms.append(dict(type='ImageToTensor', keys=['img']))

# # Remove the dict and keep the tensor
# transforms.append(rm_from_res_dict)

# transforms = Compose(transforms)

# out = transforms(im)

# im_t = out[0]


# plt.figure()
# plt.imshow(im_t.detach().cpu().numpy().transpose([1, 2, 0])[..., ::-1])
# plt.show()
# print()


# ###################]

# # import h5py


# # data_root_pth = dino_main_pth.parent.parent / 'DataFoundationModels/HCP1_hdf5'

# # dat_rsz = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5'
# # dat_ori = 'data_T1_original_depth_256_from_20_to_25.hdf5'


# # with h5py.File(str(data_root_pth/dat_ori), 'r') as data:
# #     print()
# #     print()
    
    
# # from queue import PriorityQueue

# # # q = PriorityQueue()

# # # # def put_in(val, dat):
# # # #     if q.full():
# # # #         if val <= q[]

# # # q.put((4, 'Read'))
# # # q.put((2, 'Play'))
# # # q.put((5, 'Write'))
# # # q.put((1, 'Code'))
# # # q.put((3, 'Study'))




# # from copy import deepcopy

# # class PriorityQueueFixedSz:
# #     def __init__(self, sz, keep_min=True):
# #         self.sz = sz
# #         self.minimize = keep_min
# #         self.pq = PriorityQueue(maxsize=sz)
        
# #     def update(self, score, data):            
# #         queue_score = -score if self.minimize else score
# #         new_el = (queue_score, data)
        
# #         if self.pq.full():
# #             old_worst = self.pq.get()
# #             if old_worst[0]<=new_el[0]:
# #                 self.pq.put(new_el)
# #             else:
# #                 self.pq.put(old_worst)
# #         else:
# #             self.pq.put((queue_score, data))


# #     def get_elems(self):
# #         els = []
# #         while not self.pq.empty():
# #             (queue_score, data) = self.pq.get()
# #             score = -queue_score if self.minimize else queue_score
# #             els.append((score, data))
# #         els.reverse()  # Best first
# #         return els


# # # Example usage
# # pq3 = PriorityQueueFixedSz(sz=3, keep_min=True)

# # # Feed data and scores
# # pq3.update(10, "Ten")
# # pq3.update(5, "Five")
# # pq3.update(8, "Eight")
# # pq3.update(3, "Three")
# # pq3.update(7, "Seven")
# # pq3.update(9, "Nine")
# # pq3.update(7, "Seven2")
# # pq3.update(4, "Four")
# # pq3.update(1, "One")


# # print()

# # import heapq

# # class ThreeSmallestScores:
# #     def __init__(self):
# #         self.smallest_scores = []
# #         self.minimize = True

# #     def update(self, score, data):
# #         if self.minimize:
# #             score = -score
# #         # Update the list with the 3 smallest scores and their associated data
# #         heapq.heappush(self.smallest_scores, (score, data))
# #         if len(self.smallest_scores) > 3:
# #             heapq.heappop(self.smallest_scores)

# #     def get_three_smallest(self):
# #         return sorted(self.smallest_scores)

# # # Example usage
# # three_smallest_handler = PriorityQueueFixedSz()

# # # Feed data and scores
# # three_smallest_handler.update(10, "Data1")
# # three_smallest_handler.update(5, "Data2")
# # three_smallest_handler.update(8, "Data3")
# # three_smallest_handler.update(3, "Data4")
# # three_smallest_handler.update(7, "Data5")

# # Get the 3 smallest scores and their associated data
# # result = pq3.get_three_smallest()
# # print(result)


# # print()

# ####################################


# # def train_all_batches(model : nn.Module, 
# #                       train_loader : Dataset, 
# #                       loss_fn : Callable, 
# #                       optimizer : torch.optim.Optimizer, 
# #                       device : Union[str, torch.device], 
# #                       logger : wandb.wandb_sdk.wandb_run.Run, 
# #                       epoch : int, 
# #                       log_batch : bool = False,
# #                       metrics : Optional[Dict[str, Callable]] = None) -> dict:
# #     model.train()
# #     batches = tqdm(train_loader, desc='Train Batches', leave=False)
# #     running_loss = 0
    
# #     tot_batches = len(batches)
# #     if tot_batches<=0:
# #         raise Exception('No data')
    
# #     metrics = {} if metrics is None else metrics
    
# #     # Init the epoch log dict (to be averaged over all the batches)
# #     log_epoch = dict(epoch=epoch, loss=0.)
# #     for metric_n in metrics.keys():
# #         log_epoch[metric_n] = 0.
    
# #     for i_batch, (x_batch, y_batch) in enumerate(batches):
# #         # Put the data on the selected device
# #         x_batch = x_batch.to(device=device) #@TODO model.device
# #         y_batch = y_batch.to(device=device)
        
# #         # Forward pass
# #         y_pred = model(x_batch)
# #         loss = loss_fn(y_pred, y_batch)
        
# #         # backward pass
# #         optimizer.zero_grad()
# #         loss.backward()
        
# #         # Update weights
# #         optimizer.step()
        
# #         ## Log the values
# #         if log_batch:
# #             log_batch = dict(batch_idx=i_batch, epoch=epoch, loss=loss.item)
# #         log_epoch['loss']+=loss.item
        
# #         # Compute the metrics
# #         for metric_n, metric in metrics.items():
# #             metric_val = metric(y_pred, y_batch).item
# #             if log_batch:
# #                 log_batch[metric_n] = metric_val
# #             log_epoch[metric_n] += metric_val
        
# #         if log_batch:    
# #             # Log the batch
# #             logger.log(log_batch)
    
# #     # Average out the epoch logs 
# #     for key in log_epoch.keys():
# #         log_epoch[key] /= tot_batches  
               
# #     return log_epoch

import h5py
filename = "../../DataFoundationModels/hdf5/brain/hcp1/resized/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5"
filename = "../../DataFoundationModels/hdf5/brain/hcp1/original/data_T1_original_depth_256_from_0_to_20.hdf5"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array
    print()



import math
from typing import TypeVar, Optional, Iterator

import torch
from . import Sampler, Dataset
import torch.distributed as dist

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)



class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch










