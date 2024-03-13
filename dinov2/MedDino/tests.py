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
main_pth = "../../DataFoundationModels/hdf5/"
# main_pth = "/usr/bmicnas02/data-biwi-01/foundation_models/da_data"

sub_path = "brain/hcp/"
filename = "data_T1_original_depth_256_from_0_to_20.hdf5"

pth = main_pth + sub_path + filename

train_idx = 10
val_idx = 2
test_idx = 8

with h5py.File(pth, "r") as f:

    print("Keys: %s" % f.keys())
    
    img_shape = f["images"].shape
    lab_shape = f["labels"].shape
    print(f'Images shape: {img_shape}')
    print(f'Labels shape: {lab_shape}')
    
    assert train_idx + val_idx + test_idx == img_shape[0] == lab_shape[0]
    
    train_img = f['images'][:train_idx]
    train_lab = f['labels'][:train_idx]
    
    val_img = f['images'][train_idx:val_idx+train_idx]
    val_lab = f['labels'][train_idx:val_idx+train_idx]
    
    test_img = f['images'][val_idx+train_idx:]
    test_lab = f['labels'][val_idx+train_idx:]
    

hf_train = h5py.File(main_pth+sub_path+'train.hdf5', 'w')
hf_train.create_dataset('images', data=train_img)
hf_train.create_dataset('labels', data=train_lab)
hf_train.close()


hf_val = h5py.File(main_pth+sub_path+'val.hdf5', 'w')
hf_val.create_dataset('images', data=val_img)
hf_val.create_dataset('labels', data=val_lab)
hf_val.close()

hf_test = h5py.File(main_pth+sub_path+'test.hdf5', 'w')
hf_test.create_dataset('images', data=test_img)
hf_test.create_dataset('labels', data=test_lab)
hf_test.close()


print("Files are created")



# pth = main_pth+sub_path+'test.hdf5'

# with h5py.File(pth, "r") as f:

#     print("Keys: %s" % f.keys())
    
#     img_shape = f["images"].shape
#     lab_shape = f["labels"].shape
#     print(f'Images shape: {img_shape}')
#     print(f'Labels shape: {lab_shape}')
    

    
# print()