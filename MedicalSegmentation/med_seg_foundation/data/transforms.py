# # Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# # Modifications: Support override_scale in Resize

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from mmseg.datasets.builder import PIPELINES

import scipy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
import math
from typing import List, Optional
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# @PIPELINES.register_module()
# class Resize(object):
#     """Resize images & seg.

#     This transform resizes the input image to some scale. If the input dict
#     contains the key "scale", then the scale in the input dict is used,
#     otherwise the specified scale in the init method is used.

#     ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
#     (multi-scale). There are 4 multiscale modes:

#     - ``ratio_range is not None``:
#     1. When img_scale is None, img_scale is the shape of image in results
#         (img_scale = results['img'].shape[:2]) and the image is resized based
#         on the original size. (mode 1)
#     2. When img_scale is a tuple (single-scale), randomly sample a ratio from
#         the ratio range and multiply it with the image scale. (mode 2)

#     - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
#     scale from the a range. (mode 3)

#     - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
#     scale from multiple scales. (mode 4)

#     Args:
#         img_scale (tuple or list[tuple]): Images scales for resizing.
#             Default:None.
#         multiscale_mode (str): Either "range" or "value".
#             Default: 'range'
#         ratio_range (tuple[float]): (min_ratio, max_ratio).
#             Default: None
#         keep_ratio (bool): Whether to keep the aspect ratio when resizing the
#             image. Default: True
#     """

#     def __init__(self,
#                  img_scale=None,
#                  multiscale_mode='range',
#                  ratio_range=None,
#                  keep_ratio=True,
#                  override_scale=False):
#         if img_scale is None:
#             self.img_scale = None
#         else:
#             if isinstance(img_scale, list):
#                 self.img_scale = img_scale
#             else:
#                 self.img_scale = [img_scale]
#             assert mmcv.is_list_of(self.img_scale, tuple)

#         if ratio_range is not None:
#             # mode 1: given img_scale=None and a range of image ratio
#             # mode 2: given a scale and a range of image ratio
#             assert self.img_scale is None or len(self.img_scale) == 1
#         else:
#             # mode 3 and 4: given multiple scales or a range of scales
#             assert multiscale_mode in ['value', 'range']

#         self.multiscale_mode = multiscale_mode
#         self.ratio_range = ratio_range
#         self.keep_ratio = keep_ratio
#         self.override_scale = override_scale

#     @staticmethod
#     def random_select(img_scales):
#         """Randomly select an img_scale from given candidates.

#         Args:
#             img_scales (list[tuple]): Images scales for selection.

#         Returns:
#             (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
#                 where ``img_scale`` is the selected image scale and
#                 ``scale_idx`` is the selected index in the given candidates.
#         """

#         assert mmcv.is_list_of(img_scales, tuple)
#         scale_idx = np.random.randint(len(img_scales))
#         img_scale = img_scales[scale_idx]
#         return img_scale, scale_idx

#     @staticmethod
#     def random_sample(img_scales):
#         """Randomly sample an img_scale when ``multiscale_mode=='range'``.

#         Args:
#             img_scales (list[tuple]): Images scale range for sampling.
#                 There must be two tuples in img_scales, which specify the lower
#                 and upper bound of image scales.

#         Returns:
#             (tuple, None): Returns a tuple ``(img_scale, None)``, where
#                 ``img_scale`` is sampled scale and None is just a placeholder
#                 to be consistent with :func:`random_select`.
#         """

#         assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
#         img_scale_long = [max(s) for s in img_scales]
#         img_scale_short = [min(s) for s in img_scales]
#         long_edge = np.random.randint(
#             min(img_scale_long),
#             max(img_scale_long) + 1)
#         short_edge = np.random.randint(
#             min(img_scale_short),
#             max(img_scale_short) + 1)
#         img_scale = (long_edge, short_edge)
#         return img_scale, None

#     @staticmethod
#     def random_sample_ratio(img_scale, ratio_range):
#         """Randomly sample an img_scale when ``ratio_range`` is specified.

#         A ratio will be randomly sampled from the range specified by
#         ``ratio_range``. Then it would be multiplied with ``img_scale`` to
#         generate sampled scale.

#         Args:
#             img_scale (tuple): Images scale base to multiply with ratio.
#             ratio_range (tuple[float]): The minimum and maximum ratio to scale
#                 the ``img_scale``.

#         Returns:
#             (tuple, None): Returns a tuple ``(scale, None)``, where
#                 ``scale`` is sampled ratio multiplied with ``img_scale`` and
#                 None is just a placeholder to be consistent with
#                 :func:`random_select`.
#         """

#         assert isinstance(img_scale, tuple) and len(img_scale) == 2
#         min_ratio, max_ratio = ratio_range
#         assert min_ratio <= max_ratio
#         ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
#         scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
#         return scale, None

#     def _random_scale(self, results):
#         """Randomly sample an img_scale according to ``ratio_range`` and
#         ``multiscale_mode``.

#         If ``ratio_range`` is specified, a ratio will be sampled and be
#         multiplied with ``img_scale``.
#         If multiple scales are specified by ``img_scale``, a scale will be
#         sampled according to ``multiscale_mode``.
#         Otherwise, single scale will be used.

#         Args:
#             results (dict): Result dict from :obj:`dataset`.

#         Returns:
#             dict: Two new keys 'scale` and 'scale_idx` are added into
#                 ``results``, which would be used by subsequent pipelines.
#         """

#         if self.ratio_range is not None:
#             if self.img_scale is None:
#                 h, w = results['img'].shape[:2]
#                 scale, scale_idx = self.random_sample_ratio((w, h),
#                                                             self.ratio_range)
#             else:
#                 scale, scale_idx = self.random_sample_ratio(
#                     self.img_scale[0], self.ratio_range)
#         elif len(self.img_scale) == 1:
#             scale, scale_idx = self.img_scale[0], 0
#         elif self.multiscale_mode == 'range':
#             scale, scale_idx = self.random_sample(self.img_scale)
#         elif self.multiscale_mode == 'value':
#             scale, scale_idx = self.random_select(self.img_scale)
#         else:
#             raise NotImplementedError

#         results['scale'] = scale
#         results['scale_idx'] = scale_idx

#     def _resize_img(self, results):
#         """Resize images with ``results['scale']``."""
#         if self.keep_ratio:
#             img, scale_factor = mmcv.imrescale(
#                 results['img'], results['scale'], return_scale=True)
#             # the w_scale and h_scale has minor difference
#             # a real fix should be done in the mmcv.imrescale in the future
#             new_h, new_w = img.shape[:2]
#             h, w = results['img'].shape[:2]
#             w_scale = new_w / w
#             h_scale = new_h / h
#         else:
#             img, w_scale, h_scale = mmcv.imresize(
#                 results['img'], results['scale'], return_scale=True)
#         scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
#                                 dtype=np.float32)
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['pad_shape'] = img.shape  # in case that there is no padding
#         results['scale_factor'] = scale_factor
#         results['keep_ratio'] = self.keep_ratio

#     def _resize_seg(self, results):
#         """Resize semantic segmentation map with ``results['scale']``."""
#         for key in results.get('seg_fields', []):
#             if self.keep_ratio:
#                 gt_seg = mmcv.imrescale(
#                     results[key], results['scale'], interpolation='nearest')
#             else:
#                 gt_seg = mmcv.imresize(
#                     results[key], results['scale'], interpolation='nearest')
#             results[key] = gt_seg

#     def __call__(self, results):
#         """Call function to resize images, bounding boxes, masks, semantic
#         segmentation map.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
#                 'keep_ratio' keys are added into result dict.
#         """

#         if 'scale' not in results or self.override_scale:
#             self._random_scale(results)
#         self._resize_img(results)
#         self._resize_seg(results)
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += (f'(img_scale={self.img_scale}, '
#                      f'multiscale_mode={self.multiscale_mode}, '
#                      f'ratio_range={self.ratio_range}, '
#                      f'keep_ratio={self.keep_ratio})')
#         return repr_str


# @PIPELINES.register_module()
# class RandomFlip(object):
#     """Flip the image & seg.

#     If the input dict contains the key "flip", then the flag will be used,
#     otherwise it will be randomly decided by a ratio specified in the init
#     method.

#     Args:
#         prob (float, optional): The flipping probability. Default: None.
#         direction(str, optional): The flipping direction. Options are
#             'horizontal' and 'vertical'. Default: 'horizontal'.
#     """

#     @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
#     def __init__(self, prob=None, direction='horizontal'):
#         self.prob = prob
#         self.direction = direction
#         if prob is not None:
#             assert prob >= 0 and prob <= 1
#         assert direction in ['horizontal', 'vertical']

#     def __call__(self, results):
#         """Call function to flip bounding boxes, masks, semantic segmentation
#         maps.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Flipped results, 'flip', 'flip_direction' keys are added into
#                 result dict.
#         """

#         if 'flip' not in results:
#             flip = True if np.random.rand() < self.prob else False
#             results['flip'] = flip
#         if 'flip_direction' not in results:
#             results['flip_direction'] = self.direction
#         if results['flip']:
#             # flip image
#             results['img'] = mmcv.imflip(
#                 results['img'], direction=results['flip_direction'])

#             # flip segs
#             for key in results.get('seg_fields', []):
#                 # use copy() to make numpy stride positive
#                 results[key] = mmcv.imflip(
#                     results[key], direction=results['flip_direction']).copy()
#         return results

#     def __repr__(self):
#         return self.__class__.__name__ + f'(prob={self.prob})'


# @PIPELINES.register_module()
# class Pad(object):
#     """Pad the image & mask.

#     There are two padding modes: (1) pad to a fixed size and (2) pad to the
#     minimum size that is divisible by some number.
#     Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

#     Args:
#         size (tuple, optional): Fixed padding size.
#         size_divisor (int, optional): The divisor of padded size.
#         pad_val (float, optional): Padding value. Default: 0.
#         seg_pad_val (float, optional): Padding value of segmentation map.
#             Default: 255.
#     """

#     def __init__(self,
#                  size=None,
#                  size_divisor=None,
#                  pad_val=0,
#                  seg_pad_val=255):
#         self.size = size
#         self.size_divisor = size_divisor
#         self.pad_val = pad_val
#         self.seg_pad_val = seg_pad_val
#         # only one of size and size_divisor should be valid
#         assert size is not None or size_divisor is not None
#         assert size is None or size_divisor is None

#     def _pad_img(self, results):
#         """Pad images according to ``self.size``."""
#         if self.size is not None:
#             padded_img = mmcv.impad(
#                 results['img'], shape=self.size, pad_val=self.pad_val)
#         elif self.size_divisor is not None:
#             padded_img = mmcv.impad_to_multiple(
#                 results['img'], self.size_divisor, pad_val=self.pad_val)
#         results['img'] = padded_img
#         results['pad_shape'] = padded_img.shape
#         results['pad_fixed_size'] = self.size
#         results['pad_size_divisor'] = self.size_divisor

#     def _pad_seg(self, results):
#         """Pad masks according to ``results['pad_shape']``."""
#         for key in results.get('seg_fields', []):
#             results[key] = mmcv.impad(
#                 results[key],
#                 shape=results['pad_shape'][:2],
#                 pad_val=self.seg_pad_val)

#     def __call__(self, results):
#         """Call function to pad images, masks, semantic segmentation maps.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Updated result dict.
#         """

#         self._pad_img(results)
#         self._pad_seg(results)
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
#                     f'pad_val={self.pad_val})'
#         return repr_str


# @PIPELINES.register_module()
# class Normalize(object):
#     """Normalize the image.

#     Added key is "img_norm_cfg".

#     Args:
#         mean (sequence): Mean values of 3 channels.
#         std (sequence): Std values of 3 channels.
#         to_rgb (bool): Whether to convert the image from BGR to RGB,
#             default is true.
#     """

#     def __init__(self, mean, std, to_rgb=True):
#         self.mean = np.array(mean, dtype=np.float32)
#         self.std = np.array(std, dtype=np.float32)
#         self.to_rgb = to_rgb

#     def __call__(self, results):
#         """Call function to normalize images.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Normalized results, 'img_norm_cfg' key is added into
#                 result dict.
#         """

        # results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
        #                                   self.to_rgb)
#         results['img_norm_cfg'] = dict(
#             mean=self.mean, std=self.std, to_rgb=self.to_rgb)
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
#                     f'{self.to_rgb})'
#         return repr_str


# @PIPELINES.register_module()
# class Rerange(object):
#     """Rerange the image pixel value.

#     Args:
#         min_value (float or int): Minimum value of the reranged image.
#             Default: 0.
#         max_value (float or int): Maximum value of the reranged image.
#             Default: 255.
#     """

#     def __init__(self, min_value=0, max_value=255):
#         assert isinstance(min_value, float) or isinstance(min_value, int)
#         assert isinstance(max_value, float) or isinstance(max_value, int)
#         assert min_value < max_value
#         self.min_value = min_value
#         self.max_value = max_value

#     def __call__(self, results):
#         """Call function to rerange images.

#         Args:
#             results (dict): Result dict from loading pipeline.
#         Returns:
#             dict: Reranged results.
#         """

#         img = results['img']
#         img_min_value = np.min(img)
#         img_max_value = np.max(img)

#         assert img_min_value < img_max_value
#         # rerange to [0, 1]
#         img = (img - img_min_value) / (img_max_value - img_min_value)
#         # rerange to [min_value, max_value]
#         img = img * (self.max_value - self.min_value) + self.min_value
#         results['img'] = img

#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
#         return repr_str


# @PIPELINES.register_module()
# class CLAHE(object):
#     """Use CLAHE method to process the image.

#     See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
#     Graphics Gems, 1994:474-485.` for more information.

#     Args:
#         clip_limit (float): Threshold for contrast limiting. Default: 40.0.
#         tile_grid_size (tuple[int]): Size of grid for histogram equalization.
#             Input image will be divided into equally sized rectangular tiles.
#             It defines the number of tiles in row and column. Default: (8, 8).
#     """

#     def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
#         assert isinstance(clip_limit, (float, int))
#         self.clip_limit = clip_limit
#         assert is_tuple_of(tile_grid_size, int)
#         assert len(tile_grid_size) == 2
#         self.tile_grid_size = tile_grid_size

#     def __call__(self, results):
#         """Call function to Use CLAHE method process images.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Processed results.
#         """

#         for i in range(results['img'].shape[2]):
#             results['img'][:, :, i] = mmcv.clahe(
#                 np.array(results['img'][:, :, i], dtype=np.uint8),
#                 self.clip_limit, self.tile_grid_size)

#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(clip_limit={self.clip_limit}, '\
#                     f'tile_grid_size={self.tile_grid_size})'
#         return repr_str


# @PIPELINES.register_module()
# class RandomCrop(object):
#     """Random crop the image & seg.

#     Args:
#         crop_size (tuple): Expected size after cropping, (h, w).
#         cat_max_ratio (float): The maximum ratio that single category could
#             occupy.
#     """

#     def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
#         assert crop_size[0] > 0 and crop_size[1] > 0
#         self.crop_size = crop_size
#         self.cat_max_ratio = cat_max_ratio
#         self.ignore_index = ignore_index

#     def get_crop_bbox(self, img):
#         """Randomly get a crop bounding box."""
#         margin_h = max(img.shape[0] - self.crop_size[0], 0)
#         margin_w = max(img.shape[1] - self.crop_size[1], 0)
#         offset_h = np.random.randint(0, margin_h + 1)
#         offset_w = np.random.randint(0, margin_w + 1)
#         crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
#         crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

#         return crop_y1, crop_y2, crop_x1, crop_x2

#     def crop(self, img, crop_bbox):
#         """Crop from ``img``"""
#         crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
#         img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
#         return img

#     def __call__(self, results):
#         """Call function to randomly crop images, semantic segmentation maps.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Randomly cropped results, 'img_shape' key in result dict is
#                 updated according to crop size.
#         """

#         img = results['img']
#         crop_bbox = self.get_crop_bbox(img)
#         if self.cat_max_ratio < 1.:
#             # Repeat 10 times
#             for _ in range(10):
#                 seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
#                 labels, cnt = np.unique(seg_temp, return_counts=True)
#                 cnt = cnt[labels != self.ignore_index]
#                 if len(cnt) > 1 and np.max(cnt) / np.sum(
#                         cnt) < self.cat_max_ratio:
#                     break
#                 crop_bbox = self.get_crop_bbox(img)

#         # crop the image
#         img = self.crop(img, crop_bbox)
#         img_shape = img.shape
#         results['img'] = img
#         results['img_shape'] = img_shape

#         # crop semantic seg
#         for key in results.get('seg_fields', []):
#             results[key] = self.crop(results[key], crop_bbox)

#         return results

#     def __repr__(self):
#         return self.__class__.__name__ + f'(crop_size={self.crop_size})'


# @PIPELINES.register_module()
# class RandomRotate(object):
#     """Rotate the image & seg.

#     Args:
#         prob (float): The rotation probability.
#         degree (float, tuple[float]): Range of degrees to select from. If
#             degree is a number instead of tuple like (min, max),
#             the range of degree will be (``-degree``, ``+degree``)
#         pad_val (float, optional): Padding value of image. Default: 0.
#         seg_pad_val (float, optional): Padding value of segmentation map.
#             Default: 255.
#         center (tuple[float], optional): Center point (w, h) of the rotation in
#             the source image. If not specified, the center of the image will be
#             used. Default: None.
#         auto_bound (bool): Whether to adjust the image size to cover the whole
#             rotated image. Default: False
#     """

#     def __init__(self,
#                  prob,
#                  degree,
#                  pad_val=0,
#                  seg_pad_val=255,
#                  center=None,
#                  auto_bound=False):
#         self.prob = prob
#         assert prob >= 0 and prob <= 1
#         if isinstance(degree, (float, int)):
#             assert degree > 0, f'degree {degree} should be positive'
#             self.degree = (-degree, degree)
#         else:
#             self.degree = degree
#         assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
#                                       f'tuple of (min, max)'
#         self.pal_val = pad_val
#         self.seg_pad_val = seg_pad_val
#         self.center = center
#         self.auto_bound = auto_bound

#     def __call__(self, results):
#         """Call function to rotate image, semantic segmentation maps.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Rotated results.
#         """

#         rotate = True if np.random.rand() < self.prob else False
#         degree = np.random.uniform(min(*self.degree), max(*self.degree))
#         if rotate:
#             # rotate image
#             results['img'] = mmcv.imrotate(
#                 results['img'],
#                 angle=degree,
#                 border_value=self.pal_val,
#                 center=self.center,
#                 auto_bound=self.auto_bound)

#             # rotate segs
#             for key in results.get('seg_fields', []):
#                 results[key] = mmcv.imrotate(
#                     results[key],
#                     angle=degree,
#                     border_value=self.seg_pad_val,
#                     center=self.center,
#                     auto_bound=self.auto_bound,
#                     interpolation='nearest')
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(prob={self.prob}, ' \
#                     f'degree={self.degree}, ' \
#                     f'pad_val={self.pal_val}, ' \
#                     f'seg_pad_val={self.seg_pad_val}, ' \
#                     f'center={self.center}, ' \
#                     f'auto_bound={self.auto_bound})'
#         return repr_str


# @PIPELINES.register_module()
# class RGB2Gray(object):
#     """Convert RGB image to grayscale image.

#     This transform calculate the weighted mean of input image channels with
#     ``weights`` and then expand the channels to ``out_channels``. When
#     ``out_channels`` is None, the number of output channels is the same as
#     input channels.

#     Args:
#         out_channels (int): Expected number of output channels after
#             transforming. Default: None.
#         weights (tuple[float]): The weights to calculate the weighted mean.
#             Default: (0.299, 0.587, 0.114).
#     """

#     def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
#         assert out_channels is None or out_channels > 0
#         self.out_channels = out_channels
#         assert isinstance(weights, tuple)
#         for item in weights:
#             assert isinstance(item, (float, int))
#         self.weights = weights

#     def __call__(self, results):
#         """Call function to convert RGB image to grayscale image.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Result dict with grayscale image.
#         """
#         img = results['img']
#         assert len(img.shape) == 3
#         assert img.shape[2] == len(self.weights)
#         weights = np.array(self.weights).reshape((1, 1, -1))
#         img = (img * weights).sum(2, keepdims=True)
#         if self.out_channels is None:
#             img = img.repeat(weights.shape[2], axis=2)
#         else:
#             img = img.repeat(self.out_channels, axis=2)

#         results['img'] = img
#         results['img_shape'] = img.shape

#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(out_channels={self.out_channels}, ' \
#                     f'weights={self.weights})'
#         return repr_str


# @PIPELINES.register_module()
# class AdjustGamma(object):
#     """Using gamma correction to process the image.

#     Args:
#         gamma (float or int): Gamma value used in gamma correction.
#             Default: 1.0.
#     """

#     def __init__(self, gamma=1.0):
#         assert isinstance(gamma, float) or isinstance(gamma, int)
#         assert gamma > 0
#         self.gamma = gamma
#         inv_gamma = 1.0 / gamma
#         self.table = np.array([(i / 255.0)**inv_gamma * 255
#                                for i in np.arange(256)]).astype('uint8')

#     def __call__(self, results):
#         """Call function to process the image with gamma correction.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Processed results.
#         """

#         results['img'] = mmcv.lut_transform(
#             np.array(results['img'], dtype=np.uint8), self.table)

#         return results

#     def __repr__(self):
#         return self.__class__.__name__ + f'(gamma={self.gamma})'


# @PIPELINES.register_module()
# class SegRescale(object):
#     """Rescale semantic segmentation maps.

#     Args:
#         scale_factor (float): The scale factor of the final output.
#     """

#     def __init__(self, scale_factor=1):
#         self.scale_factor = scale_factor

#     def __call__(self, results):
#         """Call function to scale the semantic segmentation map.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Result dict with semantic segmentation map scaled.
#         """
#         for key in results.get('seg_fields', []):
#             if self.scale_factor != 1:
#                 results[key] = mmcv.imrescale(
#                     results[key], self.scale_factor, interpolation='nearest')
#         return results

#     def __repr__(self):
#         return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


# @PIPELINES.register_module()
# class PhotoMetricDistortion(object):
#     """Apply photometric distortion to image sequentially, every transformation
#     is applied with a probability of 0.5. The position of random contrast is in
#     second or second to last.

#     1. random brightness
#     2. random contrast (mode 0)
#     3. convert color from BGR to HSV
#     4. random saturation
#     5. random hue
#     6. convert color from HSV to BGR
#     7. random contrast (mode 1)

#     Args:
#         brightness_delta (int): delta of brightness.
#         contrast_range (tuple): range of contrast.
#         saturation_range (tuple): range of saturation.
#         hue_delta (int): delta of hue.
#     """

#     def __init__(self,
#                  brightness_delta=32,
#                  contrast_range=(0.5, 1.5),
#                  saturation_range=(0.5, 1.5),
#                  hue_delta=18):
#         self.brightness_delta = brightness_delta
#         self.contrast_lower, self.contrast_upper = contrast_range
#         self.saturation_lower, self.saturation_upper = saturation_range
#         self.hue_delta = hue_delta

#     def convert(self, img, alpha=1, beta=0):
#         """Multiple with alpha and add beat with clip."""
#         img = img.astype(np.float32) * alpha + beta
#         img = np.clip(img, 0, 255)
#         return img.astype(np.uint8)

#     def brightness(self, img):
#         """Brightness distortion."""
#         if random.randint(2):
#             return self.convert(
#                 img,
#                 beta=random.uniform(-self.brightness_delta,
#                                     self.brightness_delta))
#         return img

#     def contrast(self, img):
#         """Contrast distortion."""
#         if random.randint(2):
#             return self.convert(
#                 img,
#                 alpha=random.uniform(self.contrast_lower, self.contrast_upper))
#         return img

#     def saturation(self, img):
#         """Saturation distortion."""
#         if random.randint(2):
#             img = mmcv.bgr2hsv(img)
#             img[:, :, 1] = self.convert(
#                 img[:, :, 1],
#                 alpha=random.uniform(self.saturation_lower,
#                                      self.saturation_upper))
#             img = mmcv.hsv2bgr(img)
#         return img

#     def hue(self, img):
#         """Hue distortion."""
#         if random.randint(2):
#             img = mmcv.bgr2hsv(img)
#             img[:, :,
#                 0] = (img[:, :, 0].astype(int) +
#                       random.randint(-self.hue_delta, self.hue_delta)) % 180
#             img = mmcv.hsv2bgr(img)
#         return img

#     def __call__(self, results):
#         """Call function to perform photometric distortion on images.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Result dict with images distorted.
#         """

#         img = results['img']
#         # random brightness
#         img = self.brightness(img)

#         # mode == 0 --> do random contrast first
#         # mode == 1 --> do random contrast last
#         mode = random.randint(2)
#         if mode == 1:
#             img = self.contrast(img)

#         # random saturation
#         img = self.saturation(img)

#         # random hue
#         img = self.hue(img)
        
#         # random contrast
#         if mode == 0:
#             img = self.contrast(img)

#         results['img'] = img
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += (f'(brightness_delta={self.brightness_delta}, '
#                      f'contrast_range=({self.contrast_lower}, '
#                      f'{self.contrast_upper}), '
#                      f'saturation_range=({self.saturation_lower}, '
#                      f'{self.saturation_upper}), '
#                      f'hue_delta={self.hue_delta})')
#         return repr_str

@PIPELINES.register_module()
class ElasticTransformation(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, data_aug_ratio: float = 0.25,
                        sigma: float = 20,
                        alpha: float = 1000,
                        ):
        
        self.data_aug_ratio = data_aug_ratio
        self.sigma = sigma
        self.alpha = alpha

    def set_random_elastic_transform(
            self, shape, random_state=None  # 2d
        ):
            if random_state is None:
                random_state = np.random.RandomState(None)

            # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
            # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
            # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
            # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
            # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
            dx = (
                gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0
                )
                * self.alpha
            )
            dy = (
                gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0
                )
                *self.alpha
            )

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            return np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    def elastic_transform(
            self, image, indices, order
        ):
            shape = image.shape
            return map_coordinates(image, indices, order=order, mode="reflect").reshape(
                shape
            )
    
    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = results['img']
        img_shape = img.shape

        # ========
        # elastic deformation
        # ========
        if np.random.rand() < self.data_aug_ratio:
            indices = self.set_random_elastic_transform(img_shape[:2])

            for i in range(img.shape[2]):
                img[:, :, i] = self.elastic_transform(img[:, :, i], indices, order=1)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = self.elastic_transform(results[key], indices, order=0)

        results['img'] = img
        results['img_shape'] = img_shape
        
        return results

    def __repr__(self):    
        repr_str = self.__class__.__name__    
        repr_str += (f'data_aug_ratio=({self.data_aug_ratio}, '
                     f'sigma={self.sigma}, '
                     f'alpha={self.alpha})')
        
        return repr_str
    

@PIPELINES.register_module()
class StructuralAug(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, data_aug_ratio: float = 0.25,
                        trans_min: float = -10,
                        trans_max: float = 10,
                        rot_min: int = -10,
                        rot_max: int = 10,
                        scale_min: float = 0.9,
                        scale_max: float = 1.1,
                        ):
        
        self.data_aug_ratio = data_aug_ratio
        self.trans_min = trans_min
        self.trans_max = trans_max
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.scale_min = scale_min
        self.scale_max = scale_max

    def crop_or_pad_slice_to_size(self, slice, nx, ny):
        x, y = slice.shape

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        if x > nx and y > ny:
            slice_cropped = slice[x_s : x_s + nx, y_s : y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c : x_c + x, :] = slice[:, y_s : y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c : y_c + y] = slice[x_s : x_s + nx, :]
            else:
                slice_cropped[x_c : x_c + x, y_c : y_c + y] = slice[:, :]

        return slice_cropped
    
    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = results['img']
        img_shape = img.shape

        # ========
        # translation
        # ========
        if np.random.rand() < self.data_aug_ratio:
            random_shift_x = np.random.uniform(self.trans_min, self.trans_max)
            random_shift_y = np.random.uniform(self.trans_min, self.trans_max)

            for i in range(img.shape[2]):
                img[:, :, i] = scipy.ndimage.shift(
                    img[:, :, i], shift=(random_shift_x, random_shift_y), order=1
                )

            for key in results.get('seg_fields', []):
                results[key] = scipy.ndimage.shift(
                    results[key], shift=(random_shift_x, random_shift_y), order=0
                )

        # ========
        # rotation
        # ========
        if np.random.rand() < self.data_aug_ratio:
            random_angle = np.random.uniform(self.rot_min, self.rot_max)

            img = scipy.ndimage.rotate(
                img, reshape=False, angle=random_angle, axes=(1, 0), order=1
            )

            for key in results.get('seg_fields', []):
                results[key] = scipy.ndimage.rotate(
                    results[key], reshape=False, angle=random_angle, axes=(1, 0), order=0
                )

        # ========
        # scaling
        # ========
        if np.random.rand() < self.data_aug_ratio:
            n_x, n_y = img.shape[:2]

            scale_val = np.round(np.random.uniform(self.scale_min, self.scale_max), 2)

            for i in range(img.shape[2]):
                images_i_tmp = transform.rescale(
                    img[:, :, i], scale_val, order=1, preserve_range=True, mode="constant"
                )

                img[:, :, i] = self.crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)            

            for key in results.get('seg_fields', []):
                labels_i_tmp = transform.rescale(
                    results[key], scale_val, order=0, preserve_range=True, mode="constant"
                )
                results[key] = self.crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)

        results['img'] = img
        results['img_shape'] = img_shape
        
        return results

    def __repr__(self):    
        repr_str = self.__class__.__name__    
        repr_str += (f'data_aug_ratio=({self.data_aug_ratio}, '
                     f'trans_min={self.trans_min}, '
                     f'trans_max={self.trans_max}, '
                     f'rot_min={self.rot_min}, '
                     f'rot_max={self.rot_max}, '
                        f'scale_min={self.scale_min}, '
                        f'scale_max={self.scale_max})')
        
        return repr_str
    
    
@PIPELINES.register_module()
class CentralPad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.  [HW]
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 make_square=False,
                 pad_val=0,
                 seg_pad_val=255):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
          
        if not make_square:      
            # only one of size and size_divisor should be valid
            assert size is not None or size_divisor is not None
            assert size is None or size_divisor is None
        else:
            assert size is None and size_divisor is None 
            self.make_square = make_square
        
    def _get_central_pads(self, orig_sz):
        # Orig_sz [HWC]  self.size [HW]
        if self.size is not None:
            pad_w_tot = max(self.size[1] - orig_sz[1], 0)
            pad_h_tot = max(self.size[0] - orig_sz[0], 0)
            
        else:
            assert self.size_divisor > 0
            pad_w_tot = math.ceil(orig_sz[1] / self.size_divisor)*self.size_divisor - orig_sz[1]
            pad_h_tot = math.ceil(orig_sz[0] / self.size_divisor)*self.size_divisor - orig_sz[0]
            
        # Pads on left and right
        pad_l = pad_w_tot // 2
        pad_r = pad_w_tot - pad_l
        
        # Pads on top and bottom
        pad_t = pad_h_tot // 2
        pad_b = pad_h_tot - pad_t
        
        padding = (pad_l, pad_t, pad_r, pad_b)
        return padding

    def _pad_img(self, results):
        # Get the central padded image
        padded_img = mmcv.impad(results['img'], 
                                padding=self._get_central_pads(results['img'].shape), 
                                pad_val=self.pad_val)    
            
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                padding=self._get_central_pads(results[key].shape),
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        if self.make_square:
            sz = max(results['img'].shape[1], results['img'].shape[0])
            self.size = (sz, sz)
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str
    
@PIPELINES.register_module()
class CentralCrop(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size. [HW]
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 make_square=False):
        if size is not None:
            if isinstance(size, int):
                size = (size, size)
            else:
                assert isinstance(size, tuple)
        self.make_square = make_square
        self.size = size
        self.size_divisor = size_divisor
        
        if not self.make_square:        
            # only one of size and size_divisor should be valid
            assert self.size is not None or self.size_divisor is not None
            assert self.size is None or self.size_divisor is None
        else:
            assert self.size is None
            assert self.size_divisor is None
            
        
    def _get_central_crops(self, orig_sz):
        if self.make_square:
            short_edge = min(orig_sz[0], orig_sz[1])
            self.size = (short_edge, short_edge)
            
        # Orig_sz [HWC]
        if self.size is not None:
            crop_w_tot = max(orig_sz[1] - self.size[1], 0)
            crop_h_tot = max(orig_sz[0] - self.size[0], 0)
            
        else:
            assert self.size_divisor > 0
            crop_w_tot = orig_sz[1] - math.floor(orig_sz[1] / self.size_divisor)*self.size_divisor
            crop_h_tot = orig_sz[0] - math.floor(orig_sz[0] / self.size_divisor)*self.size_divisor
            
        # crops on left and right
        crop_l = crop_w_tot // 2
        crop_r = crop_w_tot - crop_l
        
        # Pads on top and bottom
        crop_t = crop_h_tot // 2
        crop_b = crop_h_tot - crop_t
        
        crops = (crop_l, crop_t, crop_r, crop_b)
        return crops

    def _crop(self, res):
        (crop_l, crop_t, crop_r, crop_b) = self._get_central_crops(res.shape)
        crop_b = res.shape[0] if crop_b==0 else -crop_b
        crop_r = res.shape[1] if crop_r==0 else -crop_r
        croped = res[crop_t:crop_b, crop_l:crop_r, ...]
        return croped
            
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        
        results['img'] = self._crop(results['img'])
        results['croped_shape'] = results['img'].shape
        results['crop_size_divisor'] = self.size_divisor
        
        for key in results.get('seg_fields', []):
            results[key] = self._crop(results[key])
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor})'
        return repr_str
    

from mmcv.image.geometric import _scale_size

@PIPELINES.register_module()
class Resize2(object):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation_up:Optional[str]=None,
                 interpolation_down:Optional[str]=None,
                 only_img:bool = False,
                 only_meta:bool = False) -> None:
        assert not (only_img and only_meta), f'`only_img` and `only_meta` can not be both True at the same time'
        
        self.only_img = only_img
        self.only_meta = only_meta
        
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        if interpolation_up is None: 
            self.interpolation_up = 'bilinear'
        else:
            self.interpolation_up = interpolation_up
        if interpolation_down is None:
            self.interpolation_down = 'area'
        else:
            self.interpolation_down = interpolation_down
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        
        # Choose interpolation type
        h, w = results['img'].shape[:2]
        up = w > results['scale'][1] and h > results['scale'][0]
        interpolation = self.interpolation_up if up else self.interpolation_down

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            # Choose interpolation type
            h, w = results['gt_seg_map'].shape[:2]
            up = w > results['scale'][1] and h > results['scale'][0]
            interpolation = self.interpolation_up if up else self.interpolation_down
            
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation=interpolation,
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation=interpolation,
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def __call__(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
            
        if not self.only_meta:
            self._resize_img(results)
        if not self.only_img:
            self._resize_bboxes(results)
            self._resize_seg(results)
            self._resize_keypoints(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
    
        

@PIPELINES.register_module()
class ResizeLongestSide(Resize2):
    def __init__(self, 
                 long_side_length: int,
                 keep_ratio = False, 
                 clip_object_border = True, 
                 backend = 'cv2', 
                 interpolation_up:Optional[str]=None,
                 interpolation_down:Optional[str]=None,
                 only_img = False,
                 only_meta = False) -> None:
        
        self.long_side_length =long_side_length
        
        super().__init__(scale=1,
                         keep_ratio=keep_ratio,
                         clip_object_border=clip_object_border,
                         backend=backend,
                         interpolation_up=interpolation_up,
                         interpolation_down=interpolation_down,
                         only_img=only_img,
                         only_meta=only_meta)
        
    def __call__(self, results: dict) -> dict:
        """
        Compute the output size given input size and target long side length.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """
        
        oldh, oldw = results['img'].shape[:2]
        
        scale = self.long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        self.scale = (newh, neww)
        results['scale'] = self.scale
        
        if not self.only_meta:
            self._resize_img(results)
        if not self.only_img:
            self._resize_bboxes(results)
            self._resize_seg(results)
            self._resize_keypoints(results)
        return results



# @PIPELINES.register_module()
# class MedPhotoMetricDistortion(object):
#     """Random crop the image & seg.

#     Args:
#         crop_size (tuple): Expected size after cropping, (h, w).
#         cat_max_ratio (float): The maximum ratio that single category could
#             occupy.
#     """

#     def __init__(self, data_aug_ratio: float = 0.25,
#                         gamma_min: float = 0.5,
#                         gamma_max: float = 2.0,
#                         brightness_min: float = 0.0,
#                         brightness_max: float = 0.1,
#                         noise_min: float = 0.0,
#                         noise_max: float = 0.1
#                         ):
        
#         self.data_aug_ratio = data_aug_ratio
#         self.gamma_min = gamma_min
#         self.gamma_max = gamma_max
#         self.brightness_min = brightness_min
#         self.brightness_max = brightness_max
#         self.noise_min = noise_min
#         self.noise_max = noise_max
    
#     def __call__(self, results):
#         """Call function to randomly crop images, semantic segmentation maps.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Randomly cropped results, 'img_shape' key in result dict is
#                 updated according to crop size.
#         """

#         img = results['img'].astype(np.float32)
#         img /= 255.0

#         # ========
#         # contrast
#         # ========
#         if np.random.rand() < self.data_aug_ratio:
#             # gamma contrast augmentation
#             img **= np.round(np.random.uniform(self.gamma_min, self.gamma_max), 2)

#         # ========
#         # brightness
#         # ========
#         if np.random.rand() < self.data_aug_ratio:
#             # brightness augmentation
#             img += np.round(np.random.uniform(self.brightness_min, self.brightness_max), 2)

#         # ========
#         # noise
#         # ========
#         if np.random.rand() < self.data_aug_ratio:
#             # noise augmentation
#             img += np.random.normal(self.noise_min, self.noise_max, size=img.shape)
        
#         results['img'] = np.clip(img * 255, 0, 255).astype(np.uint8)
   
#         return results

#     def __repr__(self):    
#         repr_str = self.__class__.__name__    
#         repr_str += (f'data_aug_ratio=({self.data_aug_ratio}, '
#                         f'gamma_min={self.gamma_min}, '
#                         f'gamma_max={self.gamma_max}, '
#                         f'brightness_min={self.brightness_min}, '
#                         f'brightness_max={self.brightness_max}, '
#                         f'noise_min={self.noise_min}, '
#                         f'noise_max={self.noise_max})')
        
#         return repr_str
    
# @PIPELINES.register_module()
# class ContrastFlip(object):
#     """Random crop the image & seg.

#     Args:
#         data_aug_ratio (float): The ratio of data augmentation.
#         gamma_min (float): The minimum gamma value.
#         gamma_max (float): The maximum gamma value.
#         background (float): The background value.
#         background_cl (int): The background class.
#         random_contrast (bool): Whether to use random contrast.
#     """

#     def __init__(self, data_aug_ratio: float = 0.25,
#                         gamma_min: float = 0.85,
#                         gamma_max: float = 1.20,
#                         background: float = -1.5,
#                         background_cl: int = 0,
#                         random_contrast: bool = False
#                         ):
        
#         self.data_aug_ratio = data_aug_ratio
#         self.gamma_min = gamma_min
#         self.gamma_max = gamma_max
#         self.background = background
#         self.background_cl = background_cl
#         self.random_contrast = random_contrast

#     def __call__(self, results):
#         """Call function to randomly crop images, semantic segmentation maps.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Randomly cropped results, 'img_shape' key in result dict is
#                 updated according to crop size.
#         """

#         images_ = results['img'].astype(np.float32)
#         images_ /= 255.0

#         if np.random.rand() < self.data_aug_ratio:
#             # flip image colors
#             images_ *= -1
#             # redefine background
#             images_[images_ == self.background_cl] = self.background

#             # normalize image
#             min_val = np.min(images_)
#             max_val = np.max(images_)
#             images_ -= min_val
#             if max_val - min_val != 0:
#                 images_ /= (max_val - min_val)

#             if self.random_contrast:
#                 c = np.round(np.random.uniform(self.gamma_min, self.gamma_max), 2)
#                 images_ **= c
            
#         results['img'] = np.clip(images_ * 255, 0, 255).astype(np.uint8)
        
#         return results

#     def __repr__(self):      
#         repr_str = self.__class__.__name__  
#         repr_str += (f'data_aug_ratio=({self.data_aug_ratio}, '
#                         f'gamma_min={self.gamma_min}, '
#                         f'gamma_max={self.gamma_max}, '
#                         f'background_cl={self.background_cl}, '
#                         f'background={self.background}, '
#                         f'random_contrast={self.random_contrast}')
        
#         return repr_str