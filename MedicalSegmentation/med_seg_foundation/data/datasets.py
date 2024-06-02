from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
from mmseg.datasets.pipelines import Compose
import mmcv
import h5py
from abc import abstractmethod
import lightning as L
from torch.utils.data import DataLoader
from data.transforms import *
import SimpleITK as sitk
from MedicalSegmentation.med_seg_foundation.data.distributed import VolDistributedSampler
from MedicalSegmentation.med_seg_foundation.utils.tools import get_file_list

# import torchvision.transforms as transforms
# from torchvision.transforms import functional as F
# from sklearn.model_selection import train_test_split
# import torch.nn.functional as F
# import torchvision.transforms.functional as F
# from model import Segmentor
# from scipy.ndimage import zoom
# import matplotlib.pyplot as plt


def put_in_res_dict(img, mask=None):
    # if pil img : PIL image ==> rgb order | if path do not touch
    pil_to_nd = lambda img: np.array(img).copy() if isinstance(img, Image.Image) else img
    
    img = pil_to_nd(img)
    result = dict(img=mmcv.imread(img, flag='color', channel_order='rgb'),
                  seg_fields=[])

    if mask is not None:
        mask = pil_to_nd(mask)
        key = 'gt_seg_map'
        result[key]=mmcv.imread(mask, flag='grayscale')
        result['seg_fields'].append(key)
            
    return result

def rm_from_res_dict(results):
    img = results['img']
    
    if 'gt_seg_map' in results.keys():
        mask = results['gt_seg_map']
        return [img, mask]
    return [img]


class SegDatasetRcsBase(Dataset):
    def __init__(self, 
                 num_classes:int,
                 rcs_enabled:bool=False,
                 dtype=torch.float32,
                 augmentations=None,
                 split_dataset_idx_vol=None,
                 after:bool=True) -> None:
        super().__init__()
        
        assert num_classes>0
        self.num_classes = num_classes
        
        # If to apply rcs or not 
        self.rcs_enabled = rcs_enabled
        
        # data ret type
        self.dtype = dtype
        
        # Put the must have transforms
        transforms = []
        transforms.append(lambda data : put_in_res_dict(data[0], data[1]))
        
        # Data transforms
        # Put the optional augmentations
        if augmentations is not None:
            assert isinstance(augmentations, list)
            transforms.extend(augmentations)
                
        # Put the rest of the mandatory transforms
        # conv the img keys to torch.Tensor with [HWC] -> [CHW]  |  mmseg/datasets/pipelines/transforms  
        transforms.append(dict(type='ImageToTensor', keys=['img', 'gt_seg_map']))
        
        # Remove the dict and keep the tensor
        transforms.append(rm_from_res_dict)
        
        self.transforms = Compose(transforms)
        
        # This section is used for self training
        self.ret_split_indicator_and_idxs = split_dataset_idx_vol is not None
        assert not (self.rcs_enabled and self.ret_split_indicator_and_idxs), "RCS and self training should not co-exist"
        if self.ret_split_indicator_and_idxs:
            self.split_dataset_idx_vol = split_dataset_idx_vol
            self.after = after
                
                
    def apply_transform(self, img, msk):
        [image, mask] = self.transforms([img, msk])
        image = image.to(self.dtype) # C, H, W
        mask = mask.squeeze(0).to(torch.int64) #.reshape(image.shape[1:]) # H, W
        return image, mask
        
        
    def get_rare_class_idx(self):
        # Choose a random class following the RCS probability distributtion
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        # Choose a random sample (uniform dist) index correspomding to that class
        idx = np.random.choice(self.file_indexes_containing_classes[c])
        return idx   
        
    @abstractmethod
    def get_mask(self, idx):
        pass    
        
    def init_rcs(self):
        assert self.rcs_enabled, 'RCS flag should be enabled !'
        # An image should contain at least this many pixels from a class to be considered as it has that class 
        self.rcs_min_pixels = 4
        
        # smoothing of RCS proba dist - higher T = more uniform distribution, lower T = stronger focus on rare classes 
        self.rcs_class_temp = 0.1
        #############################################################
        
        # array of contained classes e.g. [0, 1, 2, ...]
        self.rcs_classes = list(range(self.num_classes))
        
        # list storing sample idxs containing each class
        self.file_indexes_containing_classes = [list()]*self.num_classes 
        
        # Relative frequency of classes (i.e. number of pixels with class c)
        self.class_freqs = np.zeros(self.num_classes, dtype=np.float32)
            
        for idx in range(self.__len__()):
            mask = self.get_mask(idx) 
            assert len(mask.shape)==2, f'Expected a 2D mask but got one of shape {mask.shape}'
            
            # Get the unique classes and their pixel counts
            unique_classes, counts = np.unique(mask, return_counts=True)  # Read are the img indices (single channel)
            assert unique_classes.size <= self.num_classes, \
                f"Expected {self.num_classes} classes but got {unique_classes}, index: {idx}/{self.__len__()}"
            assert np.max(unique_classes) < self.num_classes, \
                f"Expected {self.num_classes} classes but got {unique_classes}, index: {idx}/{self.__len__()}"
            
            # Relative frequency (per mask)
            freqs_per_im = counts / (mask.shape[0]*mask.shape[1])
            
            # Save file index to classes 
            for class_i, count_i in zip(unique_classes, counts):
                if count_i>self.rcs_min_pixels:
                    self.file_indexes_containing_classes[class_i].append(idx)
            
            for i, class_i in enumerate(unique_classes):
                assert freqs_per_im[i]<=1. and freqs_per_im[i]>=0., f'Class{i} freq is invlaid {freqs_per_im[i]}'
                self.class_freqs[class_i] += freqs_per_im[i]
            
        # Mean over the dataset
        self.class_freqs /= self.__len__()
        
        # proba of the associated classes computed via RCS
        freq = torch.tensor(self.class_freqs, dtype=torch.float32)
        freq = freq / torch.sum(freq)  
        freq = 1 - freq
        self.rcs_classprob = torch.softmax(freq / self.rcs_class_temp, dim=-1).cpu().numpy()
        
        
    @abstractmethod    
    def _get_n_vol_n_slice_idxs(self, idx):
        pass
    
    @abstractmethod
    def _getitem1(self, idx):
        pass
    
    def get_dataset_split_indicator(self, vol_idx):
        assert self.ret_split_indicator_and_idxs
        if self.after:
            return vol_idx>=self.split_dataset_idx_vol
        else:
            return vol_idx<self.split_dataset_idx_vol
    
    def __getitem__(self, idx):
        res = self._getitem1(idx=idx)
        
        if self.ret_split_indicator_and_idxs:
            vol_idx, slice_idx = self._get_n_vol_n_slice_idxs(idx)
            idc = self.get_dataset_split_indicator(vol_idx)
            return *res, idc, vol_idx, slice_idx
        else:
            return res


class SegmentationDataset(SegDatasetRcsBase):
    def __init__(self, img_dir, mask_dir, num_classes, 
                 file_extension=None, mask_suffix='',
                 augmentations=None,
                 dtype=torch.float32,
                 split_dataset_idx_vol=None,
                 after:bool=True,
                 ret_n_z:bool=True,
                 nz=None, rcs_enabled:bool=False,
                 preload:bool=True,
                 ):
        super().__init__(num_classes=num_classes, rcs_enabled=rcs_enabled, augmentations=augmentations, dtype=dtype,
                         split_dataset_idx_vol=split_dataset_idx_vol, after=after)
        self.preload=preload
        
        if isinstance(img_dir, Path):
            img_dir = str(img_dir)
            
        if isinstance(mask_dir, Path):
            mask_dir = str(mask_dir)
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        
        if nz is not None:
            assert nz>0
            assert isinstance(nz, int)
        self.nz = nz
        
        if self.ret_split_indicator_and_idxs:
            assert self.nz is not None, "nz must be provided for volume index spliting"
        
        self.ret_n_z = ret_n_z
        if self.ret_n_z:
            assert self.nz is not None, 'Need to provide the constant vol depth nz'
        
        
        # Only include images for which a mask is found
        # self.images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
        self.image_f_names = [img for img in get_file_list(img_dir, extension=file_extension) \
            if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + mask_suffix +'.'+ img.split(".")[-1]))]
        
        self.nb_imgs = len(self.image_f_names)

        if self.preload:
            self.images = []
            self.masks = []
            for i in range(self.nb_imgs):
                image, mask = self.load_img_and_mask(idx=i)
                self.images.append(image)
                self.masks.append(mask)
                
        if self.rcs_enabled:
            self.init_rcs()
            
    def _get_n_vol_n_slice_idxs(self, idx):
        assert self.nz is not None 
        assert idx>=0 
        assert idx<self.__len__()
        
        vol_idx = idx%self.nz
        slice_idx = idx - vol_idx*self.nz
        return vol_idx, slice_idx
        
        
    def load_img_and_mask(self, idx):
        assert idx>=0 and idx<self.nb_imgs
        img_path = os.path.join(self.img_dir, self.image_f_names[idx])
        mask_path = os.path.join(self.mask_dir, 
                                 self.get_mask_f_name(img_idx=idx))
        
        image = Image.open(img_path).convert("RGB") # H, W, C
        mask = Image.open(mask_path)  # Read are the img indices (single channel)
        return image, mask
            
    def get_mask_f_name(self, img_idx):
        assert img_idx>=0 and img_idx<self.nb_imgs
        return self.image_f_names[img_idx].split(".")[0] + self.mask_suffix +'.'+ self.image_f_names[img_idx].split(".")[-1]
        
    
    def get_mask(self, idx):
        if not self.preload:
            mask_path = os.path.join(self.mask_dir, 
                                    self.get_mask_f_name(img_idx=idx))
            with Image.open(mask_path) as img:
                np_img = np.array(img, copy=True)
            return np_img
        else:
            return np.array(self.masks[idx])
        
    def __len__(self):
        return self.nb_imgs

    def _getitem1(self, idx):
        if self.rcs_enabled:
            idx = self.get_rare_class_idx()
        
        if not self.preload:
            image, mask = self.load_img_and_mask(idx=idx)
        else:
            image = self.images[idx]
            mask = self.masks[idx]
        
        image, mask = self.apply_transform(img=image, msk=mask)
        
        if not self.ret_n_z:
            return image, mask     
        else:
            n_z = dict(nz = self.nz,
                       last_slice = (idx+1)%self.nz==0)
        
            return image, mask, n_z
        
    def __del__(self):
        if self.preload:
            for im, mask in zip(self.images, self.masks):
                im.close()
                mask.close()
                

#################################################################################################

class Segmentation3Dto2Dbase(SegDatasetRcsBase):
    def __init__(self, 
                 num_classes: int, 
                 rcs_enabled: bool = False, 
                 dtype=torch.float32, 
                 augmentations=None,
                 split_dataset_idx_vol=None,
                 after:bool=True,) -> None:
        super().__init__(num_classes=num_classes, rcs_enabled=rcs_enabled, dtype=dtype, augmentations=augmentations,
                         split_dataset_idx_vol=split_dataset_idx_vol, after=after)
        
        
    def _get_n_vol_n_slice_idxs(self, idx):
        assert hasattr(self, 'nb_slices_until')
        
        vol_idx = np.searchsorted(self.nb_slices_until, idx, side='right')
        
        if vol_idx>0:
            nb_slices_before = self.nb_slices_until[vol_idx-1]
        else:
            nb_slices_before = 0
            
        slice_idx = idx - nb_slices_before
        
        assert slice_idx >= 0, f'Should be positive but got {slice_idx}'
        assert slice_idx < self.nb_slices_until[vol_idx]-nb_slices_before
        assert nb_slices_before + slice_idx == idx, f'Mismatch, expected: {idx}, calculated {nb_slices_before+slice_idx}'
        
        return vol_idx, slice_idx 
    
    def post_process(self, image, mask):
        
        # [0, 1] to [0, 255]
        if image.max()<=1 and image.min()>=0:
            image = image*255.
        
        assert image.max()<=255 and image.min()>=0
        assert mask.max()<=self.num_classes
        
        # Convert grayscale to RGB
        if len(image.shape)<3:
            image = np.stack([image, image, image], axis=-1)
            
        assert len(mask.shape) == 2, f'mask shape = {mask.shape}'
        assert len(image.shape) == 3, f'image shape = {image.shape}'
        assert image.shape[:2] == mask.shape, f'image shape={image.shape}, maks shape = {mask.shape}'
        
        image, mask = self.apply_transform(img=image, msk=mask)
        
        return image, mask

#################################################################################################

class SegmentationDatasetHDF5(Segmentation3Dto2Dbase):
    def __init__(self, file_pth, num_classes, 
                 augmentations=None,
                 dtype=torch.float32,
                 ret_n_xyz:bool=True,
                 rcs_enabled:bool=False,
                 split_dataset_idx_vol=None,
                 after:bool=True,):
        super().__init__(num_classes=num_classes, rcs_enabled=rcs_enabled, augmentations=augmentations, dtype=dtype,
                         split_dataset_idx_vol=split_dataset_idx_vol, after=after)
        
        self.preload = True
        assert self.preload, "Not preloading is not yet implemented !"
        
        if isinstance(file_pth, Path):
            file_pth = str(file_pth)
        
        self.file_pth = file_pth
        self.ret_n_xyz = ret_n_xyz
        
        # Dataset
        self.dataset = h5py.File(self.file_pth, 'r')
        
        # Get dataset properties
        self.nb_vol = self.dataset['nz'].shape[0]
        self.nb_slices_until = np.cumsum(self.dataset['nz'][:], dtype=int)
        self.nb_slice_tot = self.dataset['nz'][:].sum().astype('int32')
        assert self.nb_slice_tot == self.nb_slices_until[-1]
        
        if self.rcs_enabled:
            self.init_rcs()        
    
    def get_mask(self, idx):
        
        if len(self.dataset['labels'].shape) == 4:
            vol_idx, slice_idx = self._get_n_vol_n_slice_idxs(idx)
            mask = self.dataset['labels'][vol_idx, slice_idx].copy().astype('uint8')  
                        
        elif len(self.dataset['labels'].shape) == 3:
            mask = self.dataset['labels'][idx].copy().astype('uint8')                    
        else:
            ValueError(f'Unsupported dataset images shape')
        return mask

    def __len__(self):
        return self.nb_slice_tot
    

    def _getitem1(self, idx):
        
        if self.rcs_enabled:
            idx = self.get_rare_class_idx()
        
        if self.dataset is None:
            self.dataset = h5py.File(self.file_pth, 'r')
            
        vol_idx, slice_idx = self._get_n_vol_n_slice_idxs(idx)
        
        if self.ret_n_xyz:
            n_xyz = dict(nx = self.dataset['nx'][vol_idx].copy().astype('int32'),
                         ny = self.dataset['ny'][vol_idx].copy().astype('int32'),
                         nz = self.dataset['nz'][vol_idx].copy().astype('int32'),
                         last_slice = slice_idx==self.dataset['nz'][vol_idx]-1)
        
        assert self.dataset['images'].shape == self.dataset['labels'].shape, \
            f'Image and labels shape mismatch, {self.dataset["images"].shape} and {self.dataset["labels"].shape}'
        
        if len(self.dataset['images'].shape) == 4:
            image = self.dataset['images'][vol_idx, slice_idx].copy().astype('float32')
            mask = self.dataset['labels'][vol_idx, slice_idx].copy().astype('uint8')
            
        elif len(self.dataset['images'].shape) == 3:
            image = self.dataset['images'][idx].copy().astype('float32')
            mask = self.dataset['labels'][idx].copy().astype('uint8')
            
        else:
            raise ValueError(f'Unsupported dataset images shape')
        
        image, mask = self.post_process(image=image, mask=mask)
        
        if not self.ret_n_xyz:            
            return image, mask
        else:
            return image, mask, n_xyz
        
    def __del__(self):
        self.dataset.close()    
    

        
 #################################################################################################
 
 
class SegmentationDatasetNIFIT(Segmentation3Dto2Dbase):
    def __init__(self, 
                 directory:Union[str, Path],
                 img_suffix:str,
                 lab_suffix:str,
                 num_classes: int, 
                 rcs_enabled: bool = False, 
                 dtype=torch.float32, 
                 augmentations=None,
                 preload=False,
                 ret_nz=True,
                 split_dataset_idx_vol=None,
                 after:bool=True,) -> None:
        super().__init__(num_classes=num_classes, rcs_enabled=rcs_enabled, dtype=dtype, augmentations=augmentations,
                         split_dataset_idx_vol=split_dataset_idx_vol, after=after)
        
        self.preload=preload
        self.extension = '.nii.gz'
        self.ret_nz = ret_nz
        
        # data directory
        if isinstance(directory, Path):
            directory = str(directory)
        self.directory = directory
        
        # image(volume) file suffix
        assert img_suffix in ["FLAIR", "T1"]
        self.img_suffix = img_suffix  # _FLAIR, _T1
        
        # Label file suffix
        assert lab_suffix == "Label"
        self.lab_suffix = lab_suffix  # _Label
        
        # Image files
        self.img_files = get_file_list(self.directory, extension=self.extension, file_suffix=self.img_suffix)
        
        # Label files
        self.lab_files = get_file_list(self.directory, extension=self.extension, file_suffix=self.lab_suffix)
        
        # Verifications
        assert len(self.img_files) == len(self.lab_files)
        self.nb_vols = len(self.img_files)
        
        self.nb_slice_per_vol = []
        for i in range(self.nb_vols):
            assert os.path.isfile(os.path.join(self.directory, 
                                               self.img_files[i].split(self.img_suffix)[0]+self.lab_suffix+self.extension))
            # Save the nb slices per vol
            lab_pth = os.path.join(self.directory, self.lab_files[i])
            self.nb_slice_per_vol.append(sitk.GetArrayFromImage(sitk.ReadImage(lab_pth)).shape[0])

        self.nb_slice_tot = sum(self.nb_slice_per_vol)
        self.nb_slices_until = np.cumsum(self.nb_slice_per_vol)
        
        if self.preload:
            self.imgs = []
            self.labels = []
            for i in range(self.nb_vols):
                img_pth = os.path.join(self.directory, self.img_files[i])
                lab_pth = os.path.join(self.directory, self.lab_files[i])
                self.imgs.append(sitk.GetArrayFromImage(sitk.ReadImage(img_pth)).astype('float32')) 
                self.labels.append(sitk.GetArrayFromImage(sitk.ReadImage(lab_pth)).astype('uint8')) 
                                
                if self.imgs[i].max()<=1 and self.imgs[i].min()>=0:
                    self.imgs[i] = self.imgs[i]*255.
        
                assert self.imgs[i].max()<=255 and self.imgs[i].min()>=0
                assert self.labels[i].max()<=self.num_classes
                
        if self.rcs_enabled:
            self.init_rcs()
                           
        
    def get_mask(self, idx):
        vol_idx, slice_idx = self._get_n_vol_n_slice_idxs(idx=idx)
        
        if self.preload:
            assert hasattr(self, 'labels')
            return self.labels[vol_idx][slice_idx]
        else:
            return sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.directory, self.lab_files[vol_idx])))\
                .astype('uint8')[slice_idx]
    
    def __len__(self):
        return self.nb_slice_tot
    
    def _getitem1(self, idx):
        if self.rcs_enabled:
            idx = self.get_rare_class_idx()
        
        vol_idx, slice_idx = self._get_n_vol_n_slice_idxs(idx=idx)
        
        if self.preload:
            image = self.imgs[vol_idx][slice_idx]
            label = self.labels[vol_idx][slice_idx]
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.directory, self.img_files[vol_idx])))\
                .astype('float32')[slice_idx]
            label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.directory, self.lab_files[vol_idx])))\
                .astype('uint8')[slice_idx]
                    
        if self.ret_nz:
            nz = dict(nz = self.nb_slice_per_vol[vol_idx],
                      last_slice = slice_idx==self.nb_slice_per_vol[vol_idx]-1)
            
        image, label = self.post_process(image=image, mask=label)

        if self.ret_nz:
            return image, label, nz
        else:
            return image, label
        
 #################################################################################################

        
class VolDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, train_dataloader_cfg,
                 vol_depth, num_gpus,
                 val_dataset=None, test_dataset=None, 
                 val_dataloader_cfg=None, test_dataloader_cfg=None,
                 ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.train_dataloader_cfg = train_dataloader_cfg
        self.val_dataloader_cfg = val_dataloader_cfg
        self.test_dataloader_cfg = test_dataloader_cfg
        
        self.vol_depth = vol_depth
        self.num_gpus = num_gpus

    def train_dataloader(self):
        # Use default behavior for training dataset
        return DataLoader(dataset=self.train_dataset, **self.train_dataloader_cfg)

    def val_dataloader(self):
        if self.num_gpus>1:
            # Use custom sampler for validation dataset
            sampler = VolDistributedSampler(dataset=self.val_dataset, num_replicas=self.num_gpus, 
                                            vol_depth=self.vol_depth, shuffle=False, drop_last=False,)
        else:
            sampler=None
            
        return DataLoader(self.val_dataset, sampler=sampler, **self.val_dataloader_cfg)   
    
    def test_dataloader(self):
        if self.num_gpus>1:
            # Use custom sampler for validation dataset
            sampler = VolDistributedSampler(dataset=self.test_dataset, num_replicas=self.num_gpus, 
                                            vol_depth=self.vol_depth, shuffle=False, drop_last=False,)
        else:
            sampler=None
            
        return DataLoader(self.test_dataset, sampler=sampler, **self.val_dataloader_cfg)  