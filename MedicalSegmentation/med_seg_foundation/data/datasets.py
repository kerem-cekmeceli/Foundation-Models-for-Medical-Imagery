# from OrigDino.dinov2.models.vision_transformer import *


from PIL import Image
import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# from torchvision.transforms import functional as F
import numpy as np
import os
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
# import torch.nn.functional as F
# import torchvision.transforms.functional as F
# from model import Segmentor
# from scipy.ndimage import zoom
# import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from mmseg.datasets.pipelines import Compose
import mmcv
from data.transforms import *
import h5py
import lightning as L
from abc import abstractmethod


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file_list(fld_pth, start_idx=0, num_img=None, extension=None, 
                  file_suffix=None, inc_exc=0):
    assert inc_exc in [0, 1], "0 to include 1 to exlude"

    # List all files in the folder
    files = os.listdir(fld_pth)

    # Filter out non-files (directories, subdirectories, etc.)
    files = [file for file in files if os.path.isfile(os.path.join(fld_pth, file))]
    
    # Filter for file type
    if extension is not None:
        files = [f for f in files if f.endswith(extension)]
        
    # Filter for our dataset
    if file_suffix is not None:
        if inc_exc == 0:
            # Include
            files = [f for f in files if os.path.splitext(f)[0].endswith(file_suffix)]
        else:
            # Exlude
            files = [f for f in files if not os.path.splitext(f)[0].endswith(file_suffix)]

    # Sort by name
    files.sort()
    
    if num_img is not None:
        num_img = num_img if num_img>len(files) else len(files)
    else:
        num_img = len(files)
    imgs_sel = files[start_idx:start_idx+num_img]
    return imgs_sel 

def put_in_res_dict(img, mask=None):
    # if pil img : Convert PIL img (RGB) --> ndarray (BGR)
    get_nd_arr = lambda img: np.array(img).copy() if isinstance(img, Image.Image) else img
    # if ndarray convert RGB to BGR
    rgb_2_bgr = lambda img: img[..., ::-1].copy() if isinstance(img, np.ndarray) else img
    
    img = rgb_2_bgr(get_nd_arr(img))
    result = dict(img=mmcv.imread(img, flag='color', channel_order='bgr'),
                  seg_fields=[])

    if mask is not None:
        mask = get_nd_arr(mask)
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
                 rcs_enabled:bool=False) -> None:
        super().__init__()
        
        assert num_classes>0
        self.num_classes = num_classes
        
        # If to apply rcs or not 
        self.rcs_enabled = rcs_enabled
        
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


class SegmentationDataset(SegDatasetRcsBase):
    def __init__(self, img_dir, mask_dir, num_classes, 
                 file_extension=None, mask_suffix='',
                 augmentations=None, images=None,
                 dtype=torch.float32,
                 ret_n_z:bool=True,
                 nz=None, rcs_enabled:bool=False):
        super().__init__(num_classes=num_classes, rcs_enabled=rcs_enabled)
        
        if isinstance(img_dir, Path):
            img_dir = str(img_dir)
            
        if isinstance(mask_dir, Path):
            mask_dir = str(mask_dir)
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.dtype = dtype
        
        if nz is not None:
            assert nz>0
            assert isinstance(nz, int)
        self.nz = nz
        
        self.ret_n_z = ret_n_z
        
        if self.ret_n_z:
            assert self.nz is not None, 'Need to provide the constant vol depth nz'
        
        # Put the must have transforms
        transforms = []
        transforms.append(lambda data : put_in_res_dict(data[0], data[1]))
        
        # Put the optional augmentations
        if augmentations is not None:
            transforms.extend(augmentations)
        
        # Put the rest of the mandatory transforms
        # conv the img keys to torch.Tensor with [HWC] -> [CHW]  |  mmseg/datasets/pipelines/transforms  
        transforms.append(dict(type='ImageToTensor', keys=['img', 'gt_seg_map']))
        
        # Remove the dict and keep the tensor
        transforms.append(rm_from_res_dict)
        
        # Compose the transforms
        self.transforms = Compose(transforms)
        
        # Only include images for which a mask is found
        if images is None:
            # self.images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
            self.images = [img for img in get_file_list(img_dir, extension=file_extension) \
                if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + mask_suffix +'.'+ img.split(".")[-1]))]
        else:
            self.images = images
            
        if self.rcs_enabled:
            self.init_rcs()
    
    def get_mask(self, idx):
        mask_path = os.path.join(self.mask_dir, 
                                 self.images[idx].split(".")[0] + self.mask_suffix +'.'+ self.images[idx].split(".")[-1])
        return np.array(Image.open(mask_path))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.rcs_enabled:
            idx = self.get_rare_class_idx()
        
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, 
                                 self.images[idx].split(".")[0] + self.mask_suffix +'.'+ self.images[idx].split(".")[-1])
        
        image = Image.open(img_path).convert("RGB") # H, W, C
        mask = Image.open(mask_path)  # Read are the img indices (single channel)
        
        [image, mask] = self.transforms([image, mask])
        image = image.to(self.dtype) # C, H, W
        mask = mask.squeeze(0).to(torch.int64) #.reshape(image.shape[1:]) # H, W
        
        if not self.ret_n_z:
            return image, mask
        else:
            n_z = dict(nz = self.nz,
                       last_slice = (idx+1)%self.nz==0)
        
            return image, mask, n_z
        

#################################################################################################

class SegmentationDatasetHDF5(SegDatasetRcsBase):
    def __init__(self, file_pth, num_classes, 
                 augmentations=None,
                 dtype=torch.float32,
                 ret_n_xyz:bool=True,
                 rcs_enabled:bool=False):
        super().__init__(num_classes=num_classes, rcs_enabled=rcs_enabled)
        
        if isinstance(file_pth, Path):
            file_pth = str(file_pth)
        
        self.file_pth = file_pth
        self.dtype = dtype
        self.ret_n_xyz = ret_n_xyz
        
        # Put the must have transforms
        transforms = []
        transforms.append(lambda data : put_in_res_dict(data[0], data[1]))
        
        # Put the optional augmentations
        if augmentations is not None:
            transforms.extend(augmentations)
        
        # Put the rest of the mandatory transforms
        # conv the img keys to torch.Tensor with [HWC] -> [CHW]  |  mmseg/datasets/pipelines/transforms  
        transforms.append(dict(type='ImageToTensor', keys=['img', 'gt_seg_map']))
        
        # Remove the dict and keep the tensor
        transforms.append(rm_from_res_dict)
        
        # Compose the transforms
        self.transforms = Compose(transforms)
        
        # Dataset
        self.dataset = None
        with h5py.File(self.file_pth, 'r') as f:
            self.nb_vol = f['nz'].shape[0]
            self.nb_slices_until = np.cumsum(f['nz'][:], dtype=int)
            self.nb_slice_tot = f['nz'][:].sum().astype('int32')
            assert self.nb_slice_tot == self.nb_slices_until[-1]
            
        if self.rcs_enabled:
            self.init_rcs()
    
    def get_mask(self, idx):
        with h5py.File(self.file_pth, 'r') as f:
            if len(f['labels'].shape) == 4:
                vol_idx, slice_idx = self._get_nb_vol_n_slice_idxs(idx)
                mask = f['labels'][vol_idx, slice_idx].copy().astype('uint8')  
                            
            elif len(f['labels'].shape) == 3:
                mask = f['labels'][idx].copy().astype('uint8')                    
            else:
                ValueError(f'Unsupported dataset images shape')
        return mask

    def __len__(self):
        return self.nb_slice_tot
    
    def _get_nb_vol_n_slice_idxs(self, idx):
        
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

    def __getitem__(self, idx):
        
        if self.rcs_enabled:
            idx = self.get_rare_class_idx()
        
        if self.dataset is None:
            self.dataset = h5py.File(self.file_pth, 'r')
            
        vol_idx, slice_idx = self._get_nb_vol_n_slice_idxs(idx)
        
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
            ValueError(f'Unsupported dataset images shape')
        
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
        
        [image, mask] = self.transforms([image, mask])
        image = image.to(self.dtype) # C, H, W
        mask = mask.squeeze(0).to(torch.int64) 
        
        if not self.ret_n_xyz:            
            return image, mask
        else:
            return image, mask, n_xyz
    
from torch.utils.data import DistributedSampler    

class VolDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset,  vol_depth:int, num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, shuffle: bool = True, 
                 seed: int = 0, drop_last: bool = False) -> None:
        
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        assert vol_depth>0
        self.vol_depth = vol_depth
        
        assert len(self.dataset)> 0
        
        assert len(self.dataset)%self.vol_depth == 0, "Dataset can not contain a partial volume"
        self.total_nb_vols_dataset = len(self.dataset)//self.vol_depth
        assert self.total_nb_vols_dataset > 0
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.total_nb_vols_dataset % self.num_replicas != 0 and \
            self.total_nb_vols_dataset>self.num_replicas:  
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_vols = math.floor(self.total_nb_vols_dataset / self.num_replicas)

        else:
            # Num volumes per rank
            self.num_vols = math.ceil(self.total_nb_vols_dataset / self.num_replicas)
            
        assert self.num_vols>0, "Each GPU must get at least 1 volume"
                        
        # Num samples per rank
        self.num_samples = self.num_vols * self.vol_depth
        
        # Total number of volumes (inc replication)
        self.total_nb_vols = self.num_vols * self.num_replicas
        self.total_size = self.total_nb_vols * self.vol_depth
        
    def __iter__(self):
        if self.shuffle:
            # # deterministically shuffle based on epoch and seed
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            ValueError('Not Implemented yet') # shuffle assigned volumes and slices in the vol but keep complete volumes for all GPUs
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            
            if padding_size>0:
                assert padding_size%self.vol_depth == 0
            
            if padding_size <= len(indices):
                indices += indices[:padding_size]  # append from the first volumes 
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        start_idx = self.rank*self.num_samples
        end_idx = start_idx + self.num_samples
        indices = indices[start_idx:end_idx:1]
        assert len(indices) == self.num_samples

        return iter(indices)


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
  
  
        

#################################################################################################

def color_map_from_imgs(fld_pth, file_ls,  device='cuda:0'):
    img_colors_all = None
    for img in file_ls:
        img = np.array(Image.open(fld_pth+'/'+img).convert('RGB'))
        img = torch.from_numpy(img).to(device)
        img_colors = img.reshape([-1, img.shape[-1]])
        # img_colors = torch.unique(img_colors, axis=0)  # Keep the unique colors only
        
        if img_colors_all is None:
            img_colors_all = img_colors
        else:
            img_colors_all = torch.cat([img_colors_all, img_colors], axis=0)
            # img_colors_all =  torch.unique(img_colors_all, axis=0)  # Keep the unique colors only
    img_colors_all = torch.unique(img_colors_all, dim=0)
    color_mapping = {}
    for i, c in enumerate(img_colors_all.cpu().numpy()):
        color_mapping[tuple(c)] = i
        
    return color_mapping  # {(R, G, B) : idx}

# def map_rgb_to_indices(rgb_tensor, color_map):
#     # Convert the color map to a PyTorch tensor
#     color_map_tensor = torch.tensor(list(color_map.keys())).float()

#     # Expand dimensions of input tensor to (N, 1, 1, C) for broadcasting
#     rgb_tensor_expanded = rgb_tensor.unsqueeze(1).unsqueeze(1)

#     # Check if each RGB value in the tensor exactly matches a color in the color map
#     exact_matches = torch.all(rgb_tensor_expanded == color_map_tensor.view(1, -1, 1, 3), dim=-1)

#     # Find the index of the matching color for each RGB value
#     indices = torch.argmax(exact_matches, dim=1)

#     # Map indices to corresponding labels using the color map
#     labels = torch.tensor(list(color_map.values()))[indices]

#     # Check if any RGB values do not have an exact match in the color map
#     if not torch.all(exact_matches.any(dim=1)):
#         raise ValueError("Some RGB values do not have an exact match in the color map.")

#     return labels


# def get_masks_from_imgs(fld_pth, file_ls):
#     for img in file_ls:
#         img = np.array(Image.open(fld_pth+'/'+img).convert('RGB'))
#         img = torch.from_numpy(img).to(device)
        
# def create_segmentation_mask(image_tensor, color_map):
#     # Flatten the image tensor to (batch_size, height * width, channels)
#     flattened_image = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(-1))

#     # Expand the color map to (1, 1, 1, num_colors, 3)
#     color_map_tensor = torch.tensor(list(color_map.keys())).view(1, 1, 1, -1, 3).float()

#     # Expand the image tensor to (batch_size, height * width, 1, channels)
#     image_tensor_expanded = flattened_image.unsqueeze(2)

#     # Compute the L2 distance between each pixel and each color in the color map
#     distances = torch.norm(image_tensor_expanded - color_map_tensor, dim=-1)

#     # Find the index of the minimum distance for each pixel
#     indices = torch.argmin(distances, dim=-1)

#     # Map the indices to the corresponding labels using the color map
#     mask_tensor = torch.tensor(list(color_map.values()))[indices]

#     # Convert the flattened mask tensor back to the original shape
#     mask_tensor = mask_tensor.view(image_tensor.size(0), image_tensor.size(1), image_tensor.size(2))

#     return mask_tensor