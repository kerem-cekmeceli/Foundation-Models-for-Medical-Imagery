# from OrigDino.dinov2.models.vision_transformer import *


from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
# import torch.nn.functional as F
import torchvision.transforms.functional as F
# from model import Segmentor
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, 
                 file_extension=None, mask_suffix='',
                 img_transform=None, mask_transform=None, images=None,
                 dtype=torch.float32,
                 *args, **kwargs):
        super(Dataset).__init__(*args, **kwargs)
        
        if isinstance(img_dir, Path):
            img_dir = str(img_dir)
            
        if isinstance(mask_dir, Path):
            mask_dir = str(mask_dir)
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.mask_suffix = mask_suffix
        self.dtype = dtype
        
        if img_transform is None:
            img_transform = F.pil_to_tensor 
            
        if mask_transform is None:
            mask_transform = lambda mask : torch.as_tensor(np.array(mask), dtype=torch.int64)
            
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        # Only include images for which a mask is found
        if images is None:
            # self.images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
            self.images = [img for img in get_file_list(img_dir, extension=file_extension) \
                if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + mask_suffix +'.'+ img.split(".")[-1]))]
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, 
                                 self.images[idx].split(".")[0] + self.mask_suffix +'.'+ self.images[idx].split(".")[-1])
        
        image = Image.open(img_path).convert("RGB") # H, W, C
        mask = Image.open(mask_path)  # Read are the img indices (single channel)

        image = self.img_transform(image).to(self.dtype) # C, H, W
        mask = self.mask_transform(mask).to(torch.int64).reshape(image.shape[1:]) # H, W
                    
        # Create a tensor to hold the binary masks (GT probas of each class)  [num_cls, H, W]
        bin_mask = torch.zeros(self.num_classes, mask.shape[0], mask.shape[1], dtype=self.dtype)
        
        for i in range(self.num_classes):
            bin_mask[i] = (mask == i).to(self.dtype)  # Ensure resulting mask is float type
            
        # Bin_mask = gt_proba of all classes [num_cls, H, W]
        return image, bin_mask
    

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