import sys
from pathlib import Path
 
med_seg_path = Path(__file__).parent.parent.parent.parent
main_pth = med_seg_path.parent
med_seg_mod_pth = med_seg_path / 'med_seg_foundation'

orig_models_pth = main_pth / 'OrigModels' 

sys.path.insert(0, str(main_pth))
sys.path.insert(1, str(med_seg_mod_pth))


import SimpleITK as sitk
import numpy as np
# from torch.nn import functional as F
from skimage.transform import resize
from MedicalSegmentation.med_seg_foundation.data.datasets import get_file_list
import os
from tqdm import tqdm
import socket

"""
    @TODO
    1) N4 bias corr
    2) 2D interpolation
    3) Z norm = (x-mean)/std (over the volume)
    # 4) Threshold to [2%, 98%] percentile (over the volume)
    4) min/max Norm [0, 1] (over the volume)
"""

# self.imgs[i] = (self.imgs[i]-self.imgs[i].min()) / self.imgs[i].max()

def process_img(img_pth, src_dir, save_dir):
    # Load the image
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(src_dir, img_pth))).astype('float32')
    
    # Z-norm
    # mean = np.mean(img)
    # std = np.std(img)
    # corrected_img = (img-mean)/std
    corrected_img = img
    
    # Min-Max Norm
    corrected_img = (corrected_img-np.min(corrected_img)) / np.max(corrected_img)
    
    # Clip the values
    corrected_img = np.clip(corrected_img, a_min=0, a_max=1)
    
    # # Percentile [2% 98%] clipping
    # percentile2 = np.percentile(corrected_img, 2)
    # percentile98 = np.percentile(corrected_img, 98)
    # corrected_img = np.clip(corrected_img, a_min=percentile2, a_max=percentile98)
    
    # Ordering to ZXY
    corrected_img = np.transpose(corrected_img, (1, 0, 2))
    
    # Bilinear interpolation
    slices = []
    for z in range(corrected_img.shape[0]):
        resized = resize(corrected_img[z], 
                         output_shape=(224, 224), 
                         order=1, 
                         preserve_range=True, 
                         mode='constant')
        slices.append(resized)
    
    # Save image
    corrected_img = np.stack(slices, axis=0, dtype=np.float32)
    assert np.max(corrected_img) <= 1
    assert np.min(corrected_img) >= 0
    corrected_img = sitk.GetImageFromArray(corrected_img)
    sitk.WriteImage(corrected_img, os.path.join(save_dir, img_pth))
    
    
def process_label(lab_pth, src_dir, save_dir):
    # Load the label
    lab = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(src_dir, lab_pth))).astype('uint8')
    
    # Ordering to ZXY
    lab = np.transpose(lab, (1, 0, 2))
    
    # Order 0 interpolate
    labels = []
    for z in range(lab.shape[0]):
        lab_slice_resized = resize(lab[z], 
                             output_shape=(224, 224), 
                             order=0, 
                             preserve_range=True, 
                             mode='constant')
        labels.append(lab_slice_resized)
        
    # Save image
    labels = np.stack(labels, axis=0, dtype=np.uint8)
    labels = sitk.GetImageFromArray(labels)
    sitk.WriteImage(labels, os.path.join(save_dir, lab_pth))
    
    
def n4_bias_corr(img_pth, src_dir, save_dir):
    # Load the image
    img = sitk.ReadImage(os.path.join(src_dir, img_pth))
    
    # N4 bias correction
    msk_thres = -2.8
    img_mask=sitk.BinaryNot(sitk.BinaryThreshold(img, msk_thres, msk_thres))
    corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
    
    # Save image
    sitk.WriteImage(corrected_img, os.path.join(save_dir, img_pth))
    
    
    
def process_dataset(src_data_pth, target_data_pth_n4, target_data_pth_norm):
    img_pths = get_file_list(fld_pth=src_data_pth, extension='.nii.gz', file_suffix="Label", inc_exc=1)
    lab_pths = get_file_list(fld_pth=src_data_pth, extension='.nii.gz', file_suffix="Label", inc_exc=0)
    
    # Do N4 bias correction
    # tqdm.write(f"N4 bias correction, src: {src_data_pth}")
    # for img_pth in tqdm(img_pths):
    #     n4_bias_corr(img_pth=img_pth, src_dir=src_data_pth, save_dir=target_data_pth_n4)
    
    # Normalize and resize images
    tqdm.write(f"Image normalization and resizing, src: {src_data_pth}")
    for img_pth in tqdm(img_pths):
        process_img(img_pth=img_pth, src_dir=src_data_pth, save_dir=target_data_pth_norm)
    
    # Resize labels
    tqdm.write(f"Label resizing, src: {src_data_pth}")
    for lab_pth in tqdm(lab_pths):
        process_label(lab_pth=lab_pth, src_dir=src_data_pth, save_dir=target_data_pth_norm)
         


cluster = 'KeremPC' != socket.gethostname()
dir_names = ["brats_val", "brats_test", "brats_train"] if cluster else ["brats_val"]

if cluster:
    main_pth = "/usr/bmicnas02/data-biwi-01/foundation_models/da_data" 
else:
    main_pth = "/home/kerem_ubuntu/Projects/DataFoundationModels"
    
for dir_name in tqdm(dir_names):
    src_data_pth = f"{main_pth}/brain/BraTS/{dir_name}/"
    target_data_pth_n4 = f"{main_pth}/brain/BraTS/{dir_name}_n4/"
    target_data_pth_norm = f"{main_pth}/brain/BraTS/{dir_name}_processed/"
    
    # Create directories if missing
    # Path(target_data_pth_n4).mkdir(parents=True, exist_ok=True)
    Path(target_data_pth_norm).mkdir(parents=True, exist_ok=True)
    
    tqdm.write(f"Source: {src_data_pth}")
    tqdm.write(f"N4 Target: {target_data_pth_n4}")
    tqdm.write(f"Norm Target: {target_data_pth_norm}")

    process_dataset(src_data_pth=src_data_pth, 
                    target_data_pth_n4=target_data_pth_n4, 
                    target_data_pth_norm=target_data_pth_norm)


print("Done !")
