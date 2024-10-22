import numpy as np
import wandb
import os
import pathlib
from operator import itemgetter
from MedicalSegmentation.med_seg_foundation.utils.constants import*

def get_class_rel_freqs(dataset):
    class_counts = np.zeros(dataset.num_classes)
    total_pixels = 0
    
    for idx in range(len(dataset)):
        tup = dataset[idx]
        mask = tup[1]
        
        class_counts += np.bincount(mask.flatten(), minlength=dataset.num_classes)  # Update class counts
        total_pixels += mask.flatten().size()[0]  # Update total pixels
        
    # Calculate relative frequencies
    relative_frequencies = class_counts / total_pixels
    assert relative_frequencies.size == dataset.num_classes
    return relative_frequencies

def log_class_rel_freqs(dataset, log_name_key):
    rel_freqs = get_class_rel_freqs(dataset)
    num_classes = rel_freqs.size
    
    for i in range(num_classes):
        metric_n = f'{log_name_key}_rel_freq_class{i}'
        wandb.define_metric(metric_n, summary="max")
        wandb.log({metric_n : rel_freqs[i]})
        
        
def get_file_list(fld_pth, start_idx=0, num_files=None, extension=None, 
                  file_suffix=None, file_name_contains=None, inc_exc=0):
    assert inc_exc in [0, 1], "0 to include 1 to exlude"

    if not os.path.isdir(fld_pth):
        return []

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
            files = [f for f in files if f.split(extension)[0].endswith(file_suffix)]
        else:
            # Exlude
            files = [f for f in files if not f.split(extension)[0].endswith(file_suffix)]
            
    if file_name_contains is not None:
        if inc_exc == 0:
            # Include
            files = [f for f in files if file_name_contains in f]
        else:
            # Exlude
            files = [f for f in files if not file_name_contains in f] 

    # Sort by name
    files.sort()
    
    if num_files is not None:
        num_files = num_files if num_files>len(files) else len(files)
    else:
        num_files = len(files)
    files_sel = files[start_idx:start_idx+num_files]
    return files_sel 


def get_ckp_path(search_dir, dataset, model_type, bb_size=None, backbone=None, dec_name=None):
    
    if model_type == ModelType.SEGMENTOR:
        assert bb_size is not None and backbone is not None and dec_name is not None
    
    if isinstance(search_dir, list):
        # A list of search directories are given
        res_ret = None
        for search_dir_i in search_dir:
            res = get_ckp_path(search_dir=search_dir_i, dataset=dataset, model_type=model_type, 
                               bb_size=bb_size, backbone=backbone, dec_name=dec_name)
            if res is not None:
                # Found a ckpt
                if res_ret is None:
                    # No previous ckpt exists
                    res_ret = res
                else:
                    # There's another ckpt from a different folder
                    print(f"Multiple checkpoints 1) {res_ret}, 2) {res}")
                    if res_ret.split('/')[-1] != res.split('/')[-1]:
                        # Take the newer  = bigger timestamp
                        res_ret = [res_ret, res]
                        res_ret.sort(reverse=True)
                        res_ret = res_ret[0]   
                     
        if model_type==ModelType.SEGMENTOR:
            assert res_ret is not None, f'Could not find a checkpoint for {backbone}, {bb_size}, {dec_name} in  {search_dir}'    
        else:
            assert res_ret is not None, f'Could not find a checkpoint for {ModelType.SEGMENTOR.name} in  {search_dir}'  
        print(f"Using: {res_ret}")
        return res_ret
                    
    else:
        if isinstance(search_dir, pathlib.Path):
            search_dir = str(search_dir)
        
        if model_type==ModelType.SEGMENTOR:
            # A single search directory is given
            if '_' in backbone:
                # A finetuning is used
                bb_dir = backbone + bb_size[0].upper()
            else:
                # Freeze model
                if backbone=='mae':
                    bb_dir = f'{backbone}_{bb_size}'
                elif backbone=='resnet':
                    bb_dir = 'resnet'
                    if bb_size=='base':
                        bb_dir+='101'
                    else:
                        raise ValueError('add size ')
                else:
                    raise ValueError('Names of the directories are not added yet !')
            
            if dec_name=='hq_hsam_mask_dec':
                dec_dir = 'HQHSAMdecHead'
            elif 'unet' in dec_name:
                dec_dir = 'UNetHead'
            else:
                raise ValueError('Not defined, add here !')
            dir_model_type = os.path.join(bb_dir, dec_dir)
            
        else:
            if model_type==ModelType.UNET:
                dir_model_type = 'UNetModel' 
            elif model_type==ModelType.SWINUNET:
                dir_model_type = 'SwinUNetModel'
            else:
                raise ValueError(f'Unknown Model Type: {model_type}')
            
        dir_i = os.path.join(search_dir, dataset, dir_model_type)
        files = get_file_list(fld_pth=dir_i, extension='.ckpt', file_name_contains='val_dice')
        
        if len(files)==0:
            return None
        elif len(files)==1:
            return os.path.join(dir_i, files[0])
        else:
            print(f'Found multiple candidates: {files}')
            # Choose the one from the latest run
            latest_timestamp = files[-1].split('-')[0]
            files_sel = [f for f in files if latest_timestamp in f]
            
            if len(files_sel)==1:
                return os.path.join(dir_i, files_sel[0])
            else:
                print(f'Found multiple ckpts files for the timestamp {latest_timestamp}: {files_sel}') 
                
                # Choose from a later epoch
                epochs = []
                for f in files_sel:
                    epochs.append(int(f.split('=')[-2].split('-')[0]))
                index, max_epoch = max(enumerate(epochs), key=itemgetter(1))    
                
                print(f'Taking the latest epoch={max_epoch}, {files_sel[index]}')
                return os.path.join(dir_i, files_sel[index])
                
            
      
       
            