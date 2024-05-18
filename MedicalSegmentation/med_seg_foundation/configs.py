from enum import Enum
from MedicalSegmentation.med_seg_foundation.models.EncDec.decoder.decoders import ConvHeadLinear, ResNetHead, UNetHead, SAMdecHead, HSAMdecHead, HQSAMdecHead, HQHSAMdecHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead
import torch
from utils.losses import DiceLoss, FocalLoss, CompositionLoss
from torch.nn import CrossEntropyLoss
from data.datasets import SegmentationDataset, SegmentationDatasetHDF5
from MedicalSegmentation.med_seg_foundation.models.EncDec.encoder.backbone_wrapper import DinoBackBone, SamBackBone, ResNetBackBone, LadderBackbone, \
    DinoReinBackbone, SamReinBackBone, MAEBackbone, MAEReinBackbone
from ModelSpecific.DinoMedical.prep_model import get_bb_name
from MedicalSegmentation.med_seg_foundation.models.segmentor import SegmentorEncDec, SegmentorModel
from MedicalSegmentation.med_seg_foundation.models.benchmarks.UNet.unet import UNet
from MedicalSegmentation.med_seg_foundation.models.benchmarks.SwinUnet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class ModelType(Enum):
    SEGMENTOR=1
    UNET=2
    SWINUNET=3

def get_data_attrs(name:str, use_hdf5=None, rcs_enabled=False):
    attrs = {}
    attrs['name'] = name
    attrs['rcs_enabled'] = rcs_enabled
        
    # Brain - HCP1
    if name=='hcp1':
        attrs['available_formats'] = ["png", "hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/hcp1'
            attrs['format'] = 'png'
        else:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'brain/hcp'
            attrs['hdf5_train_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5'
            attrs['hdf5_val_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5'
            attrs['hdf5_test_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        attrs['num_classes'] = 15
        attrs['vol_depth'] = 256  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Brain - HCP2    
    elif name=='hcp2':
        attrs['available_formats'] = ["png", "hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/hcp2'
            attrs['format'] = 'png'
        else:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'brain/hcp'
            attrs['hdf5_train_name'] = 'data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5'
            attrs['hdf5_val_name'] = 'data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5'
            attrs['hdf5_test_name'] = 'data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        attrs['num_classes'] = 15
        attrs['vol_depth'] = 256  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Brain - Abide-Caltech    
    elif name=='abide_caltech':
        attrs['available_formats'] = ["png", "hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/abide_caltech'
            attrs['format'] = 'png'
            ValueError('Varying volume depth !')
        else:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'brain/abide/caltech'
            attrs['hdf5_train_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_0.7_from_0_to_10.hdf5'
            attrs['hdf5_val_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_0.7_from_10_to_15.hdf5'
            attrs['hdf5_test_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_0.7_from_16_to_36.hdf5'
        attrs['num_classes'] = 15
        attrs['vol_depth'] = 256  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    elif name=='abide_stanford':
        attrs['available_formats'] = ["png", "hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/abide_stanford'
            attrs['format'] = 'png'
            ValueError('Varying volume depth !')
        else:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'brain/abide/stanford'
            attrs['hdf5_train_name'] = 'data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_0_to_10.hdf5'
            attrs['hdf5_val_name'] = 'data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_10_to_15.hdf5'
            attrs['hdf5_test_name'] = 'data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_16_to_36.hdf5'
        attrs['num_classes'] = 15
        attrs['vol_depth'] = 132  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Prostate - NCI    
    elif name=='prostate_nci':
        attrs['available_formats'] = ["hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['num_classes'] = 3
        attrs['vol_depth_first'] = 15  # First val volume depth NOT ALL THE SAME !
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        if use_hdf5:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'nci'
            attrs['hdf5_train_name'] = 'nci_train.hdf5'
            attrs['hdf5_val_name'] = 'nci_val.hdf5'
            attrs['hdf5_test_name'] = 'nci_test.hdf5'
        else:
            attrs['format'] = 'png'
            
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Prostate - USZ        
    elif name=='prostate_usz':
        attrs['available_formats'] = ["hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['data_path_suffix'] = 'prostate/pirad_erc'
        attrs['num_classes'] = 3
        attrs['vol_depth_first'] = 22  # First val volume depth  NOT ALL THE SAME !
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
            
        if use_hdf5:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'pirad_erc'
            attrs['hdf5_train_name'] = 'data_2d_from_40_to_68_size_256_256_res_0.625_0.625_ek.hdf5'
            attrs['hdf5_val_name'] = 'data_2d_from_20_to_40_size_256_256_res_0.625_0.625_ek.hdf5'
            attrs['hdf5_test_name'] = 'data_2d_from_0_to_20_size_256_256_res_0.625_0.625_ek.hdf5'
        else:
            attrs['format'] = 'png'
            
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Cardiac - ACDC      
    elif name=='cardiac_acdc':
        attrs['available_formats'] = ["hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['data_path_suffix'] = 'cardiac/acdc'
        attrs['num_classes'] = 4
        attrs['vol_depth_first'] = 10  # First val volume depth  NOT ALL THE SAME !
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        if use_hdf5:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'acdc'
            attrs['hdf5_train_name'] = 'acdc_train.hdf5'
            attrs['hdf5_val_name'] = 'acdc_val.hdf5'
            attrs['hdf5_test_name'] = 'acdc_test.hdf5'   
        else:
            attrs['format'] = 'png'
            
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight  
    
    # Cardiac - RVSC           
    elif name=='cardiac_rvsc':
        attrs['available_formats'] = ["hdf5"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['data_path_suffix'] = 'cardiac/rvsc'
        attrs['num_classes'] = 3
        attrs['vol_depth_first'] = 10  # First val volume depth  NOT ALL THE SAME !
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        if use_hdf5:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = 'rvsc'
            attrs['hdf5_train_name'] = 'rvsc_train.hdf5'
            attrs['hdf5_val_name'] = 'rvsc_val.hdf5'
            attrs['hdf5_test_name'] = 'rvsc_test.hdf5'
        else:
            attrs['format'] = 'png'
            
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    # LumbarSpine - MRSpineSegV     
    elif name=='spine_mrspinesegv':
        attrs['available_formats'] = ["png"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if use_hdf5:
            ValueError(f'HDF5 format is not supported for {name}')
            
        attrs['data_path_suffix'] = 'lumbarspine/MRSpineSegV'
        attrs['num_classes'] = 6
        attrs['vol_depth'] = 12  
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
            
        if use_hdf5:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = ''
            attrs['hdf5_train_name'] = '.hdf5'
            attrs['hdf5_val_name'] = '.hdf5'
            attrs['hdf5_test_name'] = '.hdf5'
        else:
            attrs['format'] = 'png'
            
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    # LumbarSpine - VerSe     
    elif name=='spine_verse':
        attrs['available_formats'] = ["png"]
        if use_hdf5 is None:
            use_hdf5 = 'hdf5' in attrs['available_formats']
            
        if use_hdf5:
            ValueError(f'HDF5 format is not supported for {name}')
            
        attrs['data_path_suffix'] = 'lumbarspine/VerSe'
        attrs['num_classes'] = 6
        attrs['vol_depth'] = 120  
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
            
        if use_hdf5:
            attrs['format'] = 'hdf5'
            attrs['data_path_suffix'] = ''
            attrs['hdf5_train_name'] = '.hdf5'
            attrs['hdf5_val_name'] = '.hdf5'
            attrs['hdf5_test_name'] = '.hdf5'
        else:
            attrs['format'] = 'png'
            
        # weight = [0.1] + [1.]*(attrs['num_classes']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    else:
        ValueError(f'Dataset: {name} is not defined')
        
    return attrs

# def get_batch_sz(data_attrs, num_gpu):
#     assert num_gpu>0
    
#     dataset_name = data_attrs['name']
    
#     if dataset_name=='hcp1':
#         batch_sz = 8
        
#     elif dataset_name=='hcp2':
#         batch_sz = 8
        
#     elif dataset_name=='abide_caltech':
#         batch_sz = 8
        
#     elif dataset_name=='abide_stanford':
#         batch_sz = 6
       
#     elif dataset_name=='cardiac_acdc':
#        batch_sz = 6 
            
#     elif dataset_name=='cardiac_rvsc':
#         batch_sz = 6 
        
#     elif dataset_name=='prostate_nci':
#         batch_sz = 8 
            
#     elif dataset_name=='prostate_usz':
#         batch_sz = 10 
        
#     else:
#         ValueError(f'Dataset: {dataset_name} is not defined')
    
#     batch_sz = batch_sz // num_gpu     
   
#     return batch_sz


def get_bb_cfg(bb_name, bb_size, train_bb, dec_name, main_pth, pretrained=True):
    # Set general fields
    bb_cps_pth = 'Checkpoints/Orig/backbone'
    out_idx = None
    outs_from_diff_conv_layers = False  # For resnet
    uniform_oup_size = True  # For resnet
    if bb_name in ["dino", "sam", "medsam", "mae", "resnet"]:
        if dec_name in ['unet', 'unetS']:
            n_out = 5*2
        elif dec_name in ['sam_mask_dec', 'hsam_mask_dec']:
            n_out=1
        elif dec_name in ['hq_sam_mask_dec', 'hq_hsam_mask_dec']:
            n_out=2
            if "sam" in bb_name:
                if bb_size=="base":
                    first_glob_attn_i = 2
                elif bb_size=="large":
                    first_glob_attn_i = 5
                elif bb_size=='huge':
                    first_glob_attn_i = 7
                out_idx = [first_glob_attn_i, -1]
                
            elif "dino" in  bb_name:
                out_idx = [0, -1]
            else:
                out_idx = [0, -1]
            
            outs_from_diff_conv_layers = True# For resnet
            uniform_oup_size = False
        else:
            n_out = 4
    
    last_out_first = True # Keep True
    
    # Backbone Specifics
    if bb_name == 'dino':
        assert bb_size in ["small", "base", "large", "giant"]
        backbone_name = get_bb_name(bb_size)
        
        if pretrained:
            bb_checkpoint_path = main_pth/bb_cps_pth/f'DinoV2/{backbone_name}_pretrain.pth'
        else:
            bb_checkpoint_path = None
        
        name = DinoBackBone.__name__
        params = dict(nb_outs=n_out,
                      name=backbone_name,
                      last_out_first=last_out_first,
                      bb_model=None,
                      cfg=dict(backbone_name=backbone_name, backbone_cp=bb_checkpoint_path),
                      train=train_bb,
                      disable_mask_tokens=True,
                      pre_normalize=False,
                      out_idx=out_idx)
        
    elif bb_name in ['rein_dino', 'reinL_dino'] :
        assert pretrained
        assert not train_bb
        dino_name_params = get_bb_cfg('dino', bb_size, train_bb, dec_name, main_pth, pretrained)
        name = DinoReinBackbone.__name__
        params = dino_name_params['params']
        del params['train']
        params['name'] = bb_name+bb_size[0].upper()
        params['lora_reins'] = bb_name=='reinL_dino'
        
    elif bb_name in ['rein_sam', 'reinL_sam', 'rein_medsam', 'reinL_medsam'] :
        assert pretrained
        assert not train_bb
        sam_name_params = get_bb_cfg(bb_name.split('_')[-1], bb_size, train_bb, dec_name, main_pth, pretrained)
        name = SamReinBackBone.__name__
        params = sam_name_params['params']
        del params['train']
        params['name'] = bb_name+bb_size[0].upper()
        params['lora_reins'] = 'reinL' in bb_name
        
    elif bb_name in ['rein_mae', 'reinL_mae']:  
        assert pretrained
        assert not train_bb
        mae_name_params = get_bb_cfg(bb_name.split('_')[-1], bb_size, train_bb, dec_name, main_pth, pretrained)
        name = MAEReinBackbone.__name__
        params = mae_name_params['params']
        del params['train']
        params['name'] = bb_name+bb_size[0].upper()
        params['lora_reins'] = 'reinL' in bb_name
        
    elif bb_name == 'sam' or bb_name == 'medsam':
        apply_neck = (dec_name=='sam_mask_dec')
        
        if pretrained:
            if bb_name == 'sam':
                assert bb_size in ["base", "large", "huge"]
                bb_checkpoint_path = main_pth/bb_cps_pth/'SAM'
                prefix = ''
                if bb_size=='base':
                    bb_checkpoint_path = bb_checkpoint_path / 'sam_vit_b_01ec64.pth'
                elif bb_size=='large':
                    bb_checkpoint_path = bb_checkpoint_path / 'sam_vit_l_0b3195.pth'
                elif bb_size=='huge':
                    bb_checkpoint_path = bb_checkpoint_path / 'sam_vit_h_4b8939.pth'
                else:
                    ValueError(f'Size: {bb_size} is not available for SAM')
            else:
                prefix = 'med'
                assert bb_size=='base', f'Medsam is only availabel for size base but got {bb_size}'
                bb_checkpoint_path = main_pth/bb_cps_pth/'MedSam'/'medsam_vit_b.pth'
                    
        else:
            bb_checkpoint_path = None
            
        name = SamBackBone.__name__
        params = dict(nb_outs=n_out,
                      name=f'{prefix}sam_enc_vit_{bb_size}',
                      last_out_first=last_out_first,
                      bb_model=None,
                      cfg=dict(bb_size=bb_size, sam_checkpoint=bb_checkpoint_path, apply_neck=apply_neck),
                      train=train_bb,
                      interp_to_inp_shape= False, #(dec_name!='sam_mask_dec'),
                      pre_normalize=False,
                      out_idx=out_idx)
        
    elif bb_name=='mae':    
        assert bb_size in ["base", "large", "huge"]
        if pretrained:
            bb_checkpoint_path = main_pth/bb_cps_pth/f'MAE/mae_pretrain_vit_{bb_size}.pth'
        else:
            bb_checkpoint_path = None
            
        name = MAEBackbone.__name__
        params = dict(name=f'{bb_name}_{bb_size}',
                      nb_outs=n_out,
                      cfg=dict(bb_size=bb_size, checkpoint=bb_checkpoint_path,
                               enc_only=True),
                      out_idx=out_idx,
                      train=train_bb,
                      last_out_first=last_out_first,)
        
    elif bb_name == 'resnet':
        if dec_name=='unet':
            ValueError(f'Decoder {dec_name} is not supported for {bb_name} backbone')
        if bb_size=='smallest':
            layers=18
        elif bb_size=='tiny':
            layers=34
        elif bb_size=='small':
            layers=50
        elif bb_size=='base':
            layers=101
        elif bb_size=='large':
            layers=152
        else:
            ValueError(f'Size is not defined for resnet {bb_size}')
        backbone_name = f'resnet{layers}'    
        
        weights = f'ResNet{layers}_Weights.DEFAULT' if pretrained else None
        
        name = ResNetBackBone.__name__
        params = dict(name=backbone_name,
                      bb_model=None,
                      cfg=dict(name=backbone_name, weights=weights),
                      train=train_bb,
                      nb_layers=layers,
                      pre_normalize=False,
                      nb_outs=n_out,
                      last_out_first=last_out_first,
                      out_idx=out_idx,
                      outs_from_diff_conv_layers=outs_from_diff_conv_layers,
                      uniform_oup_size=uniform_oup_size)   
        
    elif 'ladder' in bb_name:
        name = LadderBackbone.__name__
        
        ## BB1
        bb1_name_params = get_bb_cfg(bb_name=bb_name.split('_')[-1], 
                                     bb_size=bb_size, 
                                     train_bb=train_bb, 
                                     dec_name=dec_name, 
                                     main_pth=main_pth, 
                                     pretrained=True)
        if 'sam' in bb_name.split('_')[-1]:
            bb1_name_params['params']['cfg']['apply_neck']=True
            
        ## BB2    
        if 'ladderR' in bb_name:
            # Resnet18 (excluding the last block) as ladder bb 
            bb2_name_params = get_bb_cfg(bb_name='resnet', 
                                         bb_size='tiny', 
                                         train_bb=True, 
                                         dec_name=dec_name, 
                                         main_pth=main_pth, 
                                         pretrained=True,)
            bb2_name_params['params']['skip_last_layer']=True
        
        elif 'ladderD' in bb_name:
            # DinoS as ladder bb 
            bb2_name_params = get_bb_cfg(bb_name='dino', 
                                         bb_size='small', 
                                         train_bb=True, 
                                         dec_name=dec_name, 
                                         main_pth=main_pth, 
                                         pretrained=True)
            bb2_name_params['params']['nb_outs']=bb1_name_params['params'].get('nb_outs', 1)
            
        else:
            ValueError(f'Undefined ladder backbone type: {bb_name}')
        
        
        # Ladder backbone parameters
        bb2_name_params['train'] = True
        params = dict(name=bb_name+bb_size[0].upper(),
                      bb1_name_params=bb1_name_params,
                      bb2_name_params=bb2_name_params)
        
    else:
        ValueError(f'Undefined backbone: {bb_name}')
        
    return dict(name=name, params=params)
        
    


def get_dec_cfg(dec_name, dataset_attrs, n_in, main_path=None, bb_size=None):
    num_classes = dataset_attrs['num_classes']
    
    if dec_name == 'lin':
        class_name = ConvHeadLinear.__name__
        # Linear classification of each patch + upsampling to pixel dim
        dec_head_cfg = dict(num_classes=num_classes,
                            in_channels=[i for i in range(n_in)],
                            bilinear=True,
                            dropout_rat=0.1,)
    elif dec_name == 'fcn':
        class_name = FCNHead.__name__
        # https://arxiv.org/abs/1411.4038
        dec_head_cfg = dict(num_convs=3,
                            kernel_size=3,
                            concat_input=True,
                            dilation=1,
                            num_classes=num_classes,  # output channels
                            dropout_ratio=0.1,
                            conv_cfg=dict(type='Conv2d'), # None = conv2d
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU'),
                            in_index=[i for i in range(n_in)],
                            input_transform='resize_concat',
                            init_cfg=dict(
                                type='Normal', std=0.01, override=dict(name='conv_seg')))
    elif dec_name == 'psp':
        class_name = PSPHead.__name__
        # https://arxiv.org/abs/1612.01105
        dec_head_cfg = dict(pool_scales=(1, 2, 3, 6),
                                num_classes=num_classes,  # output channels
                                dropout_ratio=0.1,
                                conv_cfg=dict(type='Conv2d'), # None = conv2d
                                norm_cfg=dict(type='BN'),
                                act_cfg=dict(type='ReLU'),
                                in_index=[i for i in range(n_in)],
                                input_transform='resize_concat',
                                init_cfg=dict(
                                    type='Normal', std=0.01, override=dict(name='conv_seg')))
    elif dec_name == 'da':
        class_name = DAHead.__name__
        # https://arxiv.org/abs/1809.02983
        dec_head_cfg = dict(num_classes=num_classes,  # output channels
                            dropout_ratio=0.1,
                            conv_cfg=dict(type='Conv2d'), # None = conv2d
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU'),
                            in_index=[i for i in range(n_in)],
                            input_transform='resize_concat',
                            init_cfg=dict(
                                type='Normal', std=0.01, override=dict(name='conv_seg')))
        
    elif dec_name == 'segformer':
        class_name = SegformerHead.__name__
        # https://arxiv.org/abs/2105.15203
        dec_head_cfg = dict(interpolate_mode='bilinear',
                            num_classes=num_classes,  # output channels
                            dropout_ratio=0.1,
                            conv_cfg=dict(type='Conv2d'), # None = conv2d
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU'),
                            in_index=[i for i in range(n_in)],
                            init_cfg=dict(
                                type='Normal', std=0.01, override=dict(name='conv_seg')))
        
    elif dec_name == 'resnet':
        class_name = ResNetHead.__name__
        # ResNet-like with recurrent convs
        dec_head_cfg = dict(num_classes=num_classes,
                            # in_index=None,
                            # in_resize_factors=None,
                            # align_corners=False,
                            dropout_rat_cls_seg=0.1,
                            nb_up_blocks=4,
                            upsample_facs_ch=2,
                            upsample_facs_wh=2,
                            bilinear=False,
                            conv_per_up_blk=2,
                            res_con=True,
                            res_con_interv=None, # None = Largest possible (better)
                            skip_first_res_con=False, 
                            recurrent=True,
                            recursion_steps=2,
                            in_channels_red=384*n_in)
    elif dec_name in ['unet', 'unetS']:
        class_name = UNetHead.__name__
        # https://arxiv.org/abs/1505.04597 (unet papaer)
        input_group_cat_nb = 2
        dec_head_cfg = dict(num_classes=num_classes,
                            # in_index=None,
                            # in_resize_factors=None,
                            # align_corners=False,
                            dropout_rat_cls_seg=0.1,
                            nb_up_blocks=4,
                            upsample_facs_ch=2,
                            upsample_facs_wh=2,
                            bilinear=False,
                            conv_per_up_blk=2, # 5
                            res_con=True,
                            res_con_interv=None, # None = Largest possible (better)
                            skip_first_res_con=False, 
                            recurrent=True,
                            recursion_steps=2, # 3
                            resnet_cat_inp_upscaling=True,
                            input_group_cat_nb=input_group_cat_nb,
                            in_channels_red=576 if dec_name=='unet' else 240)  # 576  |  384*input_group_cat_nb
     
    elif dec_name in ['sam_mask_dec', 'hsam_mask_dec', 'hq_sam_mask_dec', 'hq_hsam_mask_dec']:
        
        pretrained=True
        if pretrained:
            assert main_path is not None
            bb_cps_pth = 'Checkpoints/Orig/backbone'
            
            sam_checkpoint_prom_enc = main_path/bb_cps_pth/'MedSam'/'medsam_vit_b.pth' 
            
            if bb_size=='base':
                    sam_checkpoint_neck = main_path/bb_cps_pth/'SAM' / 'sam_vit_b_01ec64.pth'
            elif bb_size=='large':
                sam_checkpoint_neck = main_path/bb_cps_pth/'SAM' / 'sam_vit_l_0b3195.pth'
            elif bb_size=='huge':
                sam_checkpoint_neck = main_path/bb_cps_pth/'SAM' / 'sam_vit_h_4b8939.pth'
            else:
                # raise ValueError(f'Size: {bb_size} is not available for SAM')
                sam_checkpoint_neck = None
               
        else:
            sam_checkpoint_neck = None
            sam_checkpoint_prom_enc = None
        
        if dec_name=='sam_mask_dec':
            class_name = SAMdecHead.__name__ 
        elif dec_name=='hsam_mask_dec':
            class_name = HSAMdecHead.__name__
        elif dec_name=='hq_sam_mask_dec':
            class_name = HQSAMdecHead.__name__
        elif dec_name=='hq_hsam_mask_dec':
            class_name = HQHSAMdecHead.__name__
        else:
            ValueError(f'Unknown dec name : {dec_name}')
            
        dec_head_cfg = dict(num_classes=num_classes,
                            sam_checkpoint_neck=sam_checkpoint_neck,
                            sam_checkpoint_prom_enc=sam_checkpoint_prom_enc)
        if dec_name=='sam_mask_dec':
            dec_head_cfg['train_prompt_enc']=True
    else:
        ValueError(f'Decoder name {dec_name} is not defined')
        
    return dict(name=class_name, params=dec_head_cfg)


def get_loss_cfg(loss_key, data_attr):
    
    ignore_idx_loss = data_attr['ignore_idx_loss']
    
    weight = data_attr['weight']
    
    # CE Loss
    loss_cfg_ce = dict(ignore_index=ignore_idx_loss if ignore_idx_loss is not None else -100,
                       weight=weight)

    # Dice Loss
    epsilon = 1  # smoothing factor 
    k=1  # power
    loss_cfg_dice = dict(prob_inputs=False, 
                        ignore_idxs=ignore_idx_loss, # not removing results in better segmentation
                        reduction='mean',
                        epsilon=epsilon,
                        k=k,
                        weight=weight)

    # CE-Dice Loss
    loss_cfg_dice_ce=dict(loss1=dict(name='Dice',
                                    params=loss_cfg_dice),
                        loss2=dict(name='CE', 
                                    params=loss_cfg_ce),
                        comp_rat=0.8)

    # Focal Loss
    loss_cfg_focal = dict(ignore_idxs=ignore_idx_loss,
                        gamma=2,
                        weight=weight)

    # Focal-Dice Loss
    loss_cfg_comp_foc_dice=dict(loss1=dict(name='Focal',
                                    params=loss_cfg_focal),
                                loss2=dict(name='Dice', 
                                            params=loss_cfg_dice),
                                comp_rat=20/21)
    
    if loss_key=='ce':
        return dict(name=CrossEntropyLoss.__name__, params=loss_cfg_ce)
     
    elif loss_key=='dice_ce':
        return dict(name=CompositionLoss.__name__, params=loss_cfg_dice_ce)

    elif loss_key=='focal':
        return dict(name=FocalLoss.__name__, params=loss_cfg_focal)

    elif loss_key=='focal_dice':
        return dict(name=CompositionLoss.__name__, params=loss_cfg_comp_foc_dice)
        
    else:
        ValueError(f'Loss {loss_key} is not defined')
        

def get_optimizer_cfg(lr):
     
    optm_cfg = dict(name='AdamW',
                params=dict(lr = lr,
                            weight_decay = 1e-5,   # 0.5e-4  | 1e-2
                            betas = (0.9, 0.999)))
    return optm_cfg


def get_scheduler_cfg(nb_epochs):
    warmup_iters = max(1, int(nb_epochs*0.05))  # try *0.25
    
    scheduler_configs = []
    scheduler_configs.append(\
        dict(name='LinearLR',
            params=dict(start_factor=1/3, end_factor=1.0, total_iters=warmup_iters)))
    
    # return scheduler_configs[0]
    
    scheduler_configs.append(\
        dict(name='PolynomialLR',
            params=dict(power=1.0, total_iters=(nb_epochs-warmup_iters)*2)))

    scheduler_cfg = dict(name='SequentialLR',
                        params=dict(scheduler_configs=scheduler_configs,
                                    milestones=[warmup_iters]),
                        )
    return scheduler_cfg
            
    
    
def get_metric_cfgs(data_attr, dom_gen_tst=False):
    ignore_idx_metric = data_attr['ignore_idx_metric']
    
    epsilon = 1  # smoothing factor 
    k=1  # power

    miou_cfg=dict(prob_inputs=False, # Decoder does not return probas explicitly
                soft=False,
                ignore_idxs=ignore_idx_metric,  # bg channel to be removed 
                reduction='mean',
                EN_vol_scores=True,
                epsilon=epsilon)

    dice_cfg=dict(prob_inputs=False,  # Decoder does not return probas explicitly
                soft=False,
                ignore_idxs=ignore_idx_metric,
                reduction='mean',
                k=k, 
                epsilon=epsilon,
                EN_vol_scores=True)

    metric_cfgs=[dict(name='mIoU', params=miou_cfg), 
                 dict(name='dice', params=dice_cfg)]
    
    return metric_cfgs


def get_minibatch_log_idxs(batch_sz):
    """Computes which samples in the batch to log for segmentation"""
    
    # Seg result logging cfg
    seg_log_per_batch = min(4, batch_sz)  # Log N samples from each minibatch
    assert seg_log_per_batch<=batch_sz

    # Which samles in the minibatch to log
    sp = seg_log_per_batch+1
    multp = batch_sz//sp
    # maximal separation from each other and from edges (from edges is prioritized)
    if multp>0:
        log_idxs = torch.arange(multp, 
                                multp*sp, 
                                multp)
        log_idxs = log_idxs + (batch_sz%sp)//2
    else:
        log_idxs = torch.arange(0, 
                                batch_sz, 
                                1)
    log_idxs = log_idxs.tolist()
    
    return log_idxs

def get_batch_log_idxs(batch_sz, data_attr):
    vol_depth = data_attr['vol_depth'] if 'vol_depth' in data_attr.keys() else data_attr['vol_depth_first']
    
    seg_res_nb_vols = 1  # Process minibatches for N number of 3D volumes
    seg_log_nb_batches = 16
    step = max(int(vol_depth/batch_sz/seg_log_nb_batches), 1)
    seg_log_batch_idxs = torch.arange(0+step-1, max(min(seg_log_nb_batches*step, vol_depth//batch_sz*seg_res_nb_vols), seg_log_nb_batches), step).tolist()
    assert len(seg_log_batch_idxs)==seg_log_nb_batches
    return seg_log_batch_idxs


def get_lr(model_type, **kwargs):
    if model_type==ModelType.SEGMENTOR:
        bb_name=kwargs['backbone'] 
        bb_size=kwargs['backbone_sz'] 
        train_bb=kwargs['train_backbone']
        dec_name=kwargs['dec_head_key']
        dataset_attrs = kwargs['dataset_attrs']
        if bb_name=='dino':
            if dataset_attrs['name'] in ['spine_verse', 'hcp1']:
                return 1e-5
            return 2e-5
        
        elif bb_name =='resnet': #or 'ladder' in bb_name:
            return 5e-4
        
        elif 'reins' in bb_name:
            return 1e-4
        
        elif 'ladderR' in bb_name:
            return 5e-5
        
        elif 'ladderD' in bb_name:
            if dataset_attrs['name'] in ['spine_verse', 'hcp1']:
                return 1e-5
            return 2e-5
                
        # SAM and MedSAM
        else: 
            return 5e-5 
    elif model_type in [ModelType.UNET, ModelType.SWINUNET]:
        return 5e-5 
    else:
        ValueError(f'Unknown model type {model_type}')


def get_lit_segmentor_cfg(batch_sz, nb_epochs, loss_cfg_key, dataset_attrs, gpus, model_type, 
                          dom_gen_tst=False, **kwargs):

    
    if model_type==ModelType.SEGMENTOR:
        kwargs['dataset_attrs'] = dataset_attrs
                
        # Backbone config
        bb_cfg = get_bb_cfg(bb_name=kwargs['backbone'], bb_size=kwargs['backbone_sz'], train_bb=kwargs['train_backbone'], 
                            dec_name=kwargs['dec_head_key'], main_pth=kwargs['main_pth'], pretrained=True)
        
        if bb_cfg['name'] != LadderBackbone.__name__:
            n_out_bb = bb_cfg['params'].get('nb_outs', 1)
        else:
            n_out_bb = bb_cfg['params']['bb1_name_params']['params'].get('nb_outs', 1)
        
        # Decnconfig
        dec_head_cfg = get_dec_cfg(dec_name=kwargs['dec_head_key'], dataset_attrs=dataset_attrs, n_in=n_out_bb,
                                   bb_size=kwargs['backbone_sz'], main_path=kwargs['main_pth'])
        
        # Segmentor config (Encoder-Decoder)
        segmentor_cfg = dict(name=SegmentorEncDec.__name__,
                         params=dict(backbone=bb_cfg,
                                     decode_head=dec_head_cfg,
                                     reshape_dec_oup=True,
                                     align_corners=False))
    
    else:    
        
        if model_type==ModelType.UNET:            
            model_cfg = dict(name=UNet.__name__,
                            params=dict(n_channels=3, 
                                        n_classes=dataset_attrs['num_classes'], 
                                        bilinear=False))
            
        elif model_type==ModelType.SWINUNET:            
            model_cfg = dict(name=SwinTransformerSys.__name__,
                            params=dict(n_channels=3, 
                                        num_classes=dataset_attrs['num_classes'], 
                                        ))    
            
        else:
            ValueError(f'Unknown model type {model_type}')
            
        segmentor_cfg = dict(name=SegmentorModel.__name__,
                             params=dict(reshape_dec_oup=True,
                                         align_corners=False,
                                         model=model_cfg,
                                         target_inp_shape=224))        
    
    # Get lr
    lr = get_lr(model_type=model_type, **kwargs)
    
    # Optimizer Config    
    optm_cfg = get_optimizer_cfg(lr=lr)

    # LR scheduler config
    scheduler_cfg = get_scheduler_cfg(nb_epochs=nb_epochs)

    # Loss Config
    loss_cfg = get_loss_cfg(loss_key=loss_cfg_key, data_attr=dataset_attrs)

    # Metrics
    metric_cfgs = get_metric_cfgs(data_attr=dataset_attrs, dom_gen_tst=dom_gen_tst)

    # Log indexes for segmentation for the minibatch
    log_idxs = get_minibatch_log_idxs(batch_sz=batch_sz)

    # Log indexes for segmentation for the batches
    seg_log_batch_idxs = get_batch_log_idxs(batch_sz=batch_sz, data_attr=dataset_attrs)

    # Log seg val reult every N epochs during training
    seg_res_log_itv = max(nb_epochs//5, 1)  
    
    # Lit segmentor config
    segmentor_cfg_lit = dict(segmentor=segmentor_cfg,
                         loss_config=loss_cfg, 
                         optimizer_config=optm_cfg,
                         schedulers_config=scheduler_cfg,
                         metric_configs=metric_cfgs,
                         val_metrics_over_vol=True, # Also report metrics over vol
                         seg_log_batch_idxs=seg_log_batch_idxs,
                         minibatch_log_idxs=log_idxs,
                         seg_val_intv=seg_res_log_itv,
                         sync_dist_train=gpus>1,
                         sync_dist_val=gpus>1,
                         sync_dist_test=gpus>1)
    return segmentor_cfg_lit
    
    


def get_augmentations():    
    augmentations = []  
    
    augmentations.append(dict(type='ElasticTransformation', data_aug_ratio=0.25))
    augmentations.append(dict(type='StructuralAug', data_aug_ratio=0.25))
    augmentations.append(dict(type='PhotoMetricDistortion'))
    
    return augmentations


def get_datasets(data_root_pth, data_attr, train_procs, val_test_procs):
    """Order of procs: First augmentations then pre-processings ! """
    
    data_path_suffix = data_attr['data_path_suffix']
    dataset = data_attr['name']
    num_classes = data_attr['num_classes']
    rcs_enabled = data_attr['rcs_enabled']
    
    data_root_pth = data_root_pth / data_path_suffix

    if data_attr['format']=='png':
        train_fld = 'train-filtered' if 'hcp' in dataset else 'train'
        nz = data_attr.get('vol_depth')

        train_dataset = SegmentationDataset(img_dir=data_root_pth/'images'/train_fld,
                                            mask_dir=data_root_pth/'labels'/train_fld,
                                            num_classes=num_classes,
                                            file_extension='.png',
                                            mask_suffix='_labelTrainIds',
                                            augmentations=train_procs,
                                            nz=nz,
                                            ret_n_z=False, rcs_enabled=rcs_enabled)
        val_dataset = SegmentationDataset(img_dir=data_root_pth/'images/val',
                                        mask_dir=data_root_pth/'labels/val',
                                        num_classes=num_classes,
                                        file_extension='.png',
                                        mask_suffix='_labelTrainIds',
                                        augmentations=val_test_procs,
                                        nz=nz,
                                        ret_n_z=True, rcs_enabled=False)
        test_dataset = SegmentationDataset(img_dir=data_root_pth/'images/test',
                                        mask_dir=data_root_pth/'labels/test',
                                        num_classes=num_classes,
                                        file_extension='.png',
                                        mask_suffix='_labelTrainIds',
                                        augmentations=val_test_procs,
                                        nz=nz,
                                        ret_n_z=True,
                                        rcs_enabled=False)
        
    elif data_attr['format']=='hdf5':
        hdf5_train_name = data_attr['hdf5_train_name']
        hdf5_val_name = data_attr['hdf5_val_name']
        hdf5_test_name = data_attr['hdf5_test_name']
        
        train_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_train_name, 
                                                num_classes=num_classes, 
                                                augmentations=train_procs,
                                                ret_n_xyz=False, rcs_enabled=rcs_enabled)
        val_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_val_name, 
                                                num_classes=num_classes, 
                                                augmentations=val_test_procs,
                                                ret_n_xyz=True, rcs_enabled=False)
        test_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_test_name, 
                                                num_classes=num_classes, 
                                                augmentations=val_test_procs,
                                                ret_n_xyz=True, rcs_enabled=False)
    else:
        ValueError(f'Unsupported data format {data_attr["format"]}')
        
    return train_dataset, val_dataset, test_dataset
    



    
    
        