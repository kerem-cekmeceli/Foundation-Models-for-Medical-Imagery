from enum import Enum
from layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead
import torch
from eval.losses import DiceLoss, FocalLoss, CompositionLoss
from torch.nn import CrossEntropyLoss
from data.datasets import SegmentationDataset, SegmentationDatasetHDF5
from layers.backbone_wrapper import DinoBackBone, SamBackBone, ResNetBackBone
from ModelSpecific.DinoMedical.prep_model import get_bb_name
from MedicalSegmentation.med_seg_foundation.models.segmentor import Segmentor
from MedicalSegmentation.med_seg_foundation.models.unet import UNet

class ModelType(Enum):
    SEGMENTOR=1
    UNET=2

def get_data_attrs(name:str, use_hdf5=None):
    attrs = {}
        
    # Brain - HCP1
    if name=='hcp1':
        attrs['name'] = name
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
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 256  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Brain - HCP2    
    elif name=='hcp2':
        attrs['name'] = name
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
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 256  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Brain - Abide-Caltech    
    elif name=='abide_caltech':
        attrs['name'] = name
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
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 256  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    elif name=='abide_stanford':
        attrs['name'] = name
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
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 132  # volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
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
            
        attrs['name'] = name
        attrs['num_classses'] = 3
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
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
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
            
        attrs['name'] = name
        attrs['data_path_suffix'] = 'prostate/pirad_erc'
        attrs['num_classses'] = 3
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
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
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
            
        attrs['name'] = name
        attrs['data_path_suffix'] = 'cardiac/acdc'
        attrs['num_classses'] = 3
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
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
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
            
        attrs['name'] = name
        attrs['data_path_suffix'] = 'cardiac/rvsc'
        attrs['num_classses'] = 2
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
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
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
    bb_cps_pth = 'Checkpoints/Orig/backbone'
    
    if bb_name == 'dino':
        if dec_name=='unet':
            n_out = 5*2
        else:
            n_out = 4
        last_out_first = True
        
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
                      disable_mask_tokens=True)
        
    elif bb_name == 'sam' or bb_name == 'medsam':
        if dec_name=='unet':
            n_out = 5*2
        else:
            n_out = 4
        last_out_first = True
        
        if pretrained:
            if bb_name == 'sam':
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
                      cfg=dict(bb_size=bb_size, sam_checkpoint=bb_checkpoint_path),
                      train=train_bb,
                      interp_to_inp_shape=True)
        
    elif bb_name == 'resnet':
        if dec_name=='unet':
            ValueError(f'Decoder {dec_name} is not supported for {bb_name} backbone')
        
        if bb_size=='small':
            layers=50
        elif bb_size=='base':
            layers=101
        elif bb_size=='large':
            layers=152
        else:
            ValueError(f'Size is not defined for resnet {bb_size}')
        backbone_name = f'resnet{layers}'    
        
        weights = f'ResNet{layers}_Weights.IMAGENET1K_V2' if pretrained else None
        
        name = ResNetBackBone.__name__
        params = dict(name=backbone_name,
                      bb_model=None,
                      cfg=dict(name=backbone_name, weights=weights),
                      train=train_bb,
                      )   
        
    else:
        ValueError(f'Undefined backbone: {bb_name}')
        
    return dict(name=name, params=params)
        
    


def get_dec_cfg(dec_name, dataset_attrs):
    num_classses = dataset_attrs['num_classses']
    
    if dec_name == 'lin':
        class_name = ConvHeadLinear.__name__
        n_in_ch = 4
        # Linear classification of each patch + upsampling to pixel dim
        dec_head_cfg = dict(num_classses=num_classses,
                            bilinear=True,
                            dropout_rat=0.1,)
    elif dec_name == 'fcn':
        class_name = FCNHead.__name__
        n_in_ch = 4
        # https://arxiv.org/abs/1411.4038
        dec_head_cfg = dict(num_convs=3,
                            kernel_size=3,
                            concat_input=True,
                            dilation=1,
                            num_classes=num_classses,  # output channels
                            dropout_ratio=0.1,
                            conv_cfg=dict(type='Conv2d'), # None = conv2d
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU'),
                            # in_index=[i for i in range(n_in_ch)],
                            input_transform='resize_concat',
                            init_cfg=dict(
                                type='Normal', std=0.01, override=dict(name='conv_seg')))
    elif dec_name == 'psp':
        class_name = PSPHead.__name__
        n_in_ch = 4
        # https://arxiv.org/abs/1612.01105
        dec_head_cfg = dict(pool_scales=(1, 2, 3, 6),
                                num_classes=num_classses,  # output channels
                                dropout_ratio=0.1,
                                conv_cfg=dict(type='Conv2d'), # None = conv2d
                                norm_cfg=dict(type='BN'),
                                act_cfg=dict(type='ReLU'),
                                in_index=[i for i in range(n_in_ch)],
                                input_transform='resize_concat',
                                init_cfg=dict(
                                    type='Normal', std=0.01, override=dict(name='conv_seg')))
    elif dec_name == 'da':
        class_name = DAHead.__name__
        n_in_ch = 4
        # https://arxiv.org/abs/1809.02983
        dec_head_cfg = dict(num_classes=num_classses,  # output channels
                            dropout_ratio=0.1,
                            conv_cfg=dict(type='Conv2d'), # None = conv2d
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU'),
                            in_index=[i for i in range(n_in_ch)],
                            input_transform='resize_concat',
                            init_cfg=dict(
                                type='Normal', std=0.01, override=dict(name='conv_seg')))
        
    elif dec_name == 'segformer':
        class_name = SegformerHead.__name__
        n_in_ch = 4
        # https://arxiv.org/abs/2105.15203
        dec_head_cfg = dict(interpolate_mode='bilinear',
                            num_classes=num_classses,  # output channels
                            dropout_ratio=0.1,
                            conv_cfg=dict(type='Conv2d'), # None = conv2d
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU'),
                            in_index=[i for i in range(n_in_ch)],
                            init_cfg=dict(
                                type='Normal', std=0.01, override=dict(name='conv_seg')))
        
    elif dec_name == 'resnet':
        class_name = ResNetHead.__name__
        n_in_ch = 4
        # ResNet-like with recurrent convs
        dec_head_cfg = dict(num_classses=num_classses,
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
                            in_channels_red=384*n_in_ch)
    elif dec_name == 'unet':
        class_name = UNetHead.__name__
        n_in_ch=5
        # https://arxiv.org/abs/1505.04597 (unet papaer)
        input_group_cat_nb = 2
        n_in_ch *= input_group_cat_nb
        dec_head_cfg = dict(num_classses=num_classses,
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
                            in_channels_red=576)  # 576  |  384*input_group_cat_nb
        
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
                        bg_ch_to_rm=ignore_idx_loss, # not removing results in better segmentation
                        reduction='mean',
                        epsilon=epsilon,
                        k=k,
                        weight=weight)

    # CE-Dice Loss
    loss_cfg_dice_ce=dict(loss1=dict(name='CE',
                                    params=loss_cfg_ce),
                        loss2=dict(name='Dice', 
                                    params=loss_cfg_dice),
                        comp_rat=0.5)

    # Focal Loss
    loss_cfg_focal = dict(bg_ch_to_rm=ignore_idx_loss,
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
        return dict(name=DiceLoss.__name__, params=loss_cfg_dice)
        
    if loss_key=='dice_ce':
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
                            weight_decay = 0.5e-4,   # 0.5e-4  | 1e-2
                            betas = (0.9, 0.999)))
    return optm_cfg


def get_scheduler_cfg(nb_epochs):
    warmup_iters = max(1, int(nb_epochs*0.2))  # try *0.25
    
    scheduler_configs = []
    scheduler_configs.append(\
        dict(name='LinearLR',
            params=dict(start_factor=1/3, end_factor=1.0, total_iters=warmup_iters)))
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
    # if model_type==ModelType.SEGMENTOR:
    #     pass
    # elif model_type==ModelType.UNET:
    #     pass
    # else:
    #     ValueError(f'Unknown model type {model_type}')
    return 0.5e-4 


def get_lit_segmentor_cfg(batch_sz, nb_epochs, loss_cfg_key, dataset_attrs, gpus, model_type, 
                          dom_gen_tst=False, **kwargs):

    
    if model_type==ModelType.SEGMENTOR:
        lr = get_lr(model_type=model_type)
        
        # Backbone config
        bb_cfg = get_bb_cfg(bb_name=kwargs['backbone'], bb_size=kwargs['backbone_sz'], train_bb=kwargs['train_backbone'], 
                            dec_name=kwargs['dec_head_key'], main_pth=kwargs['main_pth'], pretrained=True)
        # Decoder config
        dec_head_cfg = get_dec_cfg(dec_name=kwargs['dec_head_key'], dataset_attrs=dataset_attrs)
        
        segmentor_cfg = dict(name=Segmentor.__name__,
                         params=dict(backbone=bb_cfg,
                                     decode_head=dec_head_cfg,
                                     reshape_dec_oup=True,
                                     align_corners=False))
        
    elif model_type==ModelType.UNET:
        lr = get_lr(model_type=model_type)
        
        segmentor_cfg = dict(name=UNet.__name__,
                         params=dict(n_channels=3, 
                                     n_classes=dataset_attrs['num_classses'], 
                                     bilinear=False))
        
    else:
        ValueError(f'Unknown model type {model_type}')
        
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
    num_classses = data_attr['num_classses']
    
    data_root_pth = data_root_pth / data_path_suffix

    if data_attr['format']=='png':
        train_fld = 'train-filtered' if 'hcp' in dataset else 'train'

        train_dataset = SegmentationDataset(img_dir=data_root_pth/'images'/train_fld,
                                            mask_dir=data_root_pth/'labels'/train_fld,
                                            num_classes=num_classses,
                                            file_extension='.png',
                                            mask_suffix='_labelTrainIds',
                                            augmentations=train_procs,
                                            )
        val_dataset = SegmentationDataset(img_dir=data_root_pth/'images/val',
                                        mask_dir=data_root_pth/'labels/val',
                                        num_classes=num_classses,
                                        file_extension='.png',
                                        mask_suffix='_labelTrainIds',
                                        augmentations=val_test_procs,
                                        )
        test_dataset = SegmentationDataset(img_dir=data_root_pth/'images/test',
                                        mask_dir=data_root_pth/'labels/test',
                                        num_classes=num_classses,
                                        file_extension='.png',
                                        mask_suffix='_labelTrainIds',
                                        augmentations=val_test_procs,
                                        ret_n_xyz=True)
        
    elif data_attr['format']=='hdf5':
        hdf5_train_name = data_attr['hdf5_train_name']
        hdf5_val_name = data_attr['hdf5_val_name']
        hdf5_test_name = data_attr['hdf5_test_name']
        
        train_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_train_name, 
                                                num_classes=num_classses, 
                                                augmentations=train_procs,
                                                ret_n_xyz=False)
        val_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_val_name, 
                                                num_classes=num_classses, 
                                                augmentations=val_test_procs,
                                                ret_n_xyz=True)
        test_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_test_name, 
                                                num_classes=num_classses, 
                                                augmentations=val_test_procs,
                                                ret_n_xyz=True)
    else:
        ValueError(f'Unsupported data format {data_attr["format"]}')
        
    return train_dataset, val_dataset, test_dataset
    



    
    
        