# from prep_model import get_backone_patch_embed_sizes
from MedDino.med_dinov2.layers.segmentation import ConvHeadLinear, ResNetHead, UNetHead
from mmseg.models.decode_heads import FCNHead, PSPHead, DAHead, SegformerHead
import torch
from MedDino.med_dinov2.eval.losses import DiceLoss, FocalLoss, CompositionLoss
from torch.nn import CrossEntropyLoss
from MedDino.med_dinov2.data.datasets import SegmentationDataset, SegmentationDatasetHDF5
from prep_model import get_bb_name
from MedDino.med_dinov2.layers.backbone_wrapper import DinoBackBone

def get_data_attrs(name:str, use_hdf5=True):
    attrs = {}
    
    # Dataset parameters
    
    # Brain - HCP1
    if name=='hcp1':
        attrs['name'] = name
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/hcp1'
        else:
            attrs['data_path_suffix'] = 'brain/hcp'
            attrs['hdf5_train_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5'
            attrs['hdf5_val_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5'
            attrs['hdf5_test_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 256  # First val volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Brain - HCP2    
    elif name=='hcp2':
        attrs['name'] = name
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/hcp2'
        else:
            attrs['data_path_suffix'] = 'brain/hcp'
            attrs['hdf5_train_name'] = 'data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5'
            attrs['hdf5_val_name'] = 'data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5'
            attrs['hdf5_test_name'] = 'data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 256  # First val volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Brain - Abide-Caltech    
    elif name=='abide_caltech':
        attrs['name'] = name
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/abide_caltech'
            ValueError()
        else:
            attrs['data_path_suffix'] = 'brain/abide/caltech'
            attrs['hdf5_train_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_0.7_from_0_to_10.hdf5'
            attrs['hdf5_val_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_0.7_from_10_to_15.hdf5'
            attrs['hdf5_test_name'] = 'data_T1_2d_size_256_256_depth_256_res_0.7_0.7_0.7_from_16_to_36.hdf5'
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 256  # First val volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    elif name=='abide_stanford':
        attrs['name'] = name
        if not use_hdf5:
            attrs['data_path_suffix'] = 'brain/abide_stanford'
        else:
            attrs['data_path_suffix'] = 'brain/abide/stanford'
            attrs['hdf5_train_name'] = 'data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_0_to_10.hdf5'
            attrs['hdf5_val_name'] = 'data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_10_to_15.hdf5'
            attrs['hdf5_test_name'] = 'data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_16_to_36.hdf5'
        attrs['num_classses'] = 15
        attrs['vol_depth'] = 132  # First val volume depth (all the same)
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Prostate - NCI    
    elif name=='prostate_nci':
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['name'] = name
        attrs['num_classses'] = 3
        attrs['vol_depth'] = 15  # First val volume depth
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        if use_hdf5:
            attrs['data_path_suffix'] = 'nci'
            attrs['hdf5_train_name'] = 'train.hdf5'
            attrs['hdf5_val_name'] = 'val.hdf5'
            attrs['hdf5_test_name'] = 'test.hdf5'
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Prostate - USZ        
    elif name=='prostate_usz':
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['name'] = name
        attrs['data_path_suffix'] = 'prostate/pirad_erc'
        attrs['num_classses'] = 3
        attrs['vol_depth'] = 22  # First val volume depth
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
            
        if use_hdf5:
            attrs['data_path_suffix'] = 'nci'
            attrs['hdf5_train_name'] = 'data_2d_from_40_to_68_size_256_256_res_0.625_0.625_ek.hdf5'
            attrs['hdf5_val_name'] = 'data_2d_from_20_to_40_size_256_256_res_0.625_0.625_ek.hdf5'
            attrs['hdf5_test_name'] = 'data_2d_from_0_to_20_size_256_256_res_0.625_0.625_ek.hdf5'
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
    
    # Cardiac - ACDC      
    elif name=='cardiac_acdc':
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['name'] = name
        attrs['data_path_suffix'] = 'cardiac/acdc'
        attrs['num_classses'] = 3
        attrs['vol_depth'] = 10  # First val volume depth
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        if use_hdf5:
            attrs['data_path_suffix'] = 'nci'
            attrs['hdf5_train_name'] = 'train.hdf5'
            attrs['hdf5_val_name'] = 'val.hdf5'
            attrs['hdf5_test_name'] = 'test.hdf5'   
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight  
    
    # Cardiac - RVSC           
    elif name=='cardiac_rvsc':
        if not use_hdf5:
            ValueError('only HDF5 is supported')
            
        attrs['name'] = name
        attrs['data_path_suffix'] = 'cardiac/rvsc'
        attrs['num_classses'] = 2
        attrs['vol_depth'] = 10  # First val volume depth
        attrs['ignore_idx_loss'] = None
        attrs['ignore_idx_metric'] = 0
        if use_hdf5:
            attrs['data_path_suffix'] = 'nci'
            attrs['hdf5_train_name'] = 'train.hdf5'
            attrs['hdf5_val_name'] = 'val.hdf5'
            attrs['hdf5_test_name'] = 'test.hdf5'
            
        # weight = [0.1] + [1.]*(attrs['num_classses']-1)
        # weight = torch.Tensor(weight)
        weight = None
        
        attrs['weight'] = weight
        
    else:
        ValueError(f'Dataset: {name} is not defined')
        
    return attrs

def get_batch_sz(data_attrs, num_gpu):
    assert num_gpu>0
    
    dataset_name = data_attrs['name']
    
    if dataset_name=='hcp1':
        batch_sz = 8
        
    elif dataset_name=='hcp2':
        batch_sz = 8
        
    elif dataset_name=='abide_caltech':
        batch_sz = 8
        
    elif dataset_name=='abide_stanford':
        batch_sz = 6
       
    elif dataset_name=='cardiac_acdc':
       batch_sz = 6 #@TODO verify
            
    elif dataset_name=='cardiac_rvsc':
        batch_sz = 6 #@TODO verify
        
    elif dataset_name=='prostate_nci':
        batch_sz = 8 #@TODO verify #10
            
    elif dataset_name=='prostate_usz':
        batch_sz = 10 #@TODO verify
        
    else:
        ValueError(f'Dataset: {dataset_name} is not defined')
    
    batch_sz = batch_sz // num_gpu     
    
    vol_depth = data_attrs['vol_depth']
    # assert vol_depth % batch_sz == 0,\
    #     f'batch size must be a multiple of slice/patient but got {batch_sz} and {vol_depth}'
       
    return batch_sz


def get_bb_cfg(bb_name, bb_size, train_bb, dec_name, main_pth):
    if bb_name == 'dino':
        if dec_name=='unet':
            n_out = 5*2
        else:
            n_out = 4
            
        last_out_first = True
        
        assert bb_size in ["small", "base", "large", "giant"]
        backbone_name = get_bb_name(bb_size)
        bb_checkpoint_path = main_pth/f'Checkpoints/Orig/backbone/{backbone_name}_pretrain.pth'
        
        name = DinoBackBone.__name__
        params = dict(nb_outs=n_out,
                      name=backbone_name,
                      last_out_first=last_out_first,
                      bb_model=None,
                      cfg=dict(backbone_name=backbone_name, backbone_cp=bb_checkpoint_path),
                      train=train_bb)
        
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
        
    return dict(name=class_name, params=dec_head_cfg), n_in_ch


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
        
        

def get_metric_cfgs(data_attr):
    ignore_idx_metric = data_attr['ignore_idx_metric']
    
    epsilon = 1  # smoothing factor 
    k=1  # power

    miou_cfg=dict(prob_inputs=False, # Decoder does not return probas explicitly
                soft=False,
                bg_ch_to_rm=ignore_idx_metric,  # bg channel to be removed 
                reduction='mean',
                EN_vol_scores=True,
                epsilon=epsilon)

    dice_cfg=dict(prob_inputs=False,  # Decoder does not return probas explicitly
                soft=False,
                bg_ch_to_rm=ignore_idx_metric,
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
    seg_log_per_batch = 4  # Log N samples from each minibatch
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
    vol_depth = data_attr['vol_depth']
    
    seg_res_nb_vols = 1  # Process minibatches for N number of 3D volumes
    seg_log_nb_batches = 16
    step = max(vol_depth//batch_sz//seg_log_nb_batches, 1)
    seg_log_batch_idxs = torch.arange(0+step-1, min(seg_log_nb_batches*step, vol_depth//batch_sz*seg_res_nb_vols), step).tolist()
    # assert len(seg_log_batch_idxs)==seg_log_nb_batches
    seg_log_batch_idxs = seg_log_batch_idxs[:seg_log_nb_batches]
    return seg_log_batch_idxs


def get_augmentations(patch_sz):
    
    # Define data augmentations
    img_scale_fac = 1  # Keep at 1 
    central_crop = True
    
    augmentations = []  # For val and test
    train_augmentations = []  # For train
    
    
    train_augmentations.append(dict(type='ElasticTransformation', data_aug_ratio=0.25))
    train_augmentations.append(dict(type='StructuralAug', data_aug_ratio=0.25))
    train_augmentations.append(dict(type='PhotoMetricDistortion'))

    if img_scale_fac > 1:
        augmentations.append(dict(type='Resize2',
                                scale_factor=float(img_scale_fac), #HW
                                keep_ratio=True))

    augmentations.append(dict(type='Normalize', 
                            mean=[123.675, 116.28, 103.53],  #RGB
                            std=[58.395, 57.12, 57.375],  #RGB
                            to_rgb=True))
    if central_crop:
        augmentations.append(dict(type='CentralCrop',  
                                size_divisor=patch_sz))
    else:
        augmentations.append(dict(type='CentralPad',  
                                size_divisor=patch_sz,
                                pad_val=0, seg_pad_val=0))
        
    train_augmentations = train_augmentations + augmentations
    
    return train_augmentations, augmentations


def get_datasets(data_root_pth, hdf5_data, data_attr, train_augmentations, augmentations):
    data_path_suffix = data_attr['data_path_suffix']
    dataset = data_attr['name']
    num_classses = data_attr['num_classses']
    
    data_root_pth = data_root_pth / data_path_suffix

    if not hdf5_data:
        train_fld = 'train-filtered' if 'hcp' in dataset else 'train'

        train_dataset = SegmentationDataset(img_dir=data_root_pth/'images'/train_fld,
                                            mask_dir=data_root_pth/'labels'/train_fld,
                                            num_classes=num_classses,
                                            file_extension='.png',
                                            mask_suffix='_labelTrainIds',
                                            augmentations=train_augmentations,
                                            )
        val_dataset = SegmentationDataset(img_dir=data_root_pth/'images/val',
                                        mask_dir=data_root_pth/'labels/val',
                                        num_classes=num_classses,
                                        file_extension='.png',
                                        mask_suffix='_labelTrainIds',
                                        augmentations=augmentations,
                                        )
        test_dataset = SegmentationDataset(img_dir=data_root_pth/'images/test',
                                        mask_dir=data_root_pth/'labels/test',
                                        num_classes=num_classses,
                                        file_extension='.png',
                                        mask_suffix='_labelTrainIds',
                                        augmentations=augmentations,
                                        ret_n_xyz=True)
        
    else:
        hdf5_train_name = data_attr['hdf5_train_name']
        hdf5_val_name = data_attr['hdf5_val_name']
        hdf5_test_name = data_attr['hdf5_test_name']
        
        train_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_train_name, 
                                                num_classes=num_classses, 
                                                augmentations=train_augmentations,
                                                ret_n_xyz=False)
        val_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_val_name, 
                                                num_classes=num_classses, 
                                                augmentations=augmentations,
                                                ret_n_xyz=True)
        test_dataset = SegmentationDatasetHDF5(file_pth=data_root_pth/hdf5_test_name, 
                                                num_classes=num_classses, 
                                                augmentations=augmentations,
                                                ret_n_xyz=True)
        
    return train_dataset, val_dataset, test_dataset
    



    
    
        