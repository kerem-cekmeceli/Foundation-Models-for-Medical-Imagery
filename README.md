# Modular Segmentation Framework

A Modular Segmentation Framework is developed for training and testing medical segmentation models compatible with a range of common input data formats. The framework supports encoder-decoder architectures with various backbones and projection heads, or standalone models for predicting segmentation masks. Loss functions and optimizers are switchable, while metrics such as Dice Similarity Coefficient (DSC) and mean Intersection over Union (mIoU) are automatically computed per slice and over the volume. These metrics, along with sample segmentation masks from the validation set, are logged during training for quick assessment of performance.

All architectural details can be found in the final report "Training and Tuning Strategies for Foundation Models in Medical Imaging".

Code for the developed framework can be found under `MedicalSegmentation` directory, `train.py` is the main file to run. Original code for the supported foundation models are under `OrigModels` directory for each backbone respectively. Code for backbone or fine-tune specific implementations or adjustments required for the framewoek are under the `ModelSpecific` directory.

# Framework Support

Below listed parameters must be set in `train.py` under `MedicalSegmentation` directory.

Supported Stand Alone Benchmark Models (trained from scratch) along with the value to set for `model_type` parameter:
* UNet `ModelType.UNET`
* Swin UNet `ModelType.SWINUNET`

Supported Backbones For Encoder/Decoder Type (supporting all available sizes for all above-listed backbones):
The following must be set `model_type=ModelType.SEGMENTOR` along with the `backbone` variable as listed.
* Dino (both registered and not-registered) `dino` or `dinoReg`
* SAM `sam`
* MedSAM `medsam`
* MAE `mae`
* ResNET `resnet`

Supported Fine-Tunings For Backbones:
`fine_tune` must be set to the below values with `train_backbone=False`
* Freeze (No Fine Tune) `''`
* Ladder Fine-Tuning (RN34 or Dino-Small) 'ladderR' or `ladderD`
* Reins and Reins LoRA `'rein'` or `'reinL'`
* Full Fine-Tuning `train_backbone=True` and `fine_tune=''`

Implemented Decoders:
Set `dec_head_key` to the below values
* Linear `'lin'`
* ResNet-Type `'resnet'`
* UNet-Type `'unet'` or `'unetS'` for smaller size
* DA Head (MMSEG) `'da'`
* SegFormer Head (MMSEG) `'segformer'`
* FCN Head (MMSEG) `'fcn'`
* PSP Head (MMSEG) `'psp'`
* SAM Prompt Encoder and Mask Decoder `'sam_mask_dec'`
* HQSAM Head `'hqsam'`
* HSAM Head `'hsam'`
* HQHSAM Head `'hqhsam'`

Supported Domain Adaptation Methods:
* Entropy Minimization `ftta=True`
* Self-Training `self_training=True`

Supported Data Formats:
* NifTI
* HDF5
* PNG (requires uniform volume depth for validation and test sets)

Supported Datasets:
Can be chosen by setting the `dataset` variable
* Brain:
  - HCP (T1w and T2w) - HDF5 `'hcp1'` ot `'hcp2'`
  - ABIDE (Caltech and Stanford) - HDF5 `'abide_caltech'` or `'abide_stanford'`
    
* Lumbar Spine:
  - VerSe - PNG `'spine_verse'`
  - MrSegV - PNG `'spine_mrspinesegv'`

* Prostate:
  - NCI - HDF5 `'prostate_nci'`
  - PiradErc USZ dataset - HDF5 `'prostate_usz'`
 
* Brain Tumor:
  - BraTS (T1 and FLAIR) - NifTI `'BraTS_T1'` or `'BraTS_FLAIR'`


# Checkpoints for the Foundation Models

Checkpoints folder with the below structure and data is expected to load the weights for the foundaiton models. Files can be downloaded from respective repositories for each backbone.

* [DinoV2](https://github.com/facebookresearch/dinov2)
* [SAM](https://github.com/facebookresearch/segment-anything)
* [MedSam](https://github.com/bowang-lab/MedSAM)
* [MAE](https://github.com/facebookresearch/mae)

```
  Checkpoints
  └── Orig
      └── backbone
          ├── DinoV2
          │   ├── dinov2_vitb14_pretrain.pth
          │   ├── dinov2_vitb14_reg4_pretrain.pth
          │   ├── dinov2_vitg14_pretrain.pth
          │   ├── dinov2_vitg14_reg4_pretrain.pth
          │   ├── dinov2_vitl14_pretrain.pth
          │   ├── dinov2_vitl14_reg4_pretrain.pth
          │   ├── dinov2_vits14_pretrain.pth
          │   └── dinov2_vits14_reg4_pretrain.pth
          ├── MAE
          │   ├── mae_pretrain_vit_base.pth
          │   ├── mae_pretrain_vit_huge.pth
          │   └── mae_pretrain_vit_large.pth
          ├── MedSam
          │   └── medsam_vit_b.pth
          └── SAM
              ├── sam_vit_b_01ec64.pth
              ├── sam_vit_h_4b8939.pth
              └── sam_vit_l_0b3195.pth
```

# Training and Testing



