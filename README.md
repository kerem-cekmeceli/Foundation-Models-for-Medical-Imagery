# Modular Segmentation Framework

A Modular Segmentation Framework is developed for training and testing medical segmentation models compatible with a range of common input data formats. The framework supports encoder-decoder architectures with various backbones and projection heads, or standalone models for predicting segmentation masks. Loss functions and optimizers are switchable, while metrics such as Dice Similarity Coefficient (DSC) and mean Intersection over Union (mIoU) are automatically computed per slice and over the volume. These metrics, along with sample segmentation masks from the validation set, are logged during training for quick assessment of performance.

All architectural details can be found in the final report "Training and Tuning Strategies for Foundation Models in Medical Imaging".

# Framework Support

Supported Stand Alone Benchmark Models (trained from scratch):
* UNet
* Swin UNet

Supported Backbones For Encoder/Decoder Type (supporting all available sizes for all above-listed backbones):
* Dino (both registered and not-registered)
* SAM
* MedSAM
* MAE
* ResNET

Supported Fine-Tunings For Backbones:
* Freeze (No Fine Tune)
* Ladder Fine-Tuning (RN34 or Dino-Small)
* Reins and Reins LoRA
* Full Fine-Tuning

Implemented Decoders:
* Linear
* ResNet-Type
* UNet-Type
* DA Head
* SegFormer Head
* SAM Prompt Encoder and Mask Decoder
* HQSAM Head
* HSAM Head
* HQHSAM Head

Supported Domain Adaptation Methods:
* Entropy Minimization
* Self-Training

Supported Data Formats:
* NifTI
* HDF5
* PNG (requires uniform volume depth for validation and test sets)

Supported Datasets:
* Brain:
  - HCP (T1w and T2w)
  - ABIDE (Caltech and Stanford)
    
* Lumbar Spine:
  - VerSe
  - MrSegV

* Prostate:
  - NCI
  - PiradErc USZ dataset
 
* Brain Tumor:
  - BraTS (T1 and FLAIR)


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

# Training 

