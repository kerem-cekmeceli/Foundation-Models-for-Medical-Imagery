# Modular Segmentation Framework

A Modular Segmentation Framework is developed for training and testing medical segmentation models compatible with a range of common input data formats. The framework supports encoder-decoder architectures with various backbones and projection heads, or standalone models for predicting segmentation masks. Loss functions and optimizers are switchable, while metrics such as Dice Similarity Coefficient (DSC) and mean Intersection over Union (mIoU) are automatically computed per slice and over the volume. These metrics, along with sample segmentation masks from the validation set, are logged during training for quick assessment of performance.

All architectural details can be found in the final [report](https://github.com/kerem-cekmeceli/FoundationModels/blob/main/report.pdf) "Training and Tuning Strategies for Foundation Models in Medical Imaging".

Code for the developed framework can be found under the `MedicalSegmentation` directory, with `train.py` as the main file to run. Original code for the supported foundation models is under the `OrigModels` directory for each backbone respectively. Code for backbone or fine-tune specific implementations or adjustments required for the framework is under the `ModelSpecific` directory.

# Framework Support

The parameters listed below must be set in [`train.py`](https://github.com/kerem-cekmeceli/FoundationModels/blob/main/MedicalSegmentation/train.py) under the `MedicalSegmentation` directory.

Supported Standalone Benchmark Models (trained from scratch) along with the value to set for the `model_type` parameter:
* UNet: `ModelType.UNET`
* Swin UNet: `ModelType.SWINUNET`

Supported Backbones for Encoder/Decoder Type (supporting all available sizes for all above-listed backbones):
The following must be set `model_type=ModelType.SEGMENTOR` along with the `backbone` variable as listed:
* Dino (both registered and not-registered): `dino` or `dinoReg`
* SAM: `sam`
* MedSAM: `medsam`
* MAE: `mae`
* ResNET: `resnet`

Supported Fine-Tunings for Backbones:
`fine_tune` must be set to the below values with `train_backbone=False`:
* Freeze (No Fine Tune): `''`
* Ladder Fine-Tuning (ResNet34 or Dino-Small): `ladderR` or `ladderD`
* Reins and Reins LoRA: `rein` or `reinL`
* Full Fine-Tuning: `train_backbone=True` and `fine_tune=''`

Implemented Decoders:
Set `dec_head_key` to the below values:
* Linear: `lin`
* ResNet-Type: `resnet`
* UNet-Type: `unet` or `unetS` for smaller size
* DA Head (MMSEG): `da`
* SegFormer Head (MMSEG): `segformer`
* FCN Head (MMSEG): `fcn`
* PSP Head (MMSEG): `psp`
* SAM Prompt Encoder and Mask Decoder: `sam_mask_dec`
* HQSAM Head: `hqsam`
* HSAM Head: `hsam`
* HQHSAM Head: `hqhsam`

Supported Domain Adaptation Methods:
* Entropy Minimization: `ftta=True`
* Self-Training: `self_training=True`

Supported Data Formats:
* NifTI
* HDF5
* PNG (requires uniform volume depth for validation and test sets)

Supported Datasets:
Can be chosen by setting the `dataset` variable:
* Brain:
  - HCP (T1w and T2w) - HDF5: `hcp1` or `hcp2`
  - ABIDE (Caltech and Stanford) - HDF5: `abide_caltech` or `abide_stanford`
* Lumbar Spine:
  - VerSe - PNG: `spine_verse`
  - MrSegV - PNG: `spine_mrspinesegv`
* Prostate:
  - NCI - HDF5: `prostate_nci`
  - PiradErc USZ dataset - HDF5: `prostate_usz`
* Brain Tumor:
  - BraTS (T1 and FLAIR) - NifTI: `BraTS_T1` or `BraTS_FLAIR`

# Checkpoints for the Foundation Models

Checkpoints folder with the below structure and data is expected to load the weights for the foundation models. Files can be downloaded from the respective repositories for each backbone.

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

Running `train.py`, models are trained on the source domain and tested both in-domain and on all available datasets for the anatomy for domain generalization/domain adaptation. Dice and mIoU scores are computed over the volume for the validation and test sets, while they are computed per slice to serve as indicators during the training process. The WandB logger is used. Sample segmentation results from the validation (at defined intervals during training) and test sets (at the end) are logged. The model with the highest validation DSC over the volume is used for testing.

# Saving and Loading Models

`ckp_pth` must be set to the path where the trained models will be saved. A checkpoints directory will be created (if it doesn't already exist) with the following folder structure: `DATASET_NAME/FINETUNE_BACKBONE_NAME/DECODER_NAME`, and checkpoints are saved with timestamps and epoch numbers inside. Self-training runs are saved with the same folder structure under the `ST` folder inside the `Checkpoints` directory.

For domain adaptation, since checkpoints trained on the source domain must be loaded, `search_dir_` must be set to the path where the same folder structure as above exists and where the checkpoints are stored.

