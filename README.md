# FoundationModels
Training and Tuning Strategies for Foundation Models in Medical Imaging

# Checkpoints for the Foundation Models
Checkpoints folder with the below structure is expected to load the weights for the foundaiton models

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
