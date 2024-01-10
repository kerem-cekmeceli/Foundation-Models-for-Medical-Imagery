import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot

import dinov2.eval.segmentation.models
import os
from matplotlib import pyplot as plt



class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_segmenter(cfg, backbone_model, seg_checkpoint=None):
    model = init_segmentor(cfg, checkpoint=seg_checkpoint)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    if seg_checkpoint is None:
        model.init_weights()
    return model


# Load the pre-trained backbone
BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

# os.chdir(os.path.join(os.getcwd(), 'dinov2'))

local_backbone_cp = True

if not local_backbone_cp:
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
else:
    import os
    print(os.getcwd())
    backbone_model = torch.hub.load('./', backbone_name, source='local', pretrained=False)

    # Define the path to your local .pth checkpoint file
    checkpoint_path = './checkpoints_backbone/dinov2_vits14_pretrain.pth'

    # Load the checkpoint using torch.load
    checkpoint = torch.load(checkpoint_path)

    # If your model is saved as part of the checkpoint
    if 'model_state_dict' in checkpoint:
        backbone_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the entire model is saved (including optimizer, etc.)
        backbone_model.load_state_dict(checkpoint)

backbone_model.eval()
backbone_model.cuda()

# Load the pre-trained segmentation head Linear Boosted (+ms)
import urllib
import mmcv
from mmcv.runner import load_checkpoint


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
HEAD_TYPE = "ms" # in ("ms, "linear")


local_seg_model_cfg = False
# Get the conf file for the segmentation head
if not local_seg_model_cfg:
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    
else:
    cfg = './' #@TODO

    
if HEAD_TYPE == "ms":
    cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
    print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

local_seg_head_cp = True
# Load the checkpoint to the segmentation head
if not local_seg_head_cp:
    # Instantiate the empty segmentation head for the selected backbone
    model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
else:
    # load_checkpoint(model, './checkpoints_segHead/dinov2_vits14_voc2012_ms_head.pth', map_location="cpu")
    cp_seg = './checkpoints_segHead/dinov2_vits14_voc2012_ms_head.pth'
    model = create_segmenter(cfg, backbone_model=backbone_model, seg_checkpoint=cp_seg)
    
model.cuda()
model.eval()


# Load the sample image
import urllib
from PIL import Image


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


# EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"

# image = load_image_from_url(EXAMPLE_IMAGE_URL)
    
import cv2
image = cv2.imread('/usr/bmicnas02/data-biwi-01/foundation_models/da_data/brain/hcp2/images/test/0050.png')



plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.savefig('oup_imgs/orig.png',
            bbox_inches='tight')

# Semantic segmentation with Boosted Linear head (+ms)
import numpy as np
import dinov2.eval.segmentation.utils.colormaps as colormaps


DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}


def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    # segmentation_values = colormap_array[segmentation_logits]
    return Image.fromarray(segmentation_values)

array = np.array(image)
# print(f"shape : {array.shape}")
if len(np.array(image).shape)>2:
    array = array[:, :, ::-1] # BGR
else:
    array = array[..., np.newaxis]
    
segmentation_logits = inference_segmentor(model, array)[0]
# segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)

# Save the res of MS seg head
show_result_pyplot(model=model, img=array, 
                #    palette=DATASET_COLORMAPS[HEAD_DATASET], 
                   result=[segmentation_logits], 
                   out_file=f'./oup_imgs/seg_out_{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.png',
                   block=False)

# Load the pre-trained backbone with ViT adapter + mask2former seg head
import dinov2.eval.segmentation_m2f.models.segmentors


BACKBONE_SIZE = "giant"
HEAD_DATASET = "ade20k" 
HEAD_TYPE = "m2f" 

backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

local_seg_model_cfg = False
# Get the conf file for the segmentation model
if not local_seg_model_cfg:
    CONFIG_URL = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    cfg_str = load_config_from_url(CONFIG_URL)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    
else:
    cfg = './' # @TODO path to config


local_seg_model = True
# Load the checkpoint to the segmentation model
if not local_seg_model:
    # Instantiate the empty segmentation model
    model = init_segmentor(cfg)
    CHECKPOINT_URL = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.pth"    
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
else:
    cp = f'./checkpoints_seg_model_mask2former/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.pth'
    model = init_segmentor(cfg, checkpoint=None)
    checkpoint = load_checkpoint(model, cp, map_location="cpu")
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    # model = init_segmentor(cfg, checkpoint=cp)
    
    
model.cuda()
model.eval()

array = np.array(image)[:, :, ::-1] # BGR
segmentation_logits = inference_segmentor(model, array)[0]
# segmented_image = render_segmentation(segmentation_logits, "ade20k")

# Save the res of m2f seg head with dino plugged into a ViT adapter
show_result_pyplot(model=model, img=array, result=[segmentation_logits], 
                   out_file=f'./oup_imgs/seg_out_{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}.png',
                   block=False)
