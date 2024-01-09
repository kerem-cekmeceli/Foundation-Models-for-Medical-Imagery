import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os


def show_anns(anns):
    """Plots the computed masks on the image with random colors statring from the largest 
    """
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    

# Load example image    
os.chdir(os.path.join(os.getcwd(), 'segment-anything/my_scripts'))
                      
image = cv2.imread('../notebooks/images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the SAM checkpoint
sam_checkpoint = "../sam_checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Instantiate the auto mask generator object
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate masks with default settings
masks = mask_generator.generate(image)

# Show the masks
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('oup_masked/masked1.png',
            bbox_inches='tight')


# Generate masks with tuned settings:
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

masks2 = mask_generator_2.generate(image)

# Show the masks
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.savefig('oup_masked/masked2.png',
            bbox_inches='tight')
