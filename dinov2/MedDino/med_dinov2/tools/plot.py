import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt



def show_result(img,
                seg,
                seg_gt=None,
                palette=None,
                classes=None,
                # win_name='',
                show=False,
                # wait_time=0,
                out_file=None,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    
    # Read the image [H, W, C]
    if isinstance(img, str):
        img = cv2.imread(img) #BGR
    elif isinstance(img, torch.Tensor):
        # [C, H, W] -> [H, W, C] 
        img = img.permute([1, 2, 0]).detach().cpu().numpy()
    else:
        assert isinstance(img, np.ndarray)
        img = img.copy()
        
    img = img[..., ::-1] #BGR -> RGB
    
    if isinstance(seg, torch.Tensor):
        # [num_cls, H, W] -> [H, W, num_cls] 
        seg = seg.permute([1, 2, 0]).detach().cpu().numpy()
    else:
        assert isinstance(seg, np.ndarray)
        seg = seg.copy()
    
    if seg_gt is not None:
        if isinstance(seg_gt, torch.Tensor):
            # [num_cls, H, W] -> [H, W, num_cls] 
            seg_gt = seg_gt.permute([1, 2, 0]).detach().cpu().numpy()
        else:
            assert isinstance(seg_gt, np.ndarray)
            seg_gt = seg_gt.copy()
        
        seg_gt = seg_gt.argmax(axis=-1)  # [H, W]
        
    seg = seg.argmax(axis=-1)  # [H, W]
    
    if palette is None:
        assert classes is not None
        # Get random state before set seed,
        # and restore random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(classes, 3))
        np.random.set_state(state)

    palette = np.array(palette)
    assert palette.shape[0] == classes
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
          
    img_seg = img * (1 - opacity) + color_seg * opacity
    img_seg = img_seg.astype(np.uint8)
    
    
    if seg_gt is not None:
        color_seg_gt = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg_gt[seg_gt == label, :] = color

        img_seg_gt = img * (1 - opacity) + color_seg_gt * opacity
        img_seg_gt = img_seg_gt.astype(np.uint8)
        
        img_res = np.concatenate([img_seg, img_seg_gt], axis=1)
    else:
        img_res = img_seg
    
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        plt.figure()
        plt.imshow(img_res)
        plt.show()
    if out_file is not None:
        plt.figure()
        plt.imshow(img_res)
        plt.savefig(out_file)
    return img
