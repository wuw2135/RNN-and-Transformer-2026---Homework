"""
Copied and modified from
https://github.com/NVIDIA/pix2pixHD/tree/master/util
"""

from __future__ import print_function
import os
import torch
import numpy as np
from PIL import Image
from torchvision import utils

DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],  # 0: background
        [255, 0, 0],  # 1
        [0, 255, 0],  # 2
        [0, 0, 255],  # 3
        [255, 255, 0],  # 4
        [255, 0, 255],  # 5
        [0, 255, 255],  # 6
        [128, 0, 0],  # 7
        [0, 128, 0],  # 8
        [0, 0, 128],  # 9
        [128, 128, 0],  # 10
        [128, 0, 128],  # 11
        [0, 128, 128],  # 12
        [192, 192, 192],  # 13
    ],
    dtype=np.uint8,
)


def logits_batch_to_rgb(
    logits: torch.Tensor,  # (N, C=14, H, W) 或 (C=14, H, W)
    palette: np.ndarray = DEFAULT_PALETTE,
    return_numpy: bool = True,
):
    indices = logits.to(torch.long)  # (N, H, W)

    palette_t = torch.as_tensor(
        palette, dtype=torch.uint8, device=indices.device
    )  # (C,3)
    rgb = palette_t[indices]  # (N, H, W, 3), uint8
    rgb = rgb.permute(0, 3, 1, 2)

    if return_numpy:
        return rgb.cpu().numpy()
    else:
        return rgb


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(np.array(image_numpy, dtype=np.uint8))
    image_pil.save(image_path)


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, "fuse"):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.Conv2d):
            child.bias = torch.nn.Parameter(torch.zeros(child.weight.size(0)))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data)
    elif isinstance(tensor_data, torch.Tensor):
        tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)

    return vis


def max_min_normalize(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def de_norm(tensor_data, mean, std):
    tensor_data = tensor_data * std + mean
    return max_min_normalize(tensor_data)


def ema_alpha_update(
    val_result: dict,
    ema_prev: np.ndarray,
    momentum: float = 0.9,
    p: float = 1.0,
    clamp: tuple = (0.5, 5.0),
    eps: float = 1e-3,
):
    """
    return: alpha_np   : np.ndarray, (C,)
    """
    C = ema_prev.shape[0]

    now_iou = np.zeros(C)  # (C,)
    for i in range(C):
        now_iou[i] = max(float(val_result[f"iou_{i}"]), 0.2)

    alpha_target = np.power(np.max(now_iou) - now_iou, p) + eps  # (C,)
    alpha_target /= alpha_target.mean() + eps

    # --- EMA smooth ---
    alpha = momentum * ema_prev + (1.0 - momentum) * alpha_target

    if clamp is not None:
        alpha = np.clip(alpha, clamp[0], clamp[1])

    return alpha.astype(np.float32)
