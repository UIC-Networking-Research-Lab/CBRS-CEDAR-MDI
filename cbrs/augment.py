"""
cbrs/augment.py

SigYOLO data augmentation pipeline for spectrogram YOLO training.

Mirrors ultralytics default augmentation (ultralytics/cfg/default.yaml)
except mosaic and auto_augment are excluded — they are not physically
meaningful for time-frequency spectrogram images.

Included:
  fliplr    : horizontal flip (frequency axis reflection)
  scale     : random scale — signal at different zoom levels
  translate : random shift  — signal at different time/freq positions
  hsv_h     : hue jitter    — minor colour channel variation
  hsv_s     : saturation jitter
  hsv_v     : brightness jitter — simulates SNR variation
  erasing   : random rectangle masking — occlusion robustness

Excluded:
  mosaic    : splices 4 images — creates false signal boundaries
  flipud    : vertical (freq) flip — physically meaningless
  mixup     : linear image blend
  auto_augment (RandAugment) : includes rotate/shear/perspective
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch
from torch import Tensor
import torchvision.transforms.functional as TF


class SigYOLOAugment:
    """
    Augmentation pipeline for SigYOLO spectrogram training.

    Default values match ultralytics/cfg/default.yaml for the included ops.
    Call on (img, labels) pairs inside Dataset.__getitem__.

    Parameters
    ----------
    fliplr    : horizontal flip probability          (ultralytics default 0.5)
    scale     : random scale range ± scale           (ultralytics default 0.5)
    translate : random translate range ± translate   (ultralytics default 0.1)
    hsv_h     : hue shift ± hsv_h (fraction of 360) (ultralytics default 0.015)
    hsv_s     : saturation multiplier range ± hsv_s  (ultralytics default 0.7)
    hsv_v     : brightness multiplier range ± hsv_v  (ultralytics default 0.4)
    erasing   : random erasing probability           (ultralytics default 0.4)

    Usage
    -----
    aug = SigYOLOAugment()
    img_t, lbl_t = aug(img_t, lbl_t)   # img [3,H,W], lbl [N,5] cls x1 y1 x2 y2
    """

    def __init__(
        self,
        fliplr:    float = 0.5,
        scale:     float = 0.5,
        translate: float = 0.1,
        hsv_h:     float = 0.015,
        hsv_s:     float = 0.7,
        hsv_v:     float = 0.4,
        erasing:   float = 0.4,
    ):
        self.fliplr    = fliplr
        self.scale     = scale
        self.translate = translate
        self.hsv_h     = hsv_h
        self.hsv_s     = hsv_s
        self.hsv_v     = hsv_v
        self.erasing   = erasing

    def __call__(
        self, img: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        img    : [3, H, W]  float32  [0, 1]
        labels : [N, 5]     float32  (cls, x1, y1, x2, y2) in pixels

        Returns
        -------
        Augmented (img, labels) with same dtypes.
        """
        img, labels = self._hsv(img, labels)
        img, labels = self._affine(img, labels)
        img, labels = self._fliplr(img, labels)
        img, labels = self._erase(img, labels)
        return img, labels

    # ── HSV colour augmentation ───────────────────────────────────────────────

    def _hsv(self, img: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Hue / Saturation / Brightness jitter.

        Mirrors ultralytics/utils/augmentations.py :: augment_hsv():
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
            img_hsv[..., i] = (img_hsv[..., i] * r[i]).clip(0, lim)
        """
        if self.hsv_v > 0:
            v = random.uniform(1.0 - self.hsv_v, 1.0 + self.hsv_v)
            img = TF.adjust_brightness(img, brightness_factor=max(0.0, v))
        if self.hsv_s > 0:
            s = random.uniform(1.0 - self.hsv_s, 1.0 + self.hsv_s)
            img = TF.adjust_saturation(img, saturation_factor=max(0.0, s))
        if self.hsv_h > 0:
            # TF.adjust_hue expects hue_factor in [-0.5, 0.5]
            # ultralytics hsv_h = 0.015 → shift ≈ ±5.4° out of 360°
            h = random.uniform(-self.hsv_h, self.hsv_h)
            img = TF.adjust_hue(img, hue_factor=h)
        return img, labels

    # ── Scale + Translate (affine) ────────────────────────────────────────────

    def _affine(self, img: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Random scale ± scale and translate ± translate * img_size.

        Mirrors ultralytics/utils/augmentations.py :: random_perspective()
        with degrees=0, shear=0, perspective=0 (geometry-safe for spectrograms).

        Bounding boxes are transformed with the same affine matrix and boxes
        that become too small (< 2 px in either axis) are dropped.
        """
        _, H, W = img.shape

        s  = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        tx = random.uniform(-self.translate, self.translate) * W
        ty = random.uniform(-self.translate, self.translate) * H

        img = TF.affine(
            img,
            angle=0,
            translate=[tx, ty],
            scale=s,
            shear=0,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        )

        if labels.shape[0] == 0:
            return img, labels

        # Affine maps point (x, y) to:
        #   x' = s*(x - W/2) + W/2 + tx
        #   y' = s*(y - H/2) + H/2 + ty
        cx, cy = W / 2.0, H / 2.0
        x1 = (labels[:, 1] - cx) * s + cx + tx
        y1 = (labels[:, 2] - cy) * s + cy + ty
        x2 = (labels[:, 3] - cx) * s + cx + tx
        y2 = (labels[:, 4] - cy) * s + cy + ty

        # Pre-clip extents for area-ratio check (mirrors box_candidates in ultralytics)
        un_w = (x2 - x1).clamp_min(0.0)
        un_h = (y2 - y1).clamp_min(0.0)

        x1 = x1.clamp(0.0, float(W))
        y1 = y1.clamp(0.0, float(H))
        x2 = x2.clamp(0.0, float(W))
        y2 = y2.clamp(0.0, float(H))

        new_w = x2 - x1
        new_h = y2 - y1
        # box_candidates: min 2 px, area retained > 10 %, aspect ratio < 100
        # ultralytics/utils/augmentations.py :: box_candidates
        ar    = (new_w / (new_h + 1e-16)).maximum(new_h / (new_w + 1e-16))
        valid = (
            (new_w > 2.0) & (new_h > 2.0)
            & (new_w * new_h / (un_w * un_h + 1e-16) > 0.1)
            & (ar < 100.0)
        )
        if not valid.any():
            return img, labels[:0]

        labels      = labels[valid].clone()
        labels[:, 1] = x1[valid]
        labels[:, 2] = y1[valid]
        labels[:, 3] = x2[valid]
        labels[:, 4] = y2[valid]
        return img, labels

    # ── Horizontal flip ───────────────────────────────────────────────────────

    def _fliplr(self, img: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Horizontal flip with probability fliplr.
        Mirrors ultralytics fliplr augmentation.
        """
        if random.random() >= self.fliplr:
            return img, labels
        _, H, W = img.shape
        img = TF.hflip(img)
        if labels.shape[0]:
            x1s         = labels[:, 1].clone()
            x2s         = labels[:, 3].clone()
            labels[:, 1] = float(W) - x2s
            labels[:, 3] = float(W) - x1s
        return img, labels

    # ── Random erasing ────────────────────────────────────────────────────────

    def _erase(self, img: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Randomly erase a rectangular region, filled with image mean.

        Mirrors ultralytics/utils/augmentations.py :: RandomErasing.
        Area ratio sampled from [0.02, 0.33], aspect ratio from [0.3, 3.3].
        """
        if random.random() >= self.erasing:
            return img, labels
        _, H, W = img.shape
        area = H * W
        fill = img.mean(dim=(1, 2), keepdim=True)   # per-channel mean
        for _ in range(100):                         # 100 attempts — matches ultralytics RandomErasing
            er     = random.uniform(0.02, 0.33)
            aspect = random.uniform(0.3, 3.3)
            rh     = int(round(math.sqrt(area * er / aspect)))
            rw     = int(round(math.sqrt(area * er * aspect)))
            if 0 < rh < H and 0 < rw < W:
                ry = random.randint(0, H - rh)
                rx = random.randint(0, W - rw)
                img = img.clone()
                img[:, ry:ry + rh, rx:rx + rw] = fill.expand(-1, rh, rw)
                break
        return img, labels


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# mirrors ultralytics/utils/torch_utils.py :: ModelEMA
# ─────────────────────────────────────────────────────────────────────────────

import copy
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average of model weights (shadow model).

    Maintains a smoothed copy of the model weights that is used for
    validation and checkpointing instead of the live training weights.

    Mirrors ultralytics/utils/torch_utils.py :: ModelEMA.

    Formula (per step):
        d = decay * (1 - exp(-updates / tau))   # warmup
        ema_w = d * ema_w + (1 - d) * model_w

    Parameters
    ----------
    model : the training model
    decay : target EMA decay (ultralytics default 0.9999)
    tau   : warmup steps  — decay ramps from 0 → target over ~tau steps
            (ultralytics default 2000)

    Usage
    -----
    ema = ModelEMA(model)
    for batch in loader:
        optimizer.step()
        ema.update(model)          # call after every optimizer step
    map50 = validate(ema.ema, ...) # validate with shadow model
    torch.save(ema.ema.state_dict(), "best.pt")
    """

    def __init__(
        self,
        model:   nn.Module,
        decay:   float = 0.9999,
        tau:     int   = 2000,
        updates: int   = 0,       # pass saved value when resuming from checkpoint
    ):
        self.ema     = copy.deepcopy(model).eval()
        self.decay   = decay
        self.tau     = tau
        self.updates = updates
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update shadow weights. Call after every optimizer.step()."""
        self.updates += 1
        # Decay warmup: starts near 0, asymptotes to self.decay
        d   = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1.0 - d) * msd[k].detach()
