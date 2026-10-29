"""
cbrs/cedar_yolo.py

CEDAR-YOLO in the paper is called SigYOLO / AsymYOLO (sig_yolo_* runs) in the code.

Asymmetric YOLO family — no neck, single detection head.

Three independently configurable axes:

  fm_freq, fm_time     Feature map dimensions before the head.
                       The backbone is built automatically to achieve this size
                       from the given input dimensions.
                       fm_freq = input_w / (product of freq-axis strides in backbone)
                       fm_time = input_h / (product of time-axis strides in backbone)

  stride_freq,         Anchor spacing used by the detection head (pixels per cell).
  stride_time          Independent of fm size — controls the pixel region each anchor
                       "claims" and scales DFL offset predictions.
                       Default: input_w / fm_freq  (natural full-coverage stride)
                       Override: set larger to predict wider regions per cell,
                                 or smaller for denser anchor coverage.

  reg_max_freq,        DFL distribution bins for left/right (freq) and top/bottom
  reg_max_time         (time) offsets. Max detectable half-extent:
                         freq: reg_max_freq * stride_freq  pixels
                         time: reg_max_time * stride_time  pixels

Example configurations (input 640×640):

  fm_freq=80, fm_time=20, stride_freq=8,  stride_time=32
      → 1600 anchors, stride matches natural 640/80=8
  fm_freq=80, fm_time=20, stride_freq=16, stride_time=32
      → same 1600 anchors, but each freq cell covers 16px (useful for 1280-wide input)
  fm_freq=80, fm_time=40, stride_freq=8,  stride_time=16
      → 3200 anchors, finer time resolution (4-stage backbone instead of 5)

Backbone stage layout:
  Given input (input_h, input_w) and target (fm_time, fm_freq):
    n_freq_down = log2(input_w / fm_freq)   — number of freq-axis downsampling steps
    n_time_down = log2(input_h / fm_time)   — number of time-axis downsampling steps
    n_sym       = min(n_freq_down, n_time_down)  — symmetric stages  stride (2,2)
    n_asym_t    = n_time_down - n_sym            — time-only stages  stride (2,1)
    n_asym_f    = n_freq_down - n_sym            — freq-only stages  stride (1,2)

Size multipliers (matches Ultralytics yolov8.yaml :: scales):
  'n': depth=0.33, width=0.25, max_channels=1024
  's': depth=0.33, width=0.50, max_channels=1024
  'm': depth=0.67, width=0.75, max_channels=768
  'l': depth=1.00, width=1.00, max_channels=512
  'x': depth=1.00, width=1.25, max_channels=512

max_channels caps the per-stage base channel count BEFORE the width multiplier
is applied (per ultralytics/nn/tasks.py :: parse_model).  Without this cap the
deep stages (P5+) blow up — v8-x P5 would be 1280ch instead of 640ch, doubling
the parameter count and breaking the LR/optimizer schedule tuned for the
official sizing.
"""
from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Size presets
# ─────────────────────────────────────────────────────────────────────────────

_SIZE_PRESETS: Dict[str, Tuple[float, float, int]] = {
    # (depth_mult, width_mult, max_channels)  — mirrors ultralytics yolov8.yaml
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


# ─────────────────────────────────────────────────────────────────────────────
# Primitive: Conv + BN + SiLU
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + SiLU.  k and s accept int or (h, w) tuples."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | Tuple[int, int] = 1,
        s: int | Tuple[int, int] = 1,
        p: Optional[int | Tuple[int, int]] = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        k = (k, k) if isinstance(k, int) else tuple(k)
        s = (s, s) if isinstance(s, int) else tuple(s)
        p = (k[0] // 2, k[1] // 2) if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck variants
# ─────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """Standard 3×3 → 3×3 residual bottleneck (YOLOv8)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 3)
        self.cv2 = ConvBNAct(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3kBottleneck(nn.Module):
    """Configurable-kernel bottleneck (YOLOv11)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, k: int = 3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, k)
        self.cv2 = ConvBNAct(c_, c2, k)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SEBottleneck(nn.Module):
    """Bottleneck + Squeeze-Excitation (YOLOv26)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, se_ratio: int = 4):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 3)
        self.cv2 = ConvBNAct(c_, c2, 3)
        self.add = shortcut and c1 == c2
        c_se = max(1, c2 // se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, c_se), nn.SiLU(),
            nn.Linear(c_se, c2), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.cv2(self.cv1(x))
        out = out * self.se(out).view(out.shape[0], out.shape[1], 1, 1)
        return x + out if self.add else out


# ─────────────────────────────────────────────────────────────────────────────
# Stage blocks
# ─────────────────────────────────────────────────────────────────────────────

class C2f(nn.Module):
    """YOLOv8 Cross-Stage Partial block."""

    _bottleneck = Bottleneck
    _extra_kwargs: tuple = ()

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5, **kw):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * c_, 1)
        self.cv2 = ConvBNAct((2 + n) * c_, c2, 1)
        self.m   = nn.ModuleList(
            self.__class__._bottleneck(c_, c_, shortcut=shortcut, e=1.0, **kw)
            for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class C3k2(C2f):
    """YOLOv11: C2f with configurable-kernel bottleneck."""
    _bottleneck = C3kBottleneck


class A2C2f(C2f):
    """YOLOv26: C2f with Squeeze-Excitation bottleneck."""
    _bottleneck = SEBottleneck


# ─────────────────────────────────────────────────────────────────────────────
# Tail modules
# ─────────────────────────────────────────────────────────────────────────────

class AsymSPPF(nn.Module):
    """
    Asymmetric Spatial Pyramid Pooling.
    Pools time and freq axes with independent kernel sizes, suitable for
    non-square feature maps (e.g. 20×80).
    """

    def __init__(self, c1: int, c2: int, k_time: int = 3, k_freq: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1  = ConvBNAct(c1, c_, 1)
        self.cv2  = ConvBNAct(c_ * 4, c2, 1)
        # kernel=(H, W) = (time_kernel, freq_kernel)
        self.pool = nn.MaxPool2d(
            kernel_size=(k_time, k_freq),
            stride=1,
            padding=(k_time // 2, k_freq // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = [self.cv1(x)]
        y.extend(self.pool(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, dim=1))


class _PSAAttention(nn.Module):
    """
    Conv-based self-attention used inside PSA.
    Mirrors ultralytics/nn/modules/block.py :: Attention (exact algorithm).

    Key difference from standard MHA: Q and K have a reduced dim per head
    (key_dim = head_dim * attn_ratio) while V retains the full head_dim.
    This asymmetric Q/K projection is lighter and matches the official design.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.key_dim   = int(self.head_dim * attn_ratio)
        self.scale     = self.key_dim ** -0.5
        nh_kd = num_heads * self.key_dim
        # Single projection: V channels (dim) + Q channels (nh_kd) + K channels (nh_kd)
        self.qkv  = ConvBNAct(dim, dim + 2 * nh_kd, 1, act=False)
        self.proj = ConvBNAct(dim, dim, 1, act=False)
        self.pe   = ConvBNAct(dim, dim, 3, g=dim, act=False)   # depthwise position encoding

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N   = H * W
        qkv = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale          # [B, nh, N, N]
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        out = out + self.pe(v.reshape(B, C, H, W))
        return self.proj(out)


class PSA(nn.Module):
    """
    Polarized Self-Attention tail (YOLOv11).
    Applied after AsymSPPF, before the detection head.

    Mirrors ultralytics/nn/modules/block.py :: PSA (exact structure).
    The only difference is that anchor_pts can be non-square (handled by
    _PSAAttention which operates over any H×W spatial layout).
    """

    def __init__(self, c1: int, e: float = 0.5):
        super().__init__()
        self.c_  = int(c1 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c_, 1)
        self.cv2 = ConvBNAct(2 * self.c_, c1, 1)
        nh       = max(1, self.c_ // 64)
        self.attn = _PSAAttention(self.c_, num_heads=nh, attn_ratio=0.5)
        # Conv-based FFN — matches official (no LayerNorm)
        self.ffn  = nn.Sequential(
            ConvBNAct(self.c_, self.c_ * 2, 1),
            ConvBNAct(self.c_ * 2, self.c_, 1, act=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.cv1(x).chunk(2, dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat([a, b], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Detection head with asymmetric DFL
# ─────────────────────────────────────────────────────────────────────────────

class AsymDetect(nn.Module):
    """
    Single-scale detection head with independent freq/time DFL.

    Parameters
    ----------
    nc           : number of classes
    ch           : input feature map channels
    stride_freq  : anchor spacing along frequency axis (pixels)
    stride_time  : anchor spacing along time axis (pixels)
    reg_max_freq : DFL bins for left/right offsets  (max half-width  = reg_max_freq * stride_freq)
    reg_max_time : DFL bins for top/bottom offsets  (max half-height = reg_max_time * stride_time)

    Training  → returns (box_raw [B, 2*rmf+2*rmt, H*W], cls_raw [B, nc, H*W])
    Inference → returns (y [B, 4+nc, A] packed xyxy+sigmoid_cls, (box_raw, cls_raw))
                matches official Detect.forward() inference output format:
                ultralytics/nn/modules/head.py :: Detect.forward
    """

    def __init__(
        self,
        nc: int,
        ch: int,
        stride_freq: int,
        stride_time: int,
        reg_max_freq: int,
        reg_max_time: int,
        k_freq_head: int = 5,
        k_time_head: int = 3,
    ):
        super().__init__()
        self.nc           = nc
        self.stride_freq  = stride_freq
        self.stride_time  = stride_time
        self.reg_max_freq = reg_max_freq
        self.reg_max_time = reg_max_time
        self.no_box       = 2 * reg_max_freq + 2 * reg_max_time

        # Channel dims — exact match to official Detect.__init__:
        #   c2 = max(16, ch[0] // 4, self.reg_max * 4)   → replace 4*reg_max with no_box
        #   c3 = max(ch[0], min(nc, 100))
        # ultralytics/nn/modules/head.py :: Detect.__init__
        c_box = max(16, ch // 4, self.no_box)
        c_cls = max(ch, min(nc, 100))

        # Factored convs: (1, k_f) captures freq patterns, (k_t, 1) captures time patterns.
        # Replaces two symmetric 3×3 convs; fewer params, matches signal geometry.
        # ultralytics/nn/modules/head.py :: Detect.__init__ (same 2-conv + 1×1 structure)
        def _branch(c_hid: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                ConvBNAct(ch,    c_hid, k=(1, k_freq_head)),
                ConvBNAct(c_hid, c_hid, k=(k_time_head, 1)),
                nn.Conv2d(c_hid, c_out, 1),
            )

        self.cv2 = _branch(c_box, self.no_box)   # box branch
        self.cv3 = _branch(c_cls, nc)            # cls branch

    # ── DFL expected value ────────────────────────────────────────────────

    @staticmethod
    def _dfl(raw: Tensor, reg_max: int) -> Tensor:
        """raw [B, reg_max, A] → expected value [B, A] in stride units."""
        idx = torch.arange(reg_max, device=raw.device, dtype=raw.dtype)
        return (raw.softmax(dim=1) * idx.unsqueeze(-1)).sum(dim=1)

    def _decode(self, box_raw: Tensor, H: int, W: int) -> Tensor:
        """box_raw [B, no_box, A] → xyxy boxes [B, A, 4] in pixel coords."""
        rmf, rmt = self.reg_max_freq, self.reg_max_time
        B, _, A  = box_raw.shape

        l = self._dfl(box_raw[:, :rmf],            rmf) * self.stride_freq
        r = self._dfl(box_raw[:, rmf:2*rmf],       rmf) * self.stride_freq
        t = self._dfl(box_raw[:, 2*rmf:2*rmf+rmt], rmt) * self.stride_time
        b = self._dfl(box_raw[:, 2*rmf+rmt:],      rmt) * self.stride_time

        # anchor centres  (H = time rows, W = freq cols)
        gy = (torch.arange(H, device=box_raw.device) + 0.5) * self.stride_time  # [H]
        gx = (torch.arange(W, device=box_raw.device) + 0.5) * self.stride_freq  # [W]
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")      # [H, W]
        cx = cx.reshape(-1)   # [A]
        cy = cy.reshape(-1)   # [A]

        return torch.stack([cx - l, cy - t, cx + r, cy + b], dim=-1)  # [B, A, 4]

    def forward(self, x: Tensor):
        B, _, H, W = x.shape
        A = H * W

        box_raw = self.cv2(x).view(B, self.no_box, A)
        cls_raw = self.cv3(x).view(B, self.nc, A)

        if self.training:
            return box_raw, cls_raw

        # Inference: decode boxes, sigmoid cls, pack as [B, 4+nc, A].
        # Mirrors official Detect.forward() inference branch:
        #   dbox = decode_bboxes(dfl(box), anchors)
        #   y = cat([dbox, cls.sigmoid()], 1)  → [B, 4+nc, A]
        #   return y if export else (y, x)
        # ultralytics/nn/modules/head.py :: Detect.forward
        boxes    = self._decode(box_raw, H, W).permute(0, 2, 1)  # [B, 4, A]
        cls_prob = cls_raw.sigmoid()                              # [B, nc, A]
        y = torch.cat([boxes, cls_prob], dim=1)                   # [B, 4+nc, A]
        return y, (box_raw, cls_raw)

    def summary(self) -> dict:
        return {
            "stride_freq":       self.stride_freq,
            "stride_time":       self.stride_time,
            "reg_max_freq":      self.reg_max_freq,
            "reg_max_time":      self.reg_max_time,
            "max_half_width_px": self.reg_max_freq * self.stride_freq,
            "max_half_height_px":self.reg_max_time * self.stride_time,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backbone builder
# ─────────────────────────────────────────────────────────────────────────────

def _scaled_ch(base: int, w: float, div: int = 8) -> int:
    """Round channel count to nearest multiple of div after scaling."""
    return max(div, int(math.ceil(base * w / div) * div))


def _scaled_n(base: int, d: float) -> int:
    return max(1, round(base * d)) if base > 0 else 0


def _compute_stage_strides(
    input_w: int,
    input_h: int,
    fm_freq: int,
    fm_time: int,
) -> List[Tuple[int, int]]:
    """
    Compute per-stage (stride_h, stride_w) = (time_stride, freq_stride) tuples
    so that the backbone transforms (input_h × input_w) → (fm_time × fm_freq).

    Each stride is (2,2), (2,1) [time only], or (1,2) [freq only].
    Symmetric stages come first, then asymmetric.
    """
    def log2_exact(a: int, b: int, axis: str) -> int:
        ratio = a / b
        n = round(math.log2(ratio))
        if (1 << n) != a // b:
            raise ValueError(
                f"{axis} downsampling factor {a}/{b}={ratio:.2f} is not a power of 2. "
                f"Choose fm_{axis} as input_{axis} / 2^k."
            )
        return n

    n_freq = log2_exact(input_w, fm_freq, "freq")
    n_time = log2_exact(input_h, fm_time, "time")

    n_sym    = min(n_freq, n_time)   # stages where both axes step by 2
    n_t_only = n_time - n_sym        # stages where only time steps by 2
    n_f_only = n_freq - n_sym        # stages where only freq steps by 2

    return (
        [(2, 2)] * n_sym
        + [(2, 1)] * n_t_only
        + [(1, 2)] * n_f_only
    )


def _build_channels(n_stages: int, width_mult: float, max_channels: int) -> List[int]:
    """
    Channel progression for n_stages backbone stages.
    Mirrors official YOLOv8 channel schedule:
      P1=64, P2=128, P3=256, P4=512, P5=1024, deeper=1024.
    Per ultralytics parse_model: each stage uses min(base, max_channels) * width.
    """
    base = [64, 128, 256, 512, 1024, 1024, 1024]   # up to 7 stages
    return [3] + [
        _scaled_ch(min(base[min(i, len(base) - 1)], max_channels), width_mult)
        for i in range(n_stages)
    ]


def _build_repeats(n_stages: int, depth_mult: float) -> List[int]:
    """C2f repeat counts per stage, scaled by depth_mult."""
    base = [3, 6, 6, 3, 3, 3, 3]
    return [_scaled_n(base[min(i, len(base)-1)], depth_mult) for i in range(n_stages)]


def _make_stage(
    block_cls,
    c1: int,
    c2: int,
    n: int,
    stride_h: int,
    stride_w: int,
    **block_kwargs,
) -> nn.Sequential:
    """Downsample Conv (asymmetric stride) → optional stage block."""
    layers: List[nn.Module] = [ConvBNAct(c1, c2, k=3, s=(stride_h, stride_w))]
    if n > 0:
        layers.append(block_cls(c2, c2, n=n, shortcut=True, **block_kwargs))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

class _AsymYOLOBase(nn.Module):
    """
    Shared backbone + head logic.  Subclasses set block type and depth config.

    Key principle: backbone is built from (fm_freq, fm_time);
                   head is configured from (stride_freq, stride_time, reg_max_*).
                   These three groups are independently controllable.
    """

    _block_cls   = C2f
    _has_psa     = False
    _block_kwargs: dict = {}

    def __init__(
        self,
        nc: int = 2,
        size: Literal["n", "s", "m", "l", "x"] = "n",
        # ── Backbone: target feature map dimensions ──
        fm_freq: int = 80,
        fm_time: int = 20,
        # ── Head: anchor spacing (pixels per cell) ──
        stride_freq: Optional[int] = None,
        stride_time: Optional[int] = None,
        # ── Head: DFL bins ──
        reg_max_freq: int = 28,
        reg_max_time: int = 10,
        # ── Input ──
        input_w: int = 640,
        input_h: int = 640,
        # ── AsymSPPF kernels ──
        k_freq_sppf: int = 5,
        k_time_sppf: int = 3,
        # ── Head conv kernels ──
        k_freq_head: int = 5,
        k_time_head: int = 3,
    ):
        super().__init__()

        if size not in _SIZE_PRESETS:
            raise ValueError(f"size must be one of {list(_SIZE_PRESETS)}")
        depth_mult, width_mult, max_channels = _SIZE_PRESETS[size]

        # Default strides: natural full-coverage (input / fm)
        if stride_freq is None:
            stride_freq = input_w // fm_freq
        if stride_time is None:
            stride_time = input_h // fm_time

        # ── Backbone stage layout ─────────────────────────────────────────
        stage_strides = _compute_stage_strides(input_w, input_h, fm_freq, fm_time)
        n_stages      = len(stage_strides)
        channels      = _build_channels(n_stages, width_mult, max_channels)
        repeats       = _build_repeats(n_stages, depth_mult)

        stages = [
            _make_stage(
                self._block_cls,
                channels[i], channels[i + 1],
                n=repeats[i],
                stride_h=sh, stride_w=sw,
                **self._block_kwargs,
            )
            for i, (sh, sw) in enumerate(stage_strides)
        ]
        self.backbone = nn.Sequential(*stages)

        # ── Tail ─────────────────────────────────────────────────────────
        self.psa  = PSA(channels[-1]) if self._has_psa else nn.Identity()
        self.sppf = AsymSPPF(channels[-1], channels[-1], k_time=k_time_sppf, k_freq=k_freq_sppf)

        # ── Head ─────────────────────────────────────────────────────────
        self.head = AsymDetect(
            nc=nc, ch=channels[-1],
            stride_freq=stride_freq, stride_time=stride_time,
            reg_max_freq=reg_max_freq, reg_max_time=reg_max_time,
            k_freq_head=k_freq_head, k_time_head=k_time_head,
        )

        # Store config for introspection
        self.fm_freq    = fm_freq
        self.fm_time    = fm_time
        self.input_w    = input_w
        self.input_h    = input_h
        self._channels  = channels
        self._stage_strides = stage_strides

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        x = self.backbone(x)
        x = self.sppf(x)    # SPPF first — matches official: backbone → SPPF → (C2PSA) → head
        x = self.psa(x)     # PSA after SPPF (nn.Identity for v8/v26)
        return self.head(x)

    # ── Introspection helpers ─────────────────────────────────────────────

    def info(self) -> dict:
        """Return a summary of the model configuration."""
        n_params = sum(p.numel() for p in self.parameters())
        head = self.head.summary()
        return {
            "variant":       self.__class__.__name__,
            "fm":            f"{self.fm_time}(time) × {self.fm_freq}(freq)",
            "anchors":       self.fm_time * self.fm_freq,
            "stage_strides": [(sh, sw) for sh, sw in self._stage_strides],
            "channels":      self._channels,
            **head,
            "params":        f"{n_params/1e6:.2f}M",
        }

    def predict(
        self,
        x: Tensor,
        conf_thres:  float = 0.25,
        iou_thres:   float = 0.45,
        classes:     Optional[List[int]] = None,
        agnostic:    bool  = False,
        multi_label: bool  = False,
        labels:      tuple = (),
        max_det:     int   = 300,
    ) -> List[Tensor]:
        """
        Run inference and apply Non-Maximum Suppression.

        Mirrors ultralytics/models/yolo/detect/predict.py + ops.non_max_suppression().
        All parameters are identical to ultralytics.non_max_suppression().

        Parameters
        ----------
        x           : [B, C, H, W] input tensor
        conf_thres  : minimum class confidence (default 0.25)
        iou_thres   : NMS IoU threshold (default 0.45)
        classes     : optional list of class indices to keep; None = keep all
        agnostic    : class-agnostic NMS (all classes share one suppression pool)
        multi_label : allow multiple class labels per anchor box
        labels      : auto-labelling priors list[(M, 5)] per image, empty = disabled
                      format per row: (cls_id, x1, y1, x2, y2) in pixel coords
        max_det     : maximum detections per image

        Returns
        -------
        List[Tensor[N, 6]]  per image: columns = (x1, y1, x2, y2, conf, cls_id)
        """
        training = self.training
        try:
            self.eval()
            with torch.no_grad():
                y, _ = self.forward(x)   # y: [B, 4+nc, A]
        finally:
            self.train(training)
        return asym_non_max_suppression(
            y,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public model classes
# ─────────────────────────────────────────────────────────────────────────────

class AsymYOLOv8(_AsymYOLOBase):
    """Asymmetric YOLOv8 — C2f backbone, no neck, single asymmetric head."""
    _block_cls    = C2f
    _has_psa      = False
    _block_kwargs = {}


class AsymYOLOv11(_AsymYOLOBase):
    """Asymmetric YOLOv11 — C3k2 backbone + PSA tail, no neck, single asymmetric head."""
    _block_cls    = C3k2
    _has_psa      = True
    _block_kwargs = {"k": 3}


class AsymYOLOv26(_AsymYOLOBase):
    """Asymmetric YOLOv26 — A2C2f (SE-attention) backbone, no neck, single asymmetric head."""
    _block_cls    = A2C2f
    _has_psa      = False
    _block_kwargs = {"se_ratio": 4}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_VARIANT_MAP = {"v8": AsymYOLOv8, "v11": AsymYOLOv11, "v26": AsymYOLOv26}


def build_asym_yolo(
    variant: Literal["v8", "v11", "v26"] = "v8",
    size: Literal["n", "s", "m", "l", "x"] = "n",
    nc: int = 2,
    fm_freq: int = 80,
    fm_time: int = 20,
    stride_freq: Optional[int] = None,
    stride_time: Optional[int] = None,
    reg_max_freq: int = 28,
    reg_max_time: int = 10,
    input_w: int = 640,
    input_h: int = 640,
    **kwargs,
) -> _AsymYOLOBase:
    """
    Build an asymmetric YOLO model.

    Parameters
    ----------
    variant      : 'v8' | 'v11' | 'v26'
    size         : 'n' | 's' | 'm' | 'l' | 'x'
    nc           : number of detection classes
    fm_freq      : feature map width  (freq columns). Must satisfy: input_w / fm_freq = 2^k
    fm_time      : feature map height (time rows).   Must satisfy: input_h / fm_time = 2^k
    stride_freq  : anchor spacing in freq direction (pixels). Default = input_w / fm_freq
    stride_time  : anchor spacing in time direction (pixels). Default = input_h / fm_time
    reg_max_freq : DFL bins for left/right offsets
    reg_max_time : DFL bins for top/bottom offsets
    input_w/h    : input image size (default 640×640)

    Examples
    --------
    # Default (stride matches natural coverage)
    >>> m = build_asym_yolo("v8", "n")
    # fm=20×80, stride_freq=8, stride_time=32, 1600 anchors

    # Finer time resolution (4-stage backbone)
    >>> m = build_asym_yolo("v11", "s", fm_time=40)
    # fm=40×80, stride_freq=8, stride_time=16, 3200 anchors

    # Same fm, custom stride (e.g. for 1280-wide input)
    >>> m = build_asym_yolo("v26", "m", fm_freq=80, fm_time=20,
    ...                      stride_freq=16, stride_time=32, input_w=1280)

    # Custom reg_max
    >>> m = build_asym_yolo("v8", "n", reg_max_freq=16, reg_max_time=8)
    """
    if variant not in _VARIANT_MAP:
        raise ValueError(f"variant must be one of {list(_VARIANT_MAP)}, got {variant!r}")
    return _VARIANT_MAP[variant](
        nc=nc, size=size,
        fm_freq=fm_freq, fm_time=fm_time,
        stride_freq=stride_freq, stride_time=stride_time,
        reg_max_freq=reg_max_freq, reg_max_time=reg_max_time,
        input_w=input_w, input_h=input_h,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# IoU utilities
# mirrors ultralytics/utils/metrics.py :: bbox_iou (xywh=False, CIoU=True)
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_iou(b1: Tensor, b2: Tensor, eps: float = 1e-7, ciou: bool = True) -> Tensor:
    """Element-wise CIoU between b1 [N,4] and b2 [N,4] in xyxy format. Returns [N]."""
    w1 = (b1[:, 2] - b1[:, 0]).clamp(eps)
    h1 = (b1[:, 3] - b1[:, 1]).clamp(eps)
    w2 = (b2[:, 2] - b2[:, 0]).clamp(eps)
    h2 = (b2[:, 3] - b2[:, 1]).clamp(eps)

    inter = (b1[:, 2].minimum(b2[:, 2]) - b1[:, 0].maximum(b2[:, 0])).clamp(0) * \
            (b1[:, 3].minimum(b2[:, 3]) - b1[:, 1].maximum(b2[:, 1])).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou   = inter / union

    if not ciou:
        return iou

    cw   = b1[:, 2].maximum(b2[:, 2]) - b1[:, 0].minimum(b2[:, 0])
    ch   = b1[:, 3].maximum(b2[:, 3]) - b1[:, 1].minimum(b2[:, 1])
    c2   = cw ** 2 + ch ** 2 + eps
    rho2 = ((b1[:, 0] + b1[:, 2] - b2[:, 0] - b2[:, 2]) ** 2 +
            (b1[:, 1] + b1[:, 3] - b2[:, 1] - b2[:, 3]) ** 2) / 4
    v    = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()) ** 2
    with torch.no_grad():
        alpha_v = v / (v - iou + 1 + eps)
    return iou - (rho2 / c2 + v * alpha_v)


# ─────────────────────────────────────────────────────────────────────────────
# DFL Loss
# mirrors ultralytics/utils/loss.py :: DFLoss (computation identical)
# ─────────────────────────────────────────────────────────────────────────────

class _DFLoss(nn.Module):
    """
    Distribution Focal Loss.

    Identical to ultralytics/utils/loss.py :: DFLoss.
    Two instances are used in AsymDetectLoss: one for freq axis (reg_max_freq),
    one for time axis (reg_max_time).
    """

    def __init__(self, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """
        pred_dist : [N, reg_max]  raw logits
        target    : [N]           float offset in distribution units [0, reg_max)
        Returns   : [N]           per-anchor DFL loss
        """
        target = target.clamp(0, self.reg_max - 1 - 1e-3)
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1.0 - wl
        return (
            F.cross_entropy(pred_dist, tl, reduction="none") * wl
            + F.cross_entropy(pred_dist, tr, reduction="none") * wr
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task-Aligned Assigner
# mirrors ultralytics/utils/tal.py :: TaskAlignedAssigner (algorithm identical)
# ─────────────────────────────────────────────────────────────────────────────

class _TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner.

    Algorithm is identical to ultralytics/utils/tal.py :: TaskAlignedAssigner.
    The only difference is that anchor_pts here use an asymmetric
    (stride_freq × stride_time) grid instead of a uniform stride grid.

    Steps:
      1. in_gts   — which anchors fall inside each GT box
      2. overlaps — pairwise IoU between predicted boxes and GT boxes
      3. align    — cls_score^alpha * iou^beta * in_gts
      4. topk     — select top-k anchors per GT
      5. resolve  — each anchor → GT with highest IoU (conflict resolution)
      6. normalise alignment metric → soft cls weight (matches official)
    """

    def __init__(self, topk: int = 10, nc: int = 80,
                 alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk  = topk
        self.nc    = nc
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    @torch.no_grad()
    def forward(
        self,
        pred_cls:   Tensor,   # [B, A, nc]  sigmoid probs (detached)
        pred_boxes: Tensor,   # [B, A, 4]   decoded xyxy pixels (detached)
        anchor_pts: Tensor,   # [A, 2]      (cx, cy) in pixels
        gt_cls:     Tensor,   # [B, G]      class indices (long)
        gt_boxes:   Tensor,   # [B, G, 4]   GT xyxy in pixels
        gt_mask:    Tensor,   # [B, G]      bool, True = valid GT
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        target_cls  [B, A, nc]  alignment-weighted soft one-hot cls targets
        target_box  [B, A, 4]   assigned GT box for each anchor
        fg_mask     [B, A]      bool, True if anchor is assigned
        target_iou  [B, A]      IoU of pred vs assigned GT (0 for background)
        """
        B, A, _ = pred_cls.shape
        G       = gt_boxes.shape[1]
        device  = pred_cls.device

        # 1. in_gts  [B, A, G]
        pts    = anchor_pts[None, :, None, :]    # [1, A, 1, 2]
        lt     = gt_boxes[:, None, :, :2]        # [B, 1, G, 2]
        rb     = gt_boxes[:, None, :, 2:]        # [B, 1, G, 2]
        in_gts = ((pts > lt) & (pts < rb)).all(-1) & gt_mask[:, None, :]  # [B, A, G]

        # 2. pairwise IoU  [B, A, G]
        p = pred_boxes.unsqueeze(2)              # [B, A, 1, 4]
        g = gt_boxes.unsqueeze(1)                # [B, 1, G, 4]
        inter = (p[..., 2:].minimum(g[..., 2:]) - p[..., :2].maximum(g[..., :2])).clamp(0).prod(-1)
        area_p = (pred_boxes[..., 2:] - pred_boxes[..., :2]).prod(-1)   # [B, A]
        area_g = (gt_boxes[..., 2:]   - gt_boxes[..., :2]).prod(-1)     # [B, G]
        union   = area_p[:, :, None] + area_g[:, None, :] - inter + self.eps
        overlaps = (inter / union).clamp(0)                              # [B, A, G]

        # 3. cls alignment  [B, A, G]
        cls_idx  = gt_cls[:, None, :].expand(B, A, G).clamp(0)
        cls_scores = torch.gather(pred_cls, 2, cls_idx)                  # [B, A, G]
        align = cls_scores.pow(self.alpha) * overlaps.pow(self.beta) * in_gts.float()

        # 4. top-k per GT
        topk = min(self.topk, A)
        _, topk_idx = align.topk(topk, dim=1)                           # [B, topk, G]
        is_topk = torch.zeros(B, A, G, dtype=torch.bool, device=device)
        is_topk.scatter_(1, topk_idx, True)
        mask_pos = is_topk & in_gts & gt_mask[:, None, :]               # [B, A, G]

        # 5. conflict resolution — anchor → GT with highest IoU
        iou_masked    = overlaps * mask_pos.float()
        target_gt_idx = iou_masked.argmax(dim=2)                         # [B, A]
        fg_mask       = mask_pos.any(dim=2)                              # [B, A]

        # 6. normalise alignment metric  (mirrors official TAL normalisation)
        # Official: pos_align_metrics = align.amax(axis=-1)  → max over ANCHORS per GT
        #           pos_overlaps      = overlaps.amax(axis=-1) → max IoU over anchors per GT
        # Our axes are transposed ([B,A,G] vs official [B,G,A]), so we amax over dim=1.
        align = align * mask_pos.float()
        max_align_per_gt = align.amax(dim=1, keepdim=True)              # [B, 1, G]
        max_iou_per_gt   = iou_masked.amax(dim=1, keepdim=True)         # [B, 1, G]
        norm_w = (align * max_iou_per_gt / (max_align_per_gt + self.eps)).amax(dim=2)  # [B, A]

        # 7. build targets
        bidx       = torch.arange(B, device=device)[:, None]            # [B, 1]
        target_box = gt_boxes[bidx, target_gt_idx]                       # [B, A, 4]
        target_iou = iou_masked.amax(dim=2) * fg_mask.float()           # [B, A]

        fg_gt_cls  = gt_cls[bidx, target_gt_idx].clamp(0)               # [B, A]
        target_cls = torch.zeros(B, A, self.nc, device=device)
        target_cls.scatter_(2, fg_gt_cls.unsqueeze(-1), 1.0)
        target_cls = target_cls * fg_mask.unsqueeze(-1).float() * norm_w.unsqueeze(-1)

        return target_cls, target_box, fg_mask, target_iou


# ─────────────────────────────────────────────────────────────────────────────
# Training Loss
# mirrors ultralytics/utils/loss.py :: v8DetectionLoss
# ─────────────────────────────────────────────────────────────────────────────

class AsymDetectLoss(nn.Module):
    """
    Training loss for AsymDetect.

    Mirrors ultralytics/utils/loss.py :: v8DetectionLoss.

    ┌──────────────────────┬───────────────────────┬────────────────────────────────┐
    │ Component            │ Official (ultralytics) │ AsymDetectLoss                 │
    ├──────────────────────┼───────────────────────┼────────────────────────────────┤
    │ Box branch output    │ 4 × reg_max            │ 2×rmf + 2×rmt                  │
    │ DFL loss             │ one DFLoss(reg_max)    │ _DFLoss(rmf) + _DFLoss(rmt)    │
    │ DFL target l, r      │ offset / stride        │ offset / stride_freq           │
    │ DFL target t, b      │ offset / stride        │ offset / stride_time           │
    │ Anchor grid          │ uniform stride         │ stride_freq(x) × stride_time(y)│
    │ TAL assignment       │ identical              │ identical                      │
    │ IoU loss             │ CIoU                   │ CIoU (identical)               │
    │ Cls loss             │ BCEWithLogitsLoss      │ BCEWithLogitsLoss (identical)  │
    │ Loss normalisation   │ sum(target_scores)     │ identical                      │
    └──────────────────────┴───────────────────────┴────────────────────────────────┘

    Target format
    -------------
    targets : Tensor [N, 6]
        Columns: (batch_idx, class_idx, x1, y1, x2, y2) in pixel coordinates.
    """

    def __init__(
        self,
        model:    "_AsymYOLOBase",
        topk:     int   = 10,
        alpha:    float = 0.5,
        beta:     float = 6.0,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
    ) -> None:
        super().__init__()
        head = model.head
        self.nc          = head.nc
        self.rmf         = head.reg_max_freq
        self.rmt         = head.reg_max_time
        self.stride_freq = head.stride_freq
        self.stride_time = head.stride_time
        self.box_gain    = box_gain
        self.cls_gain    = cls_gain
        self.dfl_gain    = dfl_gain

        # ultralytics/utils/loss.py :: v8DetectionLoss.__init__
        self.bce      = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl_freq = _DFLoss(self.rmf)   # left / right (freq axis)
        self.dfl_time = _DFLoss(self.rmt)   # top  / bottom (time axis)
        self.assigner = _TaskAlignedAssigner(topk=topk, nc=self.nc, alpha=alpha, beta=beta)

        # Anchor centres [A, 2] = (cx_pixel, cy_pixel)
        # ultralytics/utils/loss.py :: make_anchors() — same formula, asymmetric strides
        gy = (torch.arange(model.fm_time,  dtype=torch.float32) + 0.5) * self.stride_time
        gx = (torch.arange(model.fm_freq, dtype=torch.float32) + 0.5) * self.stride_freq
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")
        self.register_buffer("anchor_pts", torch.stack([cx.flatten(), cy.flatten()], dim=-1))

    # ── internal: decode box_raw → xyxy for TAL ───────────────────────────

    def _decode_boxes(self, box_raw: Tensor) -> Tensor:
        """box_raw [B, no_box, A] → xyxy [B, A, 4] in pixels."""
        rmf, rmt = self.rmf, self.rmt
        device   = box_raw.device

        def _soft(logits: Tensor, reg_max: int, stride: int) -> Tensor:
            idx = torch.arange(reg_max, device=device, dtype=torch.float32)
            return (logits.permute(0, 2, 1).softmax(-1) * idx).sum(-1) * stride  # [B, A]

        l = _soft(box_raw[:, :rmf],            rmf, self.stride_freq)
        r = _soft(box_raw[:, rmf:2*rmf],       rmf, self.stride_freq)
        t = _soft(box_raw[:, 2*rmf:2*rmf+rmt], rmt, self.stride_time)
        b = _soft(box_raw[:, 2*rmf+rmt:],      rmt, self.stride_time)

        cx = self.anchor_pts[:, 0]  # [A]
        cy = self.anchor_pts[:, 1]  # [A]
        return torch.stack([cx - l, cy - t, cx + r, cy + b], dim=-1)  # [B, A, 4]

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        preds:   Tuple[Tensor, Tensor],
        targets: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        preds   : (box_raw [B, no_box, A], cls_raw [B, nc, A])  — model training output
        targets : [N, 6] (batch_idx, cls, x1, y1, x2, y2) pixels
        Returns : (total_loss, {'box': ..., 'cls': ..., 'dfl': ...})
        """
        box_raw, cls_raw = preds
        B      = box_raw.shape[0]
        device = box_raw.device
        anc    = self.anchor_pts.to(device)

        # ── parse targets → per-image GT tensors ──────────────────────────
        G_max = 1
        if targets.shape[0] > 0:
            G_max = max(1, int(torch.bincount(targets[:, 0].long(), minlength=B).max()))

        gt_cls   = torch.zeros(B, G_max, dtype=torch.long,  device=device)
        gt_boxes = torch.zeros(B, G_max, 4,                  device=device)
        gt_mask  = torch.zeros(B, G_max, dtype=torch.bool,  device=device)

        for b in range(B):
            m = targets[:, 0].long() == b
            g = targets[m]
            n = g.shape[0]
            if n > 0:
                gt_cls[b, :n]   = g[:, 1].long()
                gt_boxes[b, :n] = g[:, 2:6]
                gt_mask[b, :n]  = True

        # ── decode for TAL (detached) ──────────────────────────────────────
        with torch.no_grad():
            pred_boxes = self._decode_boxes(box_raw.detach())              # [B, A, 4]
            pred_cls   = cls_raw.detach().permute(0, 2, 1).sigmoid()      # [B, A, nc]

        # ── task-aligned assignment ────────────────────────────────────────
        target_cls, target_box, fg_mask, _ = self.assigner(
            pred_cls, pred_boxes, anc, gt_cls, gt_boxes, gt_mask,
        )

        # ── cls loss  (ultralytics: bce.sum() / target_scores_sum) ────────
        cls_pred = cls_raw.permute(0, 2, 1)                               # [B, A, nc]
        cls_loss = self.bce(cls_pred, target_cls.float()).sum()
        target_scores_sum = max(target_cls.sum().item(), 1.0)
        cls_loss = cls_loss / target_scores_sum

        # ── box + DFL loss (foreground only) ──────────────────────────────
        # ultralytics/utils/loss.py :: BboxLoss.forward
        # weight = target_scores.sum(-1)[fg_mask]  → alignment weight per fg anchor
        # loss_iou = ((1 - iou) * weight).sum() / target_scores_sum
        n_fg = int(fg_mask.sum())
        if n_fg > 0:
            fg_raw     = box_raw.permute(0, 2, 1)[fg_mask]             # [fg, no_box]
            fg_tgt     = target_box[fg_mask]                            # [fg, 4]
            fg_pred    = pred_boxes[fg_mask]                            # [fg, 4]
            fg_cx      = anc[:, 0].unsqueeze(0).expand(B, -1)[fg_mask] # [fg]
            fg_cy      = anc[:, 1].unsqueeze(0).expand(B, -1)[fg_mask] # [fg]
            fg_weights = target_cls[fg_mask].sum(-1)                    # [fg] alignment weight

            # IoU loss  (matches official: weighted sum / target_scores_sum)
            iou       = _bbox_iou(fg_pred, fg_tgt, ciou=True)          # [fg]
            iou_loss  = ((1.0 - iou) * fg_weights).sum() / target_scores_sum

            # DFL targets in distribution units
            # mirrors ultralytics: bbox2dist(), split by axis for asymmetric strides
            rmf, rmt = self.rmf, self.rmt
            l_t = (fg_cx - fg_tgt[:, 0]) / self.stride_freq
            r_t = (fg_tgt[:, 2] - fg_cx) / self.stride_freq
            t_t = (fg_cy - fg_tgt[:, 1]) / self.stride_time
            b_t = (fg_tgt[:, 3] - fg_cy) / self.stride_time

            dfl_per = (
                self.dfl_freq(fg_raw[:, :rmf],             l_t)
                + self.dfl_freq(fg_raw[:, rmf:2*rmf],       r_t)
                + self.dfl_time(fg_raw[:, 2*rmf:2*rmf+rmt], t_t)
                + self.dfl_time(fg_raw[:, 2*rmf+rmt:],      b_t)
            )  # [fg]
            dfl_loss = (dfl_per * fg_weights).sum() / target_scores_sum
        else:
            iou_loss = (box_raw * 0).sum()
            dfl_loss = (box_raw * 0).sum()

        total = self.box_gain * iou_loss + self.cls_gain * cls_loss + self.dfl_gain * dfl_loss
        return total, {
            "box": iou_loss.detach(),
            "cls": cls_loss.detach(),
            "dfl": dfl_loss.detach(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# NMS utilities
# mirrors ultralytics/utils/ops.py :: non_max_suppression
# ─────────────────────────────────────────────────────────────────────────────

def _pure_torch_nms(boxes: Tensor, scores: Tensor, iou_thres: float) -> Tensor:
    """
    Greedy NMS — identical result to torchvision.ops.nms.
    Used when torchvision is unavailable.
    """
    order = scores.argsort(descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i.item()))
        if order.numel() == 1:
            break
        rest = order[1:]
        x1 = boxes[rest, 0].clamp(min=boxes[i, 0].item())
        y1 = boxes[rest, 1].clamp(min=boxes[i, 1].item())
        x2 = boxes[rest, 2].clamp(max=boxes[i, 2].item())
        y2 = boxes[rest, 3].clamp(max=boxes[i, 3].item())
        inter     = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou       = inter / (area_i + area_rest - inter + 1e-7)
        order     = rest[iou <= iou_thres]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def asym_non_max_suppression(
    prediction:  Tensor,
    conf_thres:  float = 0.25,
    iou_thres:   float = 0.45,
    classes:     Optional[List[int]] = None,
    agnostic:    bool  = False,
    multi_label: bool  = False,
    labels:      tuple = (),
    max_det:     int   = 300,
    max_nms:     int   = 30000,
    max_wh:      float = 7680.0,
) -> List[Tensor]:
    """
    Non-Maximum Suppression for AsymYOLO inference output.

    Signature and all parameters are IDENTICAL to the official implementation.
    ultralytics/utils/ops.py :: non_max_suppression (detection, no masks/rotated)

    ┌─────────────────────┬──────────────────────────┬───────────────────────────┐
    │ Step                │ Official (ultralytics)   │ asym_non_max_suppression  │
    ├─────────────────────┼──────────────────────────┼───────────────────────────┤
    │ Input format        │ [B, 4+nc, A] concat      │ identical                 │
    │ Candidate pre-filter│ cls.amax(1) > conf_thres │ identical                 │
    │ Auto-labelling      │ labels=() param (xyxy)   │ identical                 │
    │ Single-label path   │ cls.max(1) + threshold   │ identical                 │
    │ Multi-label path    │ where(cls > conf_thres)  │ identical                 │
    │ Class filter        │ (x[:,5:6]==cls).any(1)   │ identical                 │
    │ Pre-NMS sort+cap    │ argsort desc [:max_nms]  │ identical                 │
    │ Per-class NMS offset│ cls_id * max_wh          │ identical                 │
    │ NMS backend         │ torchvision.ops.nms      │ torchvision or pure-torch │
    │ Output per image    │ [N, 6] xyxy+conf+cls     │ identical                 │
    └─────────────────────┴──────────────────────────┴───────────────────────────┘

    Parameters
    ----------
    prediction  : [B, 4+nc, A]  decoded xyxy + sigmoid cls  (AsymDetect inference y)
    conf_thres  : confidence threshold (default 0.25)
    iou_thres   : NMS IoU threshold (default 0.45)
    classes     : optional list of class indices to keep; None = keep all
    agnostic    : class-agnostic NMS (ignore class offsets during suppression)
    multi_label : allow multiple class labels per anchor (all classes > conf_thres)
    labels      : auto-labelling priors list[(M, 5)] per image;
                  each row: (cls_id, x1, y1, x2, y2) in pixel coords; empty = disabled
    max_det     : maximum detections per image
    max_nms     : pre-NMS candidate cap (sorted by conf, takes top max_nms)
    max_wh      : per-class box offset scale for batched NMS

    Returns
    -------
    List[Tensor[N, 6]]  per image: (x1, y1, x2, y2, conf, cls_id)
                        Empty images → Tensor of shape [0, 6].
    """
    try:
        from torchvision.ops import nms as _tv_nms
        _nms_fn = _tv_nms
    except ImportError:
        _nms_fn = _pure_torch_nms

    device = prediction.device
    B      = prediction.shape[0]
    nc     = prediction.shape[1] - 4                      # number of classes

    # Unpack packed tensor — identical to official split:
    # ultralytics: box, cls = prediction.split((4, nc), 1)
    # Then candidate mask: xc = cls.amax(1) > conf_thres  [B, A]
    boxes    = prediction[:, :4, :].permute(0, 2, 1)      # [B, A, 4]
    cls_prob = prediction[:, 4:,  :].permute(0, 2, 1)     # [B, A, nc]

    # ultralytics: xc = prediction[:, 4:mi].amax(1) > conf_thres
    xc = cls_prob.amax(dim=2) > conf_thres                 # [B, A]

    output: List[Tensor] = [torch.zeros(0, 6, device=device)] * B

    for xi in range(B):
        box = boxes[xi]     # [A, 4]
        cls = cls_prob[xi]  # [A, nc]

        # ── candidate pre-filter ─────────────────────────────────────────
        # ultralytics: x = x.transpose(0, -1)[xc[xi]]
        box = box[xc[xi]]                                  # [cand, 4]
        cls = cls[xc[xi]]                                  # [cand, nc]

        # ── auto-labelling (semi-supervised) ─────────────────────────────
        # ultralytics: v[:, :4] = xywh2xyxy(lb[:, 1:5])  (official uses xywh)
        #              v[range(len(lb)), lb[:,0].long() + 4] = 1.0
        # AsymYOLO: labels[xi] format = (cls_id, x1, y1, x2, y2) pixel xyxy
        if labels and len(labels) > xi and len(labels[xi]):
            lb    = labels[xi]                             # [M, 5]
            v_box = lb[:, 1:5]                             # xyxy already
            v_cls = torch.zeros(len(lb), nc, device=device)
            v_cls[range(len(lb)), lb[:, 0].long()] = 1.0
            box = torch.cat([box, v_box], dim=0)
            cls = torch.cat([cls, v_cls], dim=0)

        if box.shape[0] == 0:
            continue

        # ── build [N, 6] candidate tensor ────────────────────────────────
        if multi_label:
            # ultralytics: i, j = where(cls > conf_thres)
            #              x = cat([box[i], x[i, 4+j, None], j[:,None].float()], 1)
            ai, ci = torch.where(cls > conf_thres)         # anchor idx, class idx
            x = torch.cat(
                [box[ai], cls[ai, ci, None], ci[:, None].float()], dim=1
            )                                              # [N, 6]
        else:
            # ultralytics: conf, j = cls.max(1, keepdim=True)
            #              x = cat([box, conf, j.float()], 1)[conf.view(-1) > conf_thres]
            conf, ci = cls.max(dim=1, keepdim=True)        # [cand, 1]
            x = torch.cat([box, conf, ci.float()], dim=1)
            x = x[conf.squeeze(1) > conf_thres]

        if x.shape[0] == 0:
            continue

        # ── optional class filter ─────────────────────────────────────────
        # ultralytics: x = x[(x[:, 5:6] == tensor(classes, device)).any(1)]
        if classes is not None:
            cls_t = torch.tensor(classes, device=device, dtype=x.dtype)
            x = x[(x[:, 5:6] == cls_t).any(1)]
        if x.shape[0] == 0:
            continue

        # ── sort by confidence, cap at max_nms ───────────────────────────
        # ultralytics: x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        if x.shape[0] > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # ── per-class NMS via class-offset boxes ──────────────────────────
        # ultralytics: c = x[:, 5:6] * (0 if agnostic else max_wh)
        #              i = torchvision.ops.nms(x[:,:4]+c, x[:,4], iou_thres)
        c         = x[:, 5:6] * (0.0 if agnostic else max_wh)
        nms_boxes = x[:, :4] + c
        scores    = x[:, 4]
        keep      = _nms_fn(nms_boxes, scores, iou_thres)
        keep      = keep[:max_det]
        output[xi] = x[keep]

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    configs = [
        # (label,                  variant, size, kwargs)
        ("default 20×80 s=8",      "v8",  "n", {}),
        ("fm=20×80 stride_f=16",   "v8",  "n", {"stride_freq": 16}),
        ("fm=40×80 s=8",           "v8",  "n", {"fm_time": 40}),
        ("v11 20×80",              "v11", "s", {}),
        ("v26 20×80",              "v26", "n", {}),
        ("custom reg_max",         "v8",  "n", {"reg_max_freq": 16, "reg_max_time": 8}),
        ("v8-l 20×80",             "v8",  "l", {}),
    ]

    dummy = torch.zeros(1, 3, 640, 640)
    for label, variant, size, kw in configs:
        model = build_asym_yolo(variant, size, **kw)
        model.eval()
        with torch.no_grad():
            y, (box_raw, cls_raw) = model(dummy)
        info = model.info()
        print(f"\n── {label} ──")
        pprint.pprint(info)
        nc = model.head.nc
        print(f"   y={tuple(y.shape)}  (expected [1, {4+nc}, {model.fm_time*model.fm_freq}])")

    # ── Loss sanity check ────────────────────────────────────────────────────
    print("\n\n── Loss sanity check ──")
    model = build_asym_yolo("v8", "n")
    model.train()
    criterion = AsymDetectLoss(model)

    x = torch.randn(2, 3, 640, 640)
    preds = model(x)   # training mode: (box_raw, cls_raw)

    # Two fake GT boxes per image: (batch_idx, cls, x1, y1, x2, y2) in pixels
    targets = torch.tensor([
        [0, 0,  50,  10, 150, 620],   # image 0, class 0
        [0, 1, 200, 100, 350, 500],   # image 0, class 1
        [1, 0, 100,   5, 200, 635],   # image 1, class 0
    ], dtype=torch.float32)

    loss, parts = criterion(preds, targets)
    print(f"  total={loss.item():.4f}  box={parts['box'].item():.4f}"
          f"  cls={parts['cls'].item():.4f}  dfl={parts['dfl'].item():.4f}")

    # ── NMS / predict sanity check ───────────────────────────────────────────
    print("\n── NMS / predict sanity check ──")
    model = build_asym_yolo("v8", "n")
    x = torch.randn(2, 3, 640, 640)

    # predict() should save/restore training mode
    model.train()
    results_default = model.predict(x, conf_thres=0.25, iou_thres=0.45, max_det=300)
    assert model.training, "predict() must restore training mode"

    model.eval()
    results_all     = model.predict(x, conf_thres=0.0,  iou_thres=0.45, max_det=300)
    results_ml      = model.predict(x, conf_thres=0.0,  iou_thres=0.45, multi_label=True)
    results_ag      = model.predict(x, conf_thres=0.0,  iou_thres=0.45, agnostic=True)

    for i, (rd, ra) in enumerate(zip(results_default, results_all)):
        print(f"  image {i}: conf>0.25 → {rd.shape[0]} dets | conf>0.0 → {ra.shape[0]} dets"
              f" | multi_label → {results_ml[i].shape[0]} dets"
              f" | agnostic → {results_ag[i].shape[0]} dets")

    # Verify output format (x1, y1, x2, y2, conf, cls_id)
    for i, r in enumerate(results_all):
        assert r.shape[1] == 6,                          f"img {i}: expected 6 cols, got {r.shape[1]}"
        assert (r[:, 4] >= 0.0).all() and \
               (r[:, 4] <= 1.0).all(),                   f"img {i}: conf out of [0,1]"
        assert (r[:, 5] >= 0).all(),                     f"img {i}: cls_id < 0"
        assert (r[:, 2] >= r[:, 0]).all() and \
               (r[:, 3] >= r[:, 1]).all(),               f"img {i}: degenerate box (x2<x1 or y2<y1)"
    print("  format checks passed (6 cols | conf in [0,1] | cls_id>=0 | x2>=x1 | y2>=y1)")

    # Class filter check: keep only class 0
    results_cls0 = model.predict(x, conf_thres=0.0, classes=[0])
    for i, r in enumerate(results_cls0):
        assert (r[:, 5] == 0).all() or r.shape[0] == 0, f"img {i}: class filter failed"
    print("  class filter check passed (cls_id == 0 for all detections)")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  gradient norm sum = {grad_norm:.4f}  (>0 confirms backprop works)")
