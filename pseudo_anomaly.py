# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import os
# import re
# import math
# import hashlib
# import random
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, Sequence, Tuple, List

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


# # --------------------------
# # Config
# # --------------------------
# @dataclass
# class PseudoCfg:
#     # common capture aug
#     p_brightness: float = 0.6
#     p_contrast: float = 0.6
#     p_blur: float = 0.2
#     p_noise: float = 0.4

#     brightness_range: Tuple[float, float] = (0.85, 1.15)
#     contrast_range: Tuple[float, float] = (0.85, 1.15)
#     blur_radius_range: Tuple[float, float] = (0.3, 1.2)
#     noise_sigma_range: Tuple[float, float] = (2.0, 8.0)  # in [0..255] domain

#     # generic pseudo types (non seam-aware)
#     p_cutpaste: float = 0.65
#     p_scratch_line: float = 0.35
#     p_stain: float = 0.25
#     p_local_warp: float = 0.15

#     # seam-aware pseudo types
#     p_seam_defect: float = 0.75
#     p_seam_break: float = 0.55
#     p_double_seam: float = 0.35
#     p_missing_stitch: float = 0.30

#     # cutpaste params
#     patch_min_ratio: float = 0.10
#     patch_max_ratio: float = 0.35
#     rotate_deg: float = 20.0
#     feather: float = 2.0
#     poly_min_v: int = 5
#     poly_max_v: int = 9

#     # scratch params
#     scratch_count_range: Tuple[int, int] = (1, 3)
#     scratch_width_range: Tuple[int, int] = (1, 3)
#     scratch_alpha_range: Tuple[float, float] = (0.25, 0.55)

#     # stain params
#     stain_blob_count_range: Tuple[int, int] = (3, 8)
#     stain_alpha_range: Tuple[float, float] = (0.10, 0.35)
#     stain_blur_range: Tuple[float, float] = (3.0, 9.0)

#     # warp params
#     warp_strength_range: Tuple[float, float] = (0.003, 0.010)
#     warp_kernel_range: Tuple[int, int] = (9, 21)

#     # seam detection params (white stitch on dark fabric)
#     seam_quantile: float = 0.985
#     seam_min_thr: float = 0.60
#     seam_morph_ksize: int = 7
#     seam_min_pixels: int = 80
#     seam_band_dilate: int = 9

#     # seam break / missing stitch params
#     seam_break_len_ratio: Tuple[float, float] = (0.08, 0.22)
#     seam_break_halfwidth: Tuple[int, int] = (2, 5)
#     missing_stitch_count: Tuple[int, int] = (2, 6)

#     # double seam params
#     double_seam_shift_px: Tuple[int, int] = (2, 7)
#     double_seam_alpha: Tuple[float, float] = (0.35, 0.70)

#     # ✅ seam fill/paste (texture-based, NOT blur)
#     seam_fill_offset_px: Tuple[int, int] = (6, 16)        # seam 법선 방향으로 텍스처를 가져올 거리
#     seam_fill_jitter_px: Tuple[int, int] = (-1, 1)        # 약간의 랜덤 jitter
#     seam_mask_feather_blur: Tuple[float, float] = (0.6, 1.4)  # 마스크 경계만 살짝 feather


# # --------------------------
# # utils
# # --------------------------
# def _stable_int_hash(s: str) -> int:
#     h = hashlib.sha256(s.encode("utf-8")).hexdigest()
#     return int(h[:8], 16)

# def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
#     return np.array(img.convert("RGB"), dtype=np.uint8)

# def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
#     arr = np.clip(arr, 0, 255).astype(np.uint8)
#     return Image.fromarray(arr, mode="RGB")

# def _sanitize_filename(s: str) -> str:
#     s = s.replace(os.sep, "_")
#     s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
#     s = re.sub(r"_+", "_", s).strip("_")
#     return s[:180] if len(s) > 180 else s

# def _apply_common_capture_aug(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     out = img

#     if rng.random() < cfg.p_brightness:
#         f = rng.uniform(*cfg.brightness_range)
#         out = ImageEnhance.Brightness(out).enhance(f)

#     if rng.random() < cfg.p_contrast:
#         f = rng.uniform(*cfg.contrast_range)
#         out = ImageEnhance.Contrast(out).enhance(f)

#     if rng.random() < cfg.p_blur:
#         r = rng.uniform(*cfg.blur_radius_range)
#         out = out.filter(ImageFilter.GaussianBlur(radius=r))

#     if rng.random() < cfg.p_noise:
#         arr = _pil_to_np_rgb(out).astype(np.float32)
#         sigma = rng.uniform(*cfg.noise_sigma_range)
#         np_seed = rng.getrandbits(32)
#         noise = np.random.default_rng(np_seed).normal(0.0, sigma, arr.shape).astype(np.float32)
#         arr = arr + noise
#         out = _np_to_pil_rgb(arr)

#     return out

# def _random_polygon_mask(size: Tuple[int, int], rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     w, h = size
#     cx = rng.uniform(0.3*w, 0.7*w)
#     cy = rng.uniform(0.3*h, 0.7*h)
#     r = rng.uniform(0.25*min(w, h), 0.48*min(w, h))
#     n = rng.randint(cfg.poly_min_v, cfg.poly_max_v)

#     pts = []
#     for i in range(n):
#         ang = (2*math.pi*i/n) + rng.uniform(-0.3, 0.3)
#         rr = r * rng.uniform(0.65, 1.0)
#         x = cx + rr * math.cos(ang)
#         y = cy + rr * math.sin(ang)
#         pts.append((x, y))

#     mask = Image.new("L", (w, h), 0)
#     ImageDraw.Draw(mask).polygon(pts, fill=255)
#     if cfg.feather > 0:
#         mask = mask.filter(ImageFilter.GaussianBlur(radius=cfg.feather))
#     return mask


# # --------------------------
# # seam helpers
# # --------------------------
# def _odd(k: int) -> int:
#     k = int(k)
#     return k if k % 2 == 1 else k + 1

# def _morph_dilate(mask01: np.ndarray, k: int) -> np.ndarray:
#     k = _odd(k)
#     t = torch.from_numpy(mask01[None, None, ...].astype(np.float32))
#     y = F.max_pool2d(t, kernel_size=k, stride=1, padding=k//2)
#     return y[0, 0].numpy()

# def _morph_erode(mask01: np.ndarray, k: int) -> np.ndarray:
#     k = _odd(k)
#     t = torch.from_numpy(mask01[None, None, ...].astype(np.float32))
#     y = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=k//2)
#     return y[0, 0].numpy()

# def _morph_close(mask01: np.ndarray, k: int) -> np.ndarray:
#     return _morph_erode(_morph_dilate(mask01, k), k)

# def _morph_open(mask01: np.ndarray, k: int) -> np.ndarray:
#     return _morph_dilate(_morph_erode(mask01, k), k)

# def _estimate_seam_mask_and_axes(
#     img: Image.Image, cfg: PseudoCfg
# ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
#     """
#     seam mask 추정 + PCA로 진행방향(u) / 법선(n) 반환.
#     return:
#       seam_mask01: HxW float [0,1] or None
#       u: (ux,uy) unit vector along seam direction (x,y coords)
#       n: (nx,ny) unit vector normal to seam
#     """
#     arr = _pil_to_np_rgb(img).astype(np.float32) / 255.0
#     gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]

#     thr = float(np.quantile(gray, cfg.seam_quantile))
#     thr = max(thr, cfg.seam_min_thr)
#     mask = (gray >= thr).astype(np.float32)

#     k = _odd(cfg.seam_morph_ksize)
#     mask = _morph_close(mask, k)
#     mask = _morph_open(mask, max(3, k//2))

#     mask = (mask > 0.5).astype(np.float32)
#     if int(mask.sum()) < cfg.seam_min_pixels:
#         return None, None, None

#     ys, xs = np.where(mask > 0.5)
#     coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # (N,2) in (x,y)

#     c = coords.mean(axis=0, keepdims=True)
#     z = coords - c
#     cov = (z.T @ z) / max(1, (z.shape[0]-1))
#     w, v = np.linalg.eigh(cov)  # ascending
#     dir_vec = v[:, 1]  # largest eigenvalue

#     ux, uy = float(dir_vec[0]), float(dir_vec[1])
#     norm = math.sqrt(ux*ux + uy*uy) + 1e-12
#     ux, uy = ux/norm, uy/norm
#     nx, ny = -uy, ux
#     return mask, (ux, uy), (nx, ny)

# def _shift_mask_or_image(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
#     """
#     arr: HxW or HxWxC float32 (0..255 or 0..1)
#     dx,dy in pixels.
#     """
#     if arr.ndim == 2:
#         h, w = arr.shape
#         t = torch.from_numpy(arr[None, None, ...].astype(np.float32))
#     elif arr.ndim == 3:
#         h, w, c = arr.shape
#         t = torch.from_numpy(arr.transpose(2,0,1)[None, ...].astype(np.float32))  # 1,C,H,W
#     else:
#         raise ValueError(f"Unsupported shape: {arr.shape}")

#     dxn = 2.0 * dx / max(1.0, (w - 1.0))
#     dyn = 2.0 * dy / max(1.0, (h - 1.0))

#     theta = torch.tensor([[[1.0, 0.0, dxn],
#                            [0.0, 1.0, dyn]]], dtype=torch.float32)
#     grid = F.affine_grid(theta, size=t.size(), align_corners=True)
#     y = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)

#     y = y[0].numpy()
#     if arr.ndim == 2:
#         return y[0]
#     return y.transpose(1,2,0)

# def _rect_mask_oriented(
#     h: int, w: int,
#     cx: float, cy: float,
#     ux: float, uy: float,
#     nx: float, ny: float,
#     half_len: float, half_w: float
# ) -> np.ndarray:
#     yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
#     dx = xx - cx
#     dy = yy - cy
#     proj_d = dx*ux + dy*uy
#     proj_n = dx*nx + dy*ny
#     m = (np.abs(proj_d) <= half_len) & (np.abs(proj_n) <= half_w)
#     return m.astype(np.float32)

# def _feather_mask01(mask01: np.ndarray, blur_radius: float) -> np.ndarray:
#     """mask01: HxW in [0,1]. Returns blurred mask in [0,1]."""
#     if blur_radius <= 0:
#         return np.clip(mask01, 0.0, 1.0)
#     m = (np.clip(mask01, 0.0, 1.0) * 255.0).astype(np.uint8)
#     m_pil = Image.fromarray(m, mode="L").filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
#     return (np.array(m_pil, dtype=np.float32) / 255.0).clip(0.0, 1.0)


# # --------------------------
# # ✅ seam-aware defects (texture-based)
# # --------------------------
# def _seam_break(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     """
#     seam을 추정하고, seam 구간 일부를 'cut' 한 다음
#     주변 원단 텍스처(법선 방향으로 shift한 이미지)를 가져와 채움.
#     (블러로 지우는 느낌 최소화)
#     """
#     seam_mask, u, n = _estimate_seam_mask_and_axes(img, cfg)
#     if seam_mask is None or u is None or n is None:
#         return img

#     arr = _pil_to_np_rgb(img).astype(np.float32)  # 0..255
#     h, w = seam_mask.shape

#     # seam 두께를 포함하도록 band로 확장
#     band = _morph_dilate(seam_mask, _odd(cfg.seam_band_dilate))
#     band = (band > 0.5).astype(np.float32)

#     ys, xs = np.where(band > 0.5)
#     if len(xs) < cfg.seam_min_pixels:
#         return img

#     # seam 위 랜덤 지점 선택
#     idx = rng.randrange(0, len(xs))
#     cx, cy = float(xs[idx]), float(ys[idx])

#     ux, uy = u
#     nx, ny = n

#     # 끊을 구간(방향성 사각형)
#     half_len = rng.uniform(*cfg.seam_break_len_ratio) * min(h, w) * 0.5
#     half_w = float(rng.randint(*cfg.seam_break_halfwidth))

#     rect = _rect_mask_oriented(h, w, cx, cy, ux, uy, nx, ny, half_len=half_len, half_w=half_w)
#     break_mask = (rect * band).astype(np.float32)
#     if break_mask.sum() < 10:
#         return img

#     # ✅ 주변 원단 텍스처로 채우기: 법선 방향으로 offset만큼 이동한 픽셀을 복사
#     offset = float(rng.randint(*cfg.seam_fill_offset_px))
#     sign = -1.0 if rng.random() < 0.5 else 1.0
#     jitter = float(rng.randint(*cfg.seam_fill_jitter_px))

#     dx = (nx * offset * sign) + jitter
#     dy = (ny * offset * sign) + jitter

#     fabric_src = _shift_mask_or_image(arr, dx=dx, dy=dy)  # HxWx3

#     # 경계만 feather
#     blur_r = rng.uniform(*cfg.seam_mask_feather_blur)
#     m = _feather_mask01(break_mask, blur_radius=blur_r)[..., None]  # H,W,1

#     out = arr * (1.0 - m) + fabric_src * m
#     return _np_to_pil_rgb(out)

# def _missing_stitch(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     """
#     seam 위 여러 개의 짧은 gap을 만들기.
#     seam-break와 동일한 방식(원단 텍스처 shift)으로 채움.
#     """
#     seam_mask, u, n = _estimate_seam_mask_and_axes(img, cfg)
#     if seam_mask is None or u is None or n is None:
#         return img

#     arr = _pil_to_np_rgb(img).astype(np.float32)  # 0..255
#     h, w = seam_mask.shape

#     band = _morph_dilate(seam_mask, _odd(cfg.seam_band_dilate))
#     band = (band > 0.5).astype(np.float32)

#     ys, xs = np.where(band > 0.5)
#     if len(xs) < cfg.seam_min_pixels:
#         return img

#     ux, uy = u
#     nx, ny = n

#     gaps = rng.randint(*cfg.missing_stitch_count)
#     total_mask = np.zeros_like(band, dtype=np.float32)

#     for _ in range(gaps):
#         idx = rng.randrange(0, len(xs))
#         cx, cy = float(xs[idx]), float(ys[idx])
#         half_len = rng.uniform(*cfg.seam_break_len_ratio) * min(h, w) * 0.25
#         half_w = float(rng.randint(*cfg.seam_break_halfwidth))
#         rect = _rect_mask_oriented(h, w, cx, cy, ux, uy, nx, ny, half_len=half_len, half_w=half_w)
#         total_mask = np.maximum(total_mask, rect * band)

#     if total_mask.sum() < 10:
#         return img

#     # 원단 텍스처 shift로 채우기
#     offset = float(rng.randint(*cfg.seam_fill_offset_px))
#     sign = -1.0 if rng.random() < 0.5 else 1.0
#     jitter = float(rng.randint(*cfg.seam_fill_jitter_px))

#     dx = (nx * offset * sign) + jitter
#     dy = (ny * offset * sign) + jitter

#     fabric_src = _shift_mask_or_image(arr, dx=dx, dy=dy)

#     blur_r = rng.uniform(*cfg.seam_mask_feather_blur)
#     m = _feather_mask01(total_mask, blur_radius=blur_r)[..., None]

#     out = arr * (1.0 - m) + fabric_src * m
#     return _np_to_pil_rgb(out)

# def _double_seam(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     """
#     seam을 추정하고, seam 구간 일부를 그대로 복사해서 법선 방향으로 이동하여 붙임.
#     (흰색 선을 그리는 게 아니라, 봉제선 픽셀 자체를 복제)
#     """
#     seam_mask, u, n = _estimate_seam_mask_and_axes(img, cfg)
#     if seam_mask is None or u is None or n is None:
#         return img

#     arr = _pil_to_np_rgb(img).astype(np.float32)  # 0..255
#     h, w = seam_mask.shape

#     band = _morph_dilate(seam_mask, _odd(cfg.seam_band_dilate))
#     band = (band > 0.5).astype(np.float32)

#     ys, xs = np.where(band > 0.5)
#     if len(xs) < cfg.seam_min_pixels:
#         return img

#     idx = rng.randrange(0, len(xs))
#     cx, cy = float(xs[idx]), float(ys[idx])

#     ux, uy = u
#     nx, ny = n

#     # seam 방향의 일부 구간만 복사
#     half_len = rng.uniform(*cfg.seam_break_len_ratio) * min(h, w) * 0.6
#     half_w = float(rng.randint(*cfg.seam_break_halfwidth) + 2)

#     rect = _rect_mask_oriented(h, w, cx, cy, ux, uy, nx, ny, half_len=half_len, half_w=half_w)
#     seam_seg = (rect * band).astype(np.float32)
#     if seam_seg.sum() < 10:
#         return img

#     # seam 픽셀만 추출
#     seam_only = arr * seam_seg[..., None]  # H,W,3

#     # 붙일 위치: 법선 방향 shift
#     shift = float(rng.randint(*cfg.double_seam_shift_px))
#     sign = -1.0 if rng.random() < 0.5 else 1.0
#     dx = nx * shift * sign
#     dy = ny * shift * sign

#     seam_only_shifted = _shift_mask_or_image(seam_only, dx=dx, dy=dy)
#     seam_seg_shifted = _shift_mask_or_image(seam_seg, dx=dx, dy=dy)
#     seam_seg_shifted = np.clip(seam_seg_shifted, 0.0, 1.0)

#     alpha = float(rng.uniform(*cfg.double_seam_alpha))
#     blur_r = float(rng.uniform(*cfg.seam_mask_feather_blur))
#     m = _feather_mask01(seam_seg_shifted, blur_radius=blur_r)[..., None] * alpha

#     out = arr * (1.0 - m) + seam_only_shifted * m
#     return _np_to_pil_rgb(out)


# # --------------------------
# # non seam-aware pseudos
# # --------------------------
# def _cutpaste_like(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     arr = _pil_to_np_rgb(img).astype(np.float32)
#     h, w = arr.shape[:2]
#     m = min(h, w)

#     psize = rng.uniform(cfg.patch_min_ratio, cfg.patch_max_ratio) * m
#     ph = max(6, int(psize * rng.uniform(0.7, 1.3)))
#     pw = max(6, int(psize * rng.uniform(0.7, 1.3)))
#     ph = min(ph, h-2)
#     pw = min(pw, w-2)

#     sy = rng.randint(0, h - ph)
#     sx = rng.randint(0, w - pw)
#     patch = arr[sy:sy+ph, sx:sx+pw].copy()

#     patch_img = _np_to_pil_rgb(patch.astype(np.uint8))
#     rot = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
#     patch_img = patch_img.convert("RGBA").rotate(rot, resample=Image.BILINEAR, expand=True)
#     patch_rgba = np.array(patch_img, dtype=np.float32)  # (H2,W2,4)

#     th2, tw2 = patch_rgba.shape[0], patch_rgba.shape[1]
#     dy = rng.randint(0, max(0, h - th2))
#     dx = rng.randint(0, max(0, w - tw2))

#     mask = _random_polygon_mask((tw2, th2), rng, cfg)
#     mask_np = (np.array(mask, dtype=np.float32) / 255.0)[..., None]

#     alpha_rot = (patch_rgba[..., 3:4] / 255.0)
#     alpha = alpha_rot * mask_np
#     patch_rgb = patch_rgba[..., :3]

#     roi = arr[dy:dy+th2, dx:dx+tw2]
#     if roi.shape[0] == th2 and roi.shape[1] == tw2:
#         eps = 1e-6
#         pm = patch_rgb.mean(axis=(0,1), keepdims=True)
#         ps = patch_rgb.std(axis=(0,1), keepdims=True) + eps
#         rm = roi.mean(axis=(0,1), keepdims=True)
#         rs = roi.std(axis=(0,1), keepdims=True) + eps
#         patch_rgb = (patch_rgb - pm) / ps * rs + rm

#         blended = patch_rgb * alpha + roi * (1.0 - alpha)
#         arr[dy:dy+th2, dx:dx+tw2] = blended

#     return _np_to_pil_rgb(arr)

# def _scratch_thread_line(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     base = img.convert("RGB")
#     overlay = base.copy()
#     draw = ImageDraw.Draw(overlay)

#     w, h = base.size
#     n = rng.randint(*cfg.scratch_count_range)
#     for _ in range(n):
#         x0, y0 = rng.randint(0, w-1), rng.randint(0, h-1)
#         x1, y1 = rng.randint(0, w-1), rng.randint(0, h-1)
#         width = rng.randint(*cfg.scratch_width_range)

#         c = rng.randint(10, 60)
#         color = (c, c, c)

#         if rng.random() < 0.4:
#             t = rng.uniform(0.3, 0.7)
#             xm = int(x0 + (x1-x0)*t)
#             ym = int(y0 + (y1-y0)*t)
#             draw.line([(x0,y0),(xm,ym)], fill=color, width=width)
#         else:
#             draw.line([(x0,y0),(x1,y1)], fill=color, width=width)

#     alpha = rng.uniform(*cfg.scratch_alpha_range)
#     out = Image.blend(base, overlay, alpha=alpha)
#     return out

# def _stain_blob(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     base = img.convert("RGB")
#     w, h = base.size

#     mask = Image.new("L", (w, h), 0)
#     md = ImageDraw.Draw(mask)

#     n = rng.randint(*cfg.stain_blob_count_range)
#     for _ in range(n):
#         cx = rng.randint(0, w-1)
#         cy = rng.randint(0, h-1)
#         rx = rng.randint(int(0.03*w), int(0.12*w))
#         ry = rng.randint(int(0.03*h), int(0.12*h))
#         md.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=rng.randint(120, 255))

#     blur_r = rng.uniform(*cfg.stain_blur_range)
#     mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))

#     tone = rng.choice(["brown", "gray", "dark"])
#     if tone == "brown":
#         color = (rng.randint(60, 120), rng.randint(40, 90), rng.randint(20, 60))
#     elif tone == "gray":
#         g = rng.randint(60, 140)
#         color = (g, g, g)
#     else:
#         d = rng.randint(10, 60)
#         color = (d, d, d)

#     overlay = Image.new("RGB", (w, h), color=color)

#     alpha = rng.uniform(*cfg.stain_alpha_range)
#     base_np = np.array(base, dtype=np.float32)
#     over_np = np.array(overlay, dtype=np.float32)
#     m = (np.array(mask, dtype=np.float32) / 255.0)[..., None] * alpha
#     out = base_np * (1.0 - m) + over_np * m
#     return _np_to_pil_rgb(out)

# def _local_warp(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
#     base = _pil_to_np_rgb(img).astype(np.float32) / 255.0
#     h, w = base.shape[:2]
#     x = torch.from_numpy(base).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

#     strength = rng.uniform(*cfg.warp_strength_range)
#     k = rng.randint(*cfg.warp_kernel_range)
#     if k % 2 == 0:
#         k += 1

#     g = torch.Generator(device="cpu")
#     g.manual_seed(rng.getrandbits(32))
#     disp = torch.randn(1, 2, h, w, generator=g) * strength

#     pad = k // 2
#     disp = F.avg_pool2d(disp, kernel_size=k, stride=1, padding=pad)

#     yy, xx = torch.meshgrid(
#         torch.linspace(-1, 1, h),
#         torch.linspace(-1, 1, w),
#         indexing="ij",
#     )
#     grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # 1,H,W,2
#     grid = grid + disp.permute(0, 2, 3, 1)

#     warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
#     warped = warped.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
#     return _np_to_pil_rgb((warped * 255.0).astype(np.uint8))


# # --------------------------
# # main entry
# # --------------------------
# def make_pseudo(img: Image.Image, seed: Optional[int] = None, cfg: Optional[PseudoCfg] = None) -> Image.Image:
#     """
#     ✅ 최소 1개 이상은 항상 적용되도록 보장.
#     seam이 잡히면 seam 기반을 우선 시도하고,
#     실패/미적용이면 일반 pseudo로 fallback.
#     """
#     cfg = cfg or PseudoCfg()
#     rng = random.Random(seed if seed is not None else random.randint(0, 2**31 - 1))

#     out = img.convert("RGB")
#     out = _apply_common_capture_aug(out, rng, cfg)

#     applied = False

#     # 1) seam-aware 우선 시도
#     if rng.random() < cfg.p_seam_defect:
#         seam_applied = False

#         if rng.random() < cfg.p_missing_stitch:
#             out2 = _missing_stitch(out, rng, cfg)
#             if out2 is not out:
#                 out = out2
#                 seam_applied = True

#         if rng.random() < cfg.p_seam_break:
#             out2 = _seam_break(out, rng, cfg)
#             if out2 is not out:
#                 out = out2
#                 seam_applied = True

#         if rng.random() < cfg.p_double_seam:
#             out2 = _double_seam(out, rng, cfg)
#             if out2 is not out:
#                 out = out2
#                 seam_applied = True

#         if seam_applied:
#             applied = True

#     # 2) fallback pseudos
#     if rng.random() < cfg.p_cutpaste:
#         out = _cutpaste_like(out, rng, cfg); applied = True
#     if rng.random() < cfg.p_scratch_line:
#         out = _scratch_thread_line(out, rng, cfg); applied = True
#     if rng.random() < cfg.p_stain:
#         out = _stain_blob(out, rng, cfg); applied = True
#     if rng.random() < cfg.p_local_warp:
#         out = _local_warp(out, rng, cfg); applied = True

#     # ✅ 최소 1개 적용 보장
#     if not applied:
#         out2 = _seam_break(out, rng, cfg)
#         if out2 is not out:
#             out = out2
#         else:
#             out = _cutpaste_like(out, rng, cfg)

#     return out


# # --------------------------
# # tensor [0,1] <-> PIL
# # --------------------------
# def tensor01_to_pil(img_chw_01: torch.Tensor) -> Image.Image:
#     """
#     img_chw_01: (3,H,W), float in [0,1]
#     """
#     x = img_chw_01.detach().cpu().float().clamp(0, 1)
#     arr = (x * 255.0).byte().permute(1, 2, 0).numpy()
#     return Image.fromarray(arr, mode="RGB")

# def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
#     """
#     returns (3,H,W) float in [0,1]
#     """
#     arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
#     return torch.from_numpy(arr).permute(2, 0, 1)


# # --------------------------
# # ✅ REQUIRED BY simplenet.py
# # --------------------------
# def make_pseudo_batch(
#     imgs01: torch.Tensor,
#     seed: Optional[int] = None,
#     cfg: Optional[PseudoCfg] = None,
# ) -> torch.Tensor:
#     """
#     imgs01: (B,3,H,W) float in [0,1]
#     return: (B,3,H,W) float in [0,1]  (device preserved)
#     """
#     if imgs01.ndim != 4 or imgs01.size(1) != 3:
#         raise ValueError(f"make_pseudo_batch expects (B,3,H,W), got {tuple(imgs01.shape)}")

#     cfg = cfg or PseudoCfg()
#     device = imgs01.device
#     B = imgs01.size(0)

#     base_seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)

#     outs: List[torch.Tensor] = []
#     for i in range(B):
#         per_seed = (base_seed + i * 10007) & 0x7FFFFFFF
#         pil = tensor01_to_pil(imgs01[i])
#         pseudo_pil = make_pseudo(pil, seed=per_seed, cfg=cfg)
#         outs.append(pil_to_tensor01(pseudo_pil))

#     out = torch.stack(outs, dim=0).to(device=device, dtype=imgs01.dtype)
#     return out


# # --------------------------
# # (optional) normalize-aware helpers
# # --------------------------
# def tensor_to_pil_denorm(img_chw: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> Image.Image:
#     x = img_chw.detach().cpu().float()
#     mean_t = torch.tensor(mean).view(3, 1, 1)
#     std_t = torch.tensor(std).view(3, 1, 1)
#     x = x * std_t + mean_t
#     x = x.clamp(0, 1)
#     x = (x * 255.0).byte().permute(1, 2, 0).numpy()
#     return Image.fromarray(x, mode="RGB")

# def pil_to_tensor_norm(img: Image.Image, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
#     arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
#     x = torch.from_numpy(arr).permute(2, 0, 1)
#     mean_t = torch.tensor(mean).view(3, 1, 1)
#     std_t = torch.tensor(std).view(3, 1, 1)
#     x = (x - mean_t) / std_t
#     return x

# def generate_pseudo_from_tensor(
#     img_chw: torch.Tensor,
#     save_path: str,
#     seed: int,
#     mean: Sequence[float],
#     std: Sequence[float],
#     cfg: Optional[PseudoCfg] = None,
# ) -> torch.Tensor:
#     cfg = cfg or PseudoCfg()
#     pil = tensor_to_pil_denorm(img_chw, mean, std)
#     pseudo = make_pseudo(pil, seed=seed, cfg=cfg)

#     Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
#     pseudo.save(save_path)

#     return pil_to_tensor_norm(pseudo, mean, std)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


# --------------------------
# Config
# --------------------------
@dataclass
class PseudoCfg:
    # common capture aug (mask에는 포함 안됨)
    p_brightness: float = 0.6
    p_contrast: float = 0.6
    p_blur: float = 0.2
    p_noise: float = 0.4

    brightness_range: Tuple[float, float] = (0.85, 1.15)
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    blur_radius_range: Tuple[float, float] = (0.3, 1.2)
    noise_sigma_range: Tuple[float, float] = (2.0, 8.0)  # in [0..255] domain

    # generic pseudo types
    p_cutpaste: float = 0.65
    p_scratch_line: float = 0.35
    p_stain: float = 0.25
    p_local_warp: float = 0.15

    # seam-aware pseudo types
    p_seam_defect: float = 0.75
    p_seam_break: float = 0.55
    p_double_seam: float = 0.35
    p_missing_stitch: float = 0.30

    # cutpaste params
    patch_min_ratio: float = 0.10
    patch_max_ratio: float = 0.35
    rotate_deg: float = 20.0
    feather: float = 2.0
    poly_min_v: int = 5
    poly_max_v: int = 9

    # scratch params
    scratch_count_range: Tuple[int, int] = (1, 3)
    scratch_width_range: Tuple[int, int] = (1, 3)
    scratch_alpha_range: Tuple[float, float] = (0.25, 0.55)
    scratch_mask_blur: Tuple[float, float] = (0.6, 1.3)

    # stain params
    stain_blob_count_range: Tuple[int, int] = (3, 8)
    stain_alpha_range: Tuple[float, float] = (0.10, 0.35)
    stain_blur_range: Tuple[float, float] = (3.0, 9.0)

    # warp params
    warp_strength_range: Tuple[float, float] = (0.003, 0.010)
    warp_kernel_range: Tuple[int, int] = (9, 21)
    warp_mask_quantile: float = 0.985
    warp_mask_min_area: int = 200

    # seam detection params (white stitch on dark fabric)
    seam_quantile: float = 0.985
    seam_min_thr: float = 0.60
    seam_morph_ksize: int = 7
    seam_min_pixels: int = 80
    seam_band_dilate: int = 9

    # seam break / missing stitch params
    seam_break_len_ratio: Tuple[float, float] = (0.08, 0.22)
    seam_break_halfwidth: Tuple[int, int] = (2, 5)
    missing_stitch_count: Tuple[int, int] = (2, 6)

    # double seam params
    double_seam_shift_px: Tuple[int, int] = (2, 7)
    double_seam_alpha: Tuple[float, float] = (0.35, 0.70)

    # seam fill/paste (texture-based, NOT blur)
    seam_fill_offset_px: Tuple[int, int] = (6, 16)
    seam_fill_jitter_px: Tuple[int, int] = (-1, 1)
    seam_mask_feather_blur: Tuple[float, float] = (0.6, 1.4)


# --------------------------
# utils
# --------------------------
def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)

def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def _to_mask01(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    return np.clip(mask, 0.0, 1.0)

def _apply_common_capture_aug(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Image.Image:
    out = img

    if rng.random() < cfg.p_brightness:
        f = rng.uniform(*cfg.brightness_range)
        out = ImageEnhance.Brightness(out).enhance(f)

    if rng.random() < cfg.p_contrast:
        f = rng.uniform(*cfg.contrast_range)
        out = ImageEnhance.Contrast(out).enhance(f)

    if rng.random() < cfg.p_blur:
        r = rng.uniform(*cfg.blur_radius_range)
        out = out.filter(ImageFilter.GaussianBlur(radius=r))

    if rng.random() < cfg.p_noise:
        arr = _pil_to_np_rgb(out).astype(np.float32)
        sigma = rng.uniform(*cfg.noise_sigma_range)
        np_seed = rng.getrandbits(32)
        noise = np.random.default_rng(np_seed).normal(0.0, sigma, arr.shape).astype(np.float32)
        arr = arr + noise
        out = _np_to_pil_rgb(arr)

    return out

def _random_polygon_mask(size: Tuple[int, int], rng: random.Random, cfg: PseudoCfg) -> Image.Image:
    w, h = size
    cx = rng.uniform(0.3*w, 0.7*w)
    cy = rng.uniform(0.3*h, 0.7*h)
    r = rng.uniform(0.25*min(w, h), 0.48*min(w, h))
    n = rng.randint(cfg.poly_min_v, cfg.poly_max_v)

    pts = []
    for i in range(n):
        ang = (2*math.pi*i/n) + rng.uniform(-0.3, 0.3)
        rr = r * rng.uniform(0.65, 1.0)
        x = cx + rr * math.cos(ang)
        y = cy + rr * math.sin(ang)
        pts.append((x, y))

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    if cfg.feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=cfg.feather))
    return mask

def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1

def _morph_dilate(mask01: np.ndarray, k: int) -> np.ndarray:
    k = _odd(k)
    t = torch.from_numpy(mask01[None, None, ...].astype(np.float32))
    y = F.max_pool2d(t, kernel_size=k, stride=1, padding=k//2)
    return y[0, 0].numpy()

def _morph_erode(mask01: np.ndarray, k: int) -> np.ndarray:
    k = _odd(k)
    t = torch.from_numpy(mask01[None, None, ...].astype(np.float32))
    y = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=k//2)
    return y[0, 0].numpy()

def _morph_close(mask01: np.ndarray, k: int) -> np.ndarray:
    return _morph_erode(_morph_dilate(mask01, k), k)

def _morph_open(mask01: np.ndarray, k: int) -> np.ndarray:
    return _morph_dilate(_morph_erode(mask01, k), k)

def _feather_mask01(mask01: np.ndarray, blur_radius: float) -> np.ndarray:
    if blur_radius <= 0:
        return _to_mask01(mask01)
    m = (np.clip(mask01, 0, 1) * 255.0).astype(np.uint8)
    m_pil = Image.fromarray(m, mode="L").filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return (np.array(m_pil, dtype=np.float32) / 255.0).clip(0.0, 1.0)

def _shift_mask_or_image(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
        t = torch.from_numpy(arr[None, None, ...].astype(np.float32))
    elif arr.ndim == 3:
        h, w, c = arr.shape
        t = torch.from_numpy(arr.transpose(2,0,1)[None, ...].astype(np.float32))
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")

    dxn = 2.0 * dx / max(1.0, (w - 1.0))
    dyn = 2.0 * dy / max(1.0, (h - 1.0))

    theta = torch.tensor([[[1.0, 0.0, dxn],
                           [0.0, 1.0, dyn]]], dtype=torch.float32)
    grid = F.affine_grid(theta, size=t.size(), align_corners=True)
    y = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)

    y = y[0].numpy()
    if arr.ndim == 2:
        return y[0]
    return y.transpose(1,2,0)

def _rect_mask_oriented(
    h: int, w: int,
    cx: float, cy: float,
    ux: float, uy: float,
    nx: float, ny: float,
    half_len: float, half_w: float
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    proj_d = dx*ux + dy*uy
    proj_n = dx*nx + dy*ny
    m = (np.abs(proj_d) <= half_len) & (np.abs(proj_n) <= half_w)
    return m.astype(np.float32)

def _estimate_seam_mask_and_axes(
    img: Image.Image, cfg: PseudoCfg
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    arr = _pil_to_np_rgb(img).astype(np.float32) / 255.0
    gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]

    thr = float(np.quantile(gray, cfg.seam_quantile))
    thr = max(thr, cfg.seam_min_thr)
    mask = (gray >= thr).astype(np.float32)

    k = _odd(cfg.seam_morph_ksize)
    mask = _morph_close(mask, k)
    mask = _morph_open(mask, max(3, k//2))

    mask = (mask > 0.5).astype(np.float32)
    if int(mask.sum()) < cfg.seam_min_pixels:
        return None, None, None

    ys, xs = np.where(mask > 0.5)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    c = coords.mean(axis=0, keepdims=True)
    z = coords - c
    cov = (z.T @ z) / max(1, (z.shape[0]-1))
    w, v = np.linalg.eigh(cov)
    dir_vec = v[:, 1]

    ux, uy = float(dir_vec[0]), float(dir_vec[1])
    norm = math.sqrt(ux*ux + uy*uy) + 1e-12
    ux, uy = ux/norm, uy/norm
    nx, ny = -uy, ux
    return mask, (ux, uy), (nx, ny)


# --------------------------
# seam-aware defects (returns img, mask01)
# --------------------------
def _seam_break(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    seam_mask, u, n = _estimate_seam_mask_and_axes(img, cfg)
    if seam_mask is None or u is None or n is None:
        return img, np.zeros((img.size[1], img.size[0]), dtype=np.float32)

    arr = _pil_to_np_rgb(img).astype(np.float32)
    h, w = seam_mask.shape

    band = _morph_dilate(seam_mask, _odd(cfg.seam_band_dilate))
    band = (band > 0.5).astype(np.float32)

    ys, xs = np.where(band > 0.5)
    if len(xs) < cfg.seam_min_pixels:
        return img, np.zeros((h, w), dtype=np.float32)

    idx = rng.randrange(0, len(xs))
    cx, cy = float(xs[idx]), float(ys[idx])

    ux, uy = u
    nx, ny = n

    half_len = rng.uniform(*cfg.seam_break_len_ratio) * min(h, w) * 0.5
    half_w = float(rng.randint(*cfg.seam_break_halfwidth))

    rect = _rect_mask_oriented(h, w, cx, cy, ux, uy, nx, ny, half_len=half_len, half_w=half_w)
    break_mask = (rect * band).astype(np.float32)
    if break_mask.sum() < 10:
        return img, np.zeros((h, w), dtype=np.float32)

    offset = float(rng.randint(*cfg.seam_fill_offset_px))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    jitter = float(rng.randint(*cfg.seam_fill_jitter_px))

    dx = (nx * offset * sign) + jitter
    dy = (ny * offset * sign) + jitter

    fabric_src = _shift_mask_or_image(arr, dx=dx, dy=dy)

    blur_r = rng.uniform(*cfg.seam_mask_feather_blur)
    m = _feather_mask01(break_mask, blur_radius=blur_r)[..., None]

    out = arr * (1.0 - m) + fabric_src * m
    return _np_to_pil_rgb(out), _to_mask01(break_mask)

def _missing_stitch(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    seam_mask, u, n = _estimate_seam_mask_and_axes(img, cfg)
    if seam_mask is None or u is None or n is None:
        return img, np.zeros((img.size[1], img.size[0]), dtype=np.float32)

    arr = _pil_to_np_rgb(img).astype(np.float32)
    h, w = seam_mask.shape

    band = _morph_dilate(seam_mask, _odd(cfg.seam_band_dilate))
    band = (band > 0.5).astype(np.float32)

    ys, xs = np.where(band > 0.5)
    if len(xs) < cfg.seam_min_pixels:
        return img, np.zeros((h, w), dtype=np.float32)

    ux, uy = u
    nx, ny = n

    gaps = rng.randint(*cfg.missing_stitch_count)
    total_mask = np.zeros_like(band, dtype=np.float32)

    for _ in range(gaps):
        idx = rng.randrange(0, len(xs))
        cx, cy = float(xs[idx]), float(ys[idx])
        half_len = rng.uniform(*cfg.seam_break_len_ratio) * min(h, w) * 0.25
        half_w = float(rng.randint(*cfg.seam_break_halfwidth))
        rect = _rect_mask_oriented(h, w, cx, cy, ux, uy, nx, ny, half_len=half_len, half_w=half_w)
        total_mask = np.maximum(total_mask, rect * band)

    if total_mask.sum() < 10:
        return img, np.zeros((h, w), dtype=np.float32)

    offset = float(rng.randint(*cfg.seam_fill_offset_px))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    jitter = float(rng.randint(*cfg.seam_fill_jitter_px))

    dx = (nx * offset * sign) + jitter
    dy = (ny * offset * sign) + jitter

    fabric_src = _shift_mask_or_image(arr, dx=dx, dy=dy)

    blur_r = rng.uniform(*cfg.seam_mask_feather_blur)
    m = _feather_mask01(total_mask, blur_radius=blur_r)[..., None]

    out = arr * (1.0 - m) + fabric_src * m
    return _np_to_pil_rgb(out), _to_mask01(total_mask)

def _double_seam(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    seam_mask, u, n = _estimate_seam_mask_and_axes(img, cfg)
    if seam_mask is None or u is None or n is None:
        return img, np.zeros((img.size[1], img.size[0]), dtype=np.float32)

    arr = _pil_to_np_rgb(img).astype(np.float32)
    h, w = seam_mask.shape

    band = _morph_dilate(seam_mask, _odd(cfg.seam_band_dilate))
    band = (band > 0.5).astype(np.float32)

    ys, xs = np.where(band > 0.5)
    if len(xs) < cfg.seam_min_pixels:
        return img, np.zeros((h, w), dtype=np.float32)

    idx = rng.randrange(0, len(xs))
    cx, cy = float(xs[idx]), float(ys[idx])

    ux, uy = u
    nx, ny = n

    half_len = rng.uniform(*cfg.seam_break_len_ratio) * min(h, w) * 0.6
    half_w = float(rng.randint(*cfg.seam_break_halfwidth) + 2)

    rect = _rect_mask_oriented(h, w, cx, cy, ux, uy, nx, ny, half_len=half_len, half_w=half_w)
    seam_seg = (rect * band).astype(np.float32)
    if seam_seg.sum() < 10:
        return img, np.zeros((h, w), dtype=np.float32)

    seam_only = arr * seam_seg[..., None]

    shift = float(rng.randint(*cfg.double_seam_shift_px))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    dx = nx * shift * sign
    dy = ny * shift * sign

    seam_only_shifted = _shift_mask_or_image(seam_only, dx=dx, dy=dy)
    seam_seg_shifted = _shift_mask_or_image(seam_seg, dx=dx, dy=dy)
    seam_seg_shifted = np.clip(seam_seg_shifted, 0.0, 1.0)

    alpha = float(rng.uniform(*cfg.double_seam_alpha))
    blur_r = float(rng.uniform(*cfg.seam_mask_feather_blur))
    m = _feather_mask01(seam_seg_shifted, blur_radius=blur_r)[..., None] * alpha

    out = arr * (1.0 - m) + seam_only_shifted * m
    return _np_to_pil_rgb(out), _to_mask01(seam_seg_shifted)


# --------------------------
# non seam-aware pseudos (returns img, mask01)
# --------------------------
def _cutpaste_like(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    arr = _pil_to_np_rgb(img).astype(np.float32)
    h, w = arr.shape[:2]
    mmin = min(h, w)

    psize = rng.uniform(cfg.patch_min_ratio, cfg.patch_max_ratio) * mmin
    ph = max(6, int(psize * rng.uniform(0.7, 1.3)))
    pw = max(6, int(psize * rng.uniform(0.7, 1.3)))
    ph = min(ph, h-2)
    pw = min(pw, w-2)

    sy = rng.randint(0, h - ph)
    sx = rng.randint(0, w - pw)
    patch = arr[sy:sy+ph, sx:sx+pw].copy()

    patch_img = _np_to_pil_rgb(patch.astype(np.uint8))
    rot = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
    patch_img = patch_img.convert("RGBA").rotate(rot, resample=Image.BILINEAR, expand=True)
    patch_rgba = np.array(patch_img, dtype=np.float32)

    th2, tw2 = patch_rgba.shape[0], patch_rgba.shape[1]
    dy = rng.randint(0, max(0, h - th2))
    dx = rng.randint(0, max(0, w - tw2))

    mask = _random_polygon_mask((tw2, th2), rng, cfg)
    mask_np = (np.array(mask, dtype=np.float32) / 255.0)[..., None]

    alpha_rot = (patch_rgba[..., 3:4] / 255.0)
    alpha = alpha_rot * mask_np  # (H2,W2,1)
    patch_rgb = patch_rgba[..., :3]

    roi = arr[dy:dy+th2, dx:dx+tw2]
    if roi.shape[0] == th2 and roi.shape[1] == tw2:
        eps = 1e-6
        pm = patch_rgb.mean(axis=(0,1), keepdims=True)
        ps = patch_rgb.std(axis=(0,1), keepdims=True) + eps
        rm = roi.mean(axis=(0,1), keepdims=True)
        rs = roi.std(axis=(0,1), keepdims=True) + eps
        patch_rgb = (patch_rgb - pm) / ps * rs + rm

        blended = patch_rgb * alpha + roi * (1.0 - alpha)
        arr[dy:dy+th2, dx:dx+tw2] = blended

    # build full-size mask
    full_mask = np.zeros((h, w), dtype=np.float32)
    a2 = alpha[..., 0]
    if 0 <= dy < h and 0 <= dx < w:
        y2 = min(h, dy + th2)
        x2 = min(w, dx + tw2)
        full_mask[dy:y2, dx:x2] = np.maximum(full_mask[dy:y2, dx:x2], a2[:(y2-dy), :(x2-dx)])

    return _np_to_pil_rgb(arr), _to_mask01(full_mask)

def _scratch_thread_line(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    base = img.convert("RGB")
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)

    n = rng.randint(*cfg.scratch_count_range)
    for _ in range(n):
        x0, y0 = rng.randint(0, w-1), rng.randint(0, h-1)
        x1, y1 = rng.randint(0, w-1), rng.randint(0, h-1)
        width = rng.randint(*cfg.scratch_width_range)

        c = rng.randint(10, 60)
        color = (c, c, c)

        if rng.random() < 0.4:
            t = rng.uniform(0.3, 0.7)
            xm = int(x0 + (x1-x0)*t)
            ym = int(y0 + (y1-y0)*t)
            draw.line([(x0,y0),(xm,ym)], fill=color, width=width)
            md.line([(x0,y0),(xm,ym)], fill=255, width=max(1, width+1))
        else:
            draw.line([(x0,y0),(x1,y1)], fill=color, width=width)
            md.line([(x0,y0),(x1,y1)], fill=255, width=max(1, width+1))

    alpha = rng.uniform(*cfg.scratch_alpha_range)
    out = Image.blend(base, overlay, alpha=alpha)

    blur_r = rng.uniform(*cfg.scratch_mask_blur)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=float(blur_r)))
    mask01 = (np.array(mask, dtype=np.float32) / 255.0).clip(0, 1)
    return out, mask01

def _stain_blob(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    base = img.convert("RGB")
    w, h = base.size

    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)

    n = rng.randint(*cfg.stain_blob_count_range)
    for _ in range(n):
        cx = rng.randint(0, w-1)
        cy = rng.randint(0, h-1)
        rx = rng.randint(int(0.03*w), int(0.12*w))
        ry = rng.randint(int(0.03*h), int(0.12*h))
        md.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=rng.randint(120, 255))

    blur_r = rng.uniform(*cfg.stain_blur_range)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
    mask01 = (np.array(mask, dtype=np.float32) / 255.0).clip(0, 1)

    tone = rng.choice(["brown", "gray", "dark"])
    if tone == "brown":
        color = (rng.randint(60, 120), rng.randint(40, 90), rng.randint(20, 60))
    elif tone == "gray":
        g = rng.randint(60, 140)
        color = (g, g, g)
    else:
        d = rng.randint(10, 60)
        color = (d, d, d)

    overlay = Image.new("RGB", (w, h), color=color)
    alpha = rng.uniform(*cfg.stain_alpha_range)

    base_np = np.array(base, dtype=np.float32)
    over_np = np.array(overlay, dtype=np.float32)
    m = mask01[..., None] * alpha
    out = base_np * (1.0 - m) + over_np * m
    return _np_to_pil_rgb(out), _to_mask01(mask01)

def _local_warp(img: Image.Image, rng: random.Random, cfg: PseudoCfg) -> Tuple[Image.Image, np.ndarray]:
    base_u8 = _pil_to_np_rgb(img).astype(np.float32)
    base = base_u8 / 255.0
    h, w = base.shape[:2]
    x = torch.from_numpy(base).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

    strength = rng.uniform(*cfg.warp_strength_range)
    k = rng.randint(*cfg.warp_kernel_range)
    if k % 2 == 0:
        k += 1

    g = torch.Generator(device="cpu")
    g.manual_seed(rng.getrandbits(32))
    disp = torch.randn(1, 2, h, w, generator=g) * strength

    pad = k // 2
    disp = F.avg_pool2d(disp, kernel_size=k, stride=1, padding=pad)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    grid = grid + disp.permute(0, 2, 3, 1)

    warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    warped_np = warped.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    out_u8 = (warped_np * 255.0).astype(np.uint8)
    out = Image.fromarray(out_u8, mode="RGB")

    # mask: diff 기반 (상위 quantile)
    diff = np.abs(warped_np - base).mean(axis=2)  # (H,W)
    thr = float(np.quantile(diff, cfg.warp_mask_quantile))
    mask = (diff >= thr).astype(np.float32)
    mask = _morph_close(mask, 5)
    mask = _morph_open(mask, 3)

    if int(mask.sum()) < cfg.warp_mask_min_area:
        mask = np.zeros((h, w), dtype=np.float32)

    return out, _to_mask01(mask)


# --------------------------
# main entry (img + mask)
# --------------------------
def make_pseudo_with_mask(
    img: Image.Image,
    seed: Optional[int] = None,
    cfg: Optional[PseudoCfg] = None,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Returns:
      pseudo_img: PIL RGB
      mask01: HxW float in [0,1] (pseudo defect region union)
    """
    cfg = cfg or PseudoCfg()
    rng = random.Random(seed if seed is not None else random.randint(0, 2**31 - 1))

    out = img.convert("RGB")
    H, W = out.size[1], out.size[0]
    mask01 = np.zeros((H, W), dtype=np.float32)

    # common capture aug (mask에는 포함 안함)
    out = _apply_common_capture_aug(out, rng, cfg)

    applied = False

    # seam-aware
    if rng.random() < cfg.p_seam_defect:
        seam_applied = False

        if rng.random() < cfg.p_missing_stitch:
            out2, m2 = _missing_stitch(out, rng, cfg)
            if out2 is not out:
                out = out2
                mask01 = np.maximum(mask01, m2)
                seam_applied = True

        if rng.random() < cfg.p_seam_break:
            out2, m2 = _seam_break(out, rng, cfg)
            if out2 is not out:
                out = out2
                mask01 = np.maximum(mask01, m2)
                seam_applied = True

        if rng.random() < cfg.p_double_seam:
            out2, m2 = _double_seam(out, rng, cfg)
            if out2 is not out:
                out = out2
                mask01 = np.maximum(mask01, m2)
                seam_applied = True

        if seam_applied:
            applied = True

    # fallback pseudos
    if rng.random() < cfg.p_cutpaste:
        out, m2 = _cutpaste_like(out, rng, cfg)
        mask01 = np.maximum(mask01, m2)
        applied = True

    if rng.random() < cfg.p_scratch_line:
        out, m2 = _scratch_thread_line(out, rng, cfg)
        mask01 = np.maximum(mask01, m2)
        applied = True

    if rng.random() < cfg.p_stain:
        out, m2 = _stain_blob(out, rng, cfg)
        mask01 = np.maximum(mask01, m2)
        applied = True

    if rng.random() < cfg.p_local_warp:
        out, m2 = _local_warp(out, rng, cfg)
        mask01 = np.maximum(mask01, m2)
        applied = True

    # ensure at least one
    if not applied:
        out2, m2 = _seam_break(out, rng, cfg)
        if out2 is not out and m2.sum() > 0:
            out = out2
            mask01 = np.maximum(mask01, m2)
        else:
            out, m2 = _cutpaste_like(out, rng, cfg)
            mask01 = np.maximum(mask01, m2)

    mask01 = _to_mask01(mask01)
    return out, mask01


def make_pseudo(img: Image.Image, seed: Optional[int] = None, cfg: Optional[PseudoCfg] = None) -> Image.Image:
    """Backward compatible: returns only image."""
    out, _ = make_pseudo_with_mask(img, seed=seed, cfg=cfg)
    return out


# --------------------------
# tensor [0,1] <-> PIL
# --------------------------
def tensor01_to_pil(img_chw_01: torch.Tensor) -> Image.Image:
    x = img_chw_01.detach().cpu().float().clamp(0, 1)
    arr = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")

def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def mask01_to_tensor01(mask01_hw: np.ndarray) -> torch.Tensor:
    m = np.clip(mask01_hw.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(m)[None, ...]  # (1,H,W)


# --------------------------
# ✅ REQUIRED BY simplenet.py
# --------------------------
def make_pseudo_batch(
    imgs01: torch.Tensor,
    seed: Optional[int] = None,
    cfg: Optional[PseudoCfg] = None,
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    imgs01: (B,3,H,W) float in [0,1]
    return_mask=False:
      return: (B,3,H,W) float in [0,1]
    return_mask=True:
      return: (pseudo_imgs01, pseudo_masks01)
        pseudo_imgs01:  (B,3,H,W) float in [0,1]
        pseudo_masks01: (B,1,H,W) float in [0,1]
    """
    if imgs01.ndim != 4 or imgs01.size(1) != 3:
        raise ValueError(f"make_pseudo_batch expects (B,3,H,W), got {tuple(imgs01.shape)}")

    cfg = cfg or PseudoCfg()
    device = imgs01.device
    B = imgs01.size(0)

    base_seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)

    outs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for i in range(B):
        per_seed = (base_seed + i * 10007) & 0x7FFFFFFF

        pil = tensor01_to_pil(imgs01[i])
        pseudo_pil, mask01 = make_pseudo_with_mask(pil, seed=per_seed, cfg=cfg)

        outs.append(pil_to_tensor01(pseudo_pil))
        if return_mask:
            masks.append(mask01_to_tensor01(mask01))

    out = torch.stack(outs, dim=0).to(device=device, dtype=imgs01.dtype)
    if not return_mask:
        return out

    m = torch.stack(masks, dim=0).to(device=device, dtype=imgs01.dtype)  # (B,1,H,W)
    return out, m


# --------------------------
# (optional) normalize-aware helpers (kept)
# --------------------------
def tensor_to_pil_denorm(img_chw: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> Image.Image:
    x = img_chw.detach().cpu().float()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    x = x * std_t + mean_t
    x = x.clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")

def pil_to_tensor_norm(img: Image.Image, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x
