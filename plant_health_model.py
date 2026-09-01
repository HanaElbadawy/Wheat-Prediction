"""
Plant health / crop-vigor scanner — SegFormer (MiT-B0) inference module.

WHAT THIS MODEL ACTUALLY IS
----------------------------
This is NOT a leaf-photo disease classifier. It is a per-pixel semantic
segmentation model trained on 9-band multispectral UAV (drone) imagery:

    Blue, Green, Red, RedEdge, NIR, NDVI, NDRE, CI_RedEdge, GNDVI

...at 224x224 patches, predicting one of three crop-vigor classes per pixel,
derived from Canopy Chlorophyll Content (CCC):

    0 = Low     1 = Medium     2 = High

It cannot be run on an ordinary phone photo (3-band RGB, no calibrated
reflectance, no NIR/RedEdge bands, no pre-computed vegetation indices).
The original page contract in this app (single label + confidence for a
leaf photo) does not fit this model, so the scanner endpoint and page were
rewritten around what the model actually does, rather than forcing this
model to answer a question it cannot answer.

DEPENDENCY VERSION WARNING
----------------------------
The checkpoint was trained against an older internal module layout inside
`transformers` (`segformer.encoder.patch_embeddings` / `.block` / naming
for attention as `query`/`key`/`value`). Recent `transformers` releases
(5.x) restructured these internals (`segformer.stages[i]...`, attention
renamed to `q_proj`/`k_proj`/`v_proj`). Loading this checkpoint under a
mismatched version does NOT raise an error — `load_state_dict(strict=False)`
silently drops ~160 tensors and predictions collapse to a single class.
`transformers==4.44.2` is confirmed (tested against all 5 delivered
samples) to load with zero missing/unexpected keys. Do not upgrade this
dependency without re-validating against the sample data below.
"""

from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

BASE = Path(__file__).parent
CHECKPOINT_PATH = BASE / "models" / "best_segformer.pth"
SAMPLE_DIR = BASE / "sample_data"

NUM_CLASSES = 3
IN_CHANNELS = 9
PATCH_SIZE = 224
CLIP_RANGE = (-10.0, 10.0)
NODATA_TARGET_VALUE = 15

# Ground Sampling Distance — real-world size of one pixel edge, in metres.
# 7 cm/pixel is the resolution these UAV captures were flown at. If a batch
# of imagery is captured at a different altitude/sensor setup, update this
# (or pass gsd_m explicitly into predict()) — area is wrong otherwise.
GSD_METERS = 0.07
PIXEL_AREA_M2 = GSD_METERS ** 2  # 0.0049 m² per pixel at 7 cm GSD

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}
CLASS_COLORS = {0: (214, 39, 40), 1: (255, 127, 14), 2: (44, 160, 44)}  # red / orange / green
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "NDVI", "NDRE", "CI_RedEdge", "GNDVI"]

# Reported in README.txt, full held-out test set (not just the 5 samples here).
REPORTED_METRICS = {"mIoU": 0.480, "pixel_accuracy": 0.620, "test_loss": 0.49}


def _build_model() -> nn.Module:
    config = SegformerConfig(num_labels=NUM_CLASSES)
    model = SegformerForSemanticSegmentation(config)

    # Same first-layer expansion used in training: average the pretrained
    # 3-channel RGB kernel and repeat it across all 9 input channels before
    # loading the fine-tuned weights over it.
    proj = model.segformer.encoder.patch_embeddings[0].proj
    new_proj = nn.Conv2d(
        IN_CHANNELS, proj.out_channels, kernel_size=proj.kernel_size,
        stride=proj.stride, padding=proj.padding, bias=(proj.bias is not None),
    )
    model.segformer.encoder.patch_embeddings[0].proj = new_proj
    return model


@lru_cache(maxsize=1)
def load_model() -> nn.Module:
    """Load once, cache for the process lifetime."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Expected "
            "models/best_segformer.pth to be bundled with the app."
        )
    model = _build_model()
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # Checkpoint keys are prefixed "model." (saved from a wrapper class
    # that held SegformerForSemanticSegmentation as `self.model`).
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint did not load cleanly ({len(missing)} missing, "
            f"{len(unexpected)} unexpected tensors). This almost always means "
            "the installed `transformers` version doesn't match the one this "
            "checkpoint was trained under — pin transformers==4.44.2."
        )
    model.eval()
    return model


def read_multiband_tif(data: bytes) -> np.ndarray:
    """Read a 9-band GeoTIFF from raw bytes -> (9, H, W) float32 array."""
    import rasterio
    import io

    with rasterio.MemoryFile(data) as memfile:
        with memfile.open() as src:
            if src.count < IN_CHANNELS:
                raise ValueError(
                    f"This file has {src.count} band(s); the model needs "
                    f"{IN_CHANNELS} ({', '.join(BAND_NAMES)}). This looks like "
                    "a regular photo, not a multispectral UAV capture — see "
                    "the scanner page for what input this model expects."
                )
            feature = src.read().astype(np.float32)[:IN_CHANNELS]
            nodata = src.nodata
    if nodata is not None:
        feature[feature == nodata] = 0.0
    feature = np.nan_to_num(feature, nan=0.0, posinf=CLIP_RANGE[1], neginf=CLIP_RANGE[0])
    feature = np.clip(feature, CLIP_RANGE[0], CLIP_RANGE[1])
    return feature


def predict(feature: np.ndarray, gsd_m: float = GSD_METERS) -> dict:
    """
    feature: (9, H, W) float32, already clipped/cleaned.
    gsd_m: ground sampling distance in metres/pixel (default 7 cm capture).
    Returns per-pixel class map plus a summary, including real-world area
    per class.
    """
    model = load_model()
    h, w = feature.shape[1], feature.shape[2]
    x = torch.from_numpy(feature).unsqueeze(0)

    with torch.no_grad():
        logits = model(pixel_values=x).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        probs = torch.softmax(upsampled, dim=1).squeeze(0).numpy()
        pred = np.argmax(probs, axis=0)

    pixel_area_m2 = gsd_m ** 2
    total = pred.size
    total_area_m2 = total * pixel_area_m2

    class_pixels = {c: int((pred == c).sum()) for c in range(NUM_CLASSES)}
    class_pct = {CLASS_NAMES[c]: round(100 * class_pixels[c] / total, 1) for c in range(NUM_CLASSES)}
    class_area_m2 = {CLASS_NAMES[c]: round(class_pixels[c] * pixel_area_m2, 2) for c in range(NUM_CLASSES)}
    dominant = max(class_pixels, key=class_pixels.get)
    mean_confidence = float(np.mean(np.max(probs, axis=0)))

    return {
        "pred_map": pred,               # (H, W) int array, for the caller to colourise
        "class_pixel_pct": class_pct,
        "class_area_m2": class_area_m2,
        "total_area_m2": round(total_area_m2, 2),
        "gsd_m": gsd_m,
        "dominant_class": CLASS_NAMES[dominant],
        "mean_confidence": round(mean_confidence, 3),
    }


def colourise(pred_map: np.ndarray) -> np.ndarray:
    """(H, W) int array -> (H, W, 3) uint8 RGB image."""
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, colour in CLASS_COLORS.items():
        rgb[pred_map == c] = colour
    return rgb


def list_samples() -> list:
    """Bundled 9-band demo patches a user can try without their own UAV data."""
    if not SAMPLE_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLE_DIR.glob("*_9band.tif"))
