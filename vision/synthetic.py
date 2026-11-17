"""Deterministic rendering for software checks. Pixel sizes do not imply distance."""

import cv2
import numpy as np

PROFILES = {
    "normal": (1.0, (18, 55, 80), False),
    "dark": (0.47, (18, 55, 80), False),
    "glare": (1.18, (18, 55, 80), True),
    "low_contrast": (0.86, (24, 116, 148), True),
}
RESOLUTIONS = ((320, 240), (640, 360), (1280, 720))


def scene(rng, size, profile, kind, center=None):
    width, height = size
    exposure, background_hsv, glare = PROFILES[profile]
    hsv = np.empty((height, width, 3), np.uint8)
    if profile == "low_contrast":
        hsv[:] = background_hsv
    else:
        hsv[:] = [rng.integers(0, 180), rng.integers(20, 110), rng.integers(35, 130)]
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    texture = cv2.resize(rng.normal(0, 8, (8, 12)).astype(np.float32), size)
    frame += texture[..., None]
    diameter = round(rng.uniform(0.07, 0.22) * height)
    rx = max(5, diameter // 2)
    ry = round(rx * rng.uniform(1, 1.27))
    x, y = center or (int(rng.integers(ry + 6, width - ry - 6)),
                      int(rng.integers(ry + 6, height - ry - 6)))
    label = None
    if kind in ("balloon", "red_balloon"):
        u, v = np.meshgrid(np.arange(-rx, rx + 1) / rx, np.arange(-ry, ry + 1) / ry)
        mask = u * u + v * v <= 1
        light = np.sqrt(np.maximum(0, 1 - u * u - v * v))
        hotspot = np.exp(-((u + .35)**2 + (v + .42)**2) / (.07 if glare else .03))
        value = 112 + 107 * light + (130 if glare else 37) * hotspot
        saturation = 202 - (180 if glare else 70) * hotspot
        hue = int(rng.integers(25, 34)) if kind == "balloon" else 1
        patch = np.stack([np.full_like(value, hue), saturation, value.clip(0, 255)], axis=-1)
        roi = frame[y-ry:y+ry+1, x-rx:x+rx+1]
        roi[mask] = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_HSV2BGR)[mask]
        if kind == "balloon":
            label = (x-rx, y-ry, x+rx+1, y+ry+1)
    elif kind == "disk":
        cv2.circle(frame, (x, y), rx, (35, 205, 224), -1)
    elif kind == "rectangle":
        cv2.rectangle(frame, (x-rx, y-ry), (x+rx, y+ry), (28, 202, 221), -1)
    elif kind == "ring":
        cv2.circle(frame, (x, y), rx, (21, 202, 220), max(2, rx//4))
    frame = frame * exposure + rng.normal(0, 1.6, (height, width, 1)).astype(np.float32)
    frame = cv2.GaussianBlur(frame.clip(0, 255).astype(np.uint8), (3, 3), .55)
    return frame, label


KINDS = ("balloon", "balloon", "balloon", "disk", "rectangle", "ring", "red_balloon", "empty")
