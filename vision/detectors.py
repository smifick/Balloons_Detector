from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).with_name("models")


def candidates(frame):
    """Generate candidate regions and describe their shape and light distribution."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    foreground = cv2.inRange(hsv, (17, 62, 40), (43, 255, 255))
    yellow_hue = cv2.inRange(hsv[..., 0], 17, 43)
    if cv2.countNonZero(yellow_hue) > foreground.size / 2:
        # When yellow fills the image, separate the more saturated foreground.
        saturation_floor = max(62, int(np.median(hsv[..., 1])) + 22)
        foreground = cv2.inRange(hsv, (17, saturation_floor, 40), (43, 255, 255))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 35:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 7 or x == 0 or y == 0 or x+w >= frame.shape[1] or y+h >= frame.shape[0]:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x, -y))
        hue, saturation, light = hsv[y:y+h, x:x+w].transpose(2, 0, 1)
        yy, xx = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
        radius2 = xx**2 + yy**2
        interior = mask > 0
        core = interior & (radius2 < .24)
        edge = interior & (radius2 > .55)
        if not core.any() or not edge.any():
            continue
        values = light[interior].astype(np.float32)
        outline = cv2.approxPolyDP(contour, max(.75, .025 * min(w, h)), True)
        perimeter = cv2.arcLength(outline, True)
        if perimeter == 0:
            continue
        features = [
            area / (w*h), min(w, h) / max(w, h),
            4*np.pi*area / perimeter**2,
            foreground[y:y+h, x:x+w][interior].mean() / 255,
            hue[interior].mean() / 180, saturation[interior].mean() / 255,
            values.std() / max(1, values.mean()),
            (light[core].mean() - light[edge].mean()) / max(1, values.mean()),
        ]
        yield (x, y, x+w, y+h), features


def overlap(a, b):
    common = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return common / max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - common)


class OpenCVDetector:
    def __init__(self, weights=MODELS / "balloon-svm.xml"):
        self.model = cv2.ml.SVM_load(str(weights))

    def detect(self, frame):
        regions = list(candidates(frame))
        if not regions:
            return []
        _, labels = self.model.predict(np.asarray([features for _, features in regions], np.float32))
        return [box for (box, _), label in zip(regions, labels.ravel()) if label == 1]


class YOLODetector:
    def __init__(self, weights=MODELS / "balloon-yolo.pt", confidence=.25):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(2)
        self.model = YOLO(str(weights))
        self.confidence = confidence

    def detect(self, frame):
        result = self.model.predict(frame, imgsz=320, device="cpu", conf=self.confidence,
                                    classes=[0], verbose=False)[0]
        return [tuple(row) for row in result.boxes.xyxy.cpu().tolist()]
