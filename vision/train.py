import argparse
import json
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np

from vision.detectors import MODELS, candidates, overlap
from vision.synthetic import KINDS, PROFILES, RESOLUTIONS, scene


def train_svm():
    rng = np.random.default_rng(137)
    rows, labels = [], []
    for i in range(864):
        size = RESOLUTIONS[i % len(RESOLUTIONS)]
        profile = tuple(PROFILES)[i // len(RESOLUTIONS) % len(PROFILES)]
        frame, truth = scene(rng, size, profile, KINDS[i // 12 % len(KINDS)])
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 64 + i % 28])
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        for sample in (frame, compressed):
            for box, features in candidates(sample):
                rows.append(features)
                labels.append(int(truth is not None and overlap(box, truth) >= .5))
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(48)
    svm.setGamma(16)
    svm.train(np.asarray(rows, np.float32), cv2.ml.ROW_SAMPLE, np.asarray(labels, np.int32))
    svm.save(str(MODELS / "balloon-svm.xml"))


def train_yolo(root, epochs, initial_weights):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    root = root.resolve()
    for split, count, seed in (("train", 384, 178), ("val", 96, 179)):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        for i in range(count):
            width, height = size = RESOLUTIONS[i % 3]
            profile = tuple(PROFILES)[i // 3 % 4]
            frame, truth = scene(rng, size, profile, KINDS[i // 12 % len(KINDS)])
            if not cv2.imwrite(str(images / f"{i:04}.jpg"), frame):
                raise OSError(f"Could not save training frame {i}")
            annotation = ""
            if truth is not None:
                x1, y1, x2, y2 = truth
                annotation = f"0 {(x1+x2)/2/width} {(y1+y2)/2/height} {(x2-x1)/width} {(y2-y1)/height}\n"
            (labels / f"{i:04}.txt").write_text(annotation)
    config = root / "data.yaml"
    config.write_text(f"path: {json.dumps(root.as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\nnames: [yellow_balloon]\n")
    model = YOLO(initial_weights)
    model.train(data=str(config), imgsz=320, epochs=epochs, batch=16, workers=0,
                device="cpu", project=str(root / "runs"), seed=137, plots=False,
                mosaic=0, mixup=0, hsv_h=0, hsv_s=.12, hsv_v=.12, scale=.2,
                translate=.08, degrees=5, amp=False)
    copyfile(model.trainer.best, MODELS / "balloon-yolo.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["svm", "yolo"])
    parser.add_argument("--work-dir", type=Path, default=Path("runs/balloon-data"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--initial-weights", default="yolov8n.pt")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    MODELS.mkdir(exist_ok=True)
    if args.model == "svm":
        train_svm()
    else:
        train_yolo(args.work_dir, args.epochs, args.initial_weights)
