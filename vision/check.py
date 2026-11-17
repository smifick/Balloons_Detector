"""Evaluate both weight files on an independent synthetic test seed."""

import argparse

import cv2
import numpy as np

from vision.detectors import OpenCVDetector, YOLODetector, overlap
from vision.synthetic import PROFILES, RESOLUTIONS, scene


def evaluate(detector):
    rng = np.random.default_rng(1971)
    accepted = True
    for size in RESOLUTIONS:
        for profile in PROFILES:
            hits, positives, false_positives = 0, 0, 0
            for kind in ("balloon", "rectangle", "disk", "ring", "red_balloon", "empty"):
                for _ in range(6):
                    frame, truth = scene(rng, size, profile, kind)
                    boxes = detector.detect(frame)
                    match = truth is not None and any(overlap(box, truth) >= .5 for box in boxes)
                    hits += int(match)
                    positives += int(truth is not None)
                    false_positives += len(boxes) - int(match)
            print(f"{size[0]}x{size[1]:<3} {profile:<13} matched {hits}/{positives}, FP {false_positives}")
            accepted &= hits == positives and false_positives == 0
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["opencv", "yolo", "both"], default="both")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    results = []
    for name, factory in (("opencv", OpenCVDetector), ("yolo", YOLODetector)):
        if args.detector in (name, "both"):
            print(name, flush=True)
            results.append(evaluate(factory()))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
