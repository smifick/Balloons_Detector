"""Annotate a video file with detections and a single persistent visual track."""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from vision.detectors import OpenCVDetector, YOLODetector, overlap


def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2) / 2, (y1+y2) / 2])


class VisualTrack:
    """Keep an identity across short gaps. Return measured boxes only."""

    def __init__(self, max_gap=6):
        self.max_gap = max_gap
        self.box = None
        self.velocity = np.zeros(2)
        self.misses = 0

    def update(self, boxes):
        if self.box is None:
            if len(boxes) == 1:
                self.box = boxes[0]
            return self.box

        position = center(self.box)
        span = max(self.box[2] - self.box[0], self.box[3] - self.box[1])
        expected = position + (self.misses + 1) * self.velocity
        matches = []
        for box in boxes:
            distance = np.linalg.norm(center(box) - expected)
            shifted = np.asarray(self.box) + np.tile(center(box) - position, 2)
            if distance <= max(8, span * .75) and overlap(shifted, box) >= .55:
                matches.append((distance, box))
        matches.sort(key=lambda match: match[0])
        if matches and (len(matches) == 1 or matches[1][0] - matches[0][0] > max(3, .2 * span)):
            self.box = matches[0][1]
            measured_velocity = (center(self.box) - position) / (self.misses + 1)
            self.velocity = .65 * measured_velocity + .35 * self.velocity
            self.misses = 0
            return self.box

        self.misses += 1
        if self.misses > self.max_gap:
            self.box = None
            self.velocity.fill(0)
            self.misses = 0
        return None


def annotate(source, output, detector):
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output filename must end in .mp4")
    if output.exists():
        raise FileExistsError(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Unreadable video or frame rate: {source}")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Cannot write MP4 video: {output}")
        track = VisualTrack(max_gap=max(1, round(fps * .3)))
        while ok:
            boxes = detector.detect(frame)
            selected = track.update(boxes)
            for box in boxes:
                x1, y1, x2, y2 = (round(coord) for coord in box)
                color = (40, 230, 90) if selected is box else (30, 210, 240)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            if selected is not None:
                cv2.circle(frame, tuple(np.round(center(selected)).astype(int)), 3, (40, 230, 90), -1)
            writer.write(frame)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--detector", choices=["opencv", "yolo"], default="opencv")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    detector = OpenCVDetector() if args.detector == "opencv" else YOLODetector()
    annotate(args.source, args.output, detector)


if __name__ == "__main__":
    main()
