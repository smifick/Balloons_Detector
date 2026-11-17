import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision.annotate import VisualTrack, annotate
from vision.detectors import OpenCVDetector, YOLODetector
from vision.synthetic import scene


def box(x, y):
    return x - 8, y - 8, x + 8, y + 8


class VisualTrackTests(unittest.TestCase):
    def test_preserves_track_across_gap_with_distractor(self):
        track = VisualTrack(max_gap=4)
        self.assertEqual(track.update([box(40, 60)]), box(40, 60))
        self.assertEqual(track.update([box(47, 60), box(90, 60)]), box(47, 60))
        self.assertIsNone(track.update([box(130, 60)]))
        self.assertEqual(track.update([box(130, 60), box(58, 60)]), box(58, 60))

    def test_ambiguity_suppresses_measurement_and_identity_expires(self):
        track = VisualTrack(max_gap=1)
        self.assertIsNone(track.update([box(40, 60), box(50, 60)]))
        self.assertEqual(track.update([box(40, 60)]), box(40, 60))
        self.assertIsNone(track.update([box(38, 60), box(42, 60)]))
        self.assertIsNone(track.update([box(100, 60)]))
        self.assertEqual(track.update([box(100, 60)]), box(100, 60))

    def test_opencv_video_round_trip(self):
        self.check_video(OpenCVDetector())

    def test_yolo_video_round_trip(self):
        self.check_video(YOLODetector())

    def check_video(self, detector):
        with tempfile.TemporaryDirectory() as temp:
            source, output = (Path(temp) / name for name in ("input.mp4", "output.mp4"))
            size = (320, 240)
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, size)
            self.assertTrue(writer.isOpened())
            try:
                for x in (100, 104, 108, 112, 116):
                    rng = np.random.default_rng(202)
                    frame, _ = scene(rng, size, "glare", "balloon", (x, 120))
                    writer.write(frame)
            finally:
                writer.release()
            annotate(source, output, detector)
            reader = cv2.VideoCapture(str(output))
            try:
                for x in (100, 104, 108, 112, 116):
                    ok, frame = reader.read()
                    self.assertTrue(ok)
                    region = frame[118:123, x-2:x+3].astype(int)
                    blue, green, red = region.transpose(2, 0, 1)
                    self.assertTrue(((green - red > 70) & (green - blue > 70)).any())
                self.assertFalse(reader.read()[0])
            finally:
                reader.release()


if __name__ == "__main__":
    unittest.main()
