"""Score smoke ROIs and preview or export the result of an offline video test.

Run with: python smoke_laplacian.py INPUT.mp4 --output RESULT.mp4 --show
Dependencies: OpenCV with HighGUI support for --show, and NumPy.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import numpy as np


Box = tuple[int, int, int, int]  # Half-open coordinates in the input frame.
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
CONTOUR_KERNELS = (
    (np.ones((5, 5), np.uint8), 14, 0.48),
    (np.ones((7, 7), np.uint8), 15, 0.58),
)
HORIZONTAL_CLOSE = np.ones((1, 7), np.uint8)
HORIZONTAL_OPEN = np.ones((1, 20), np.uint8)


@dataclass(frozen=True)
class SmokeScore:
    bbox: Box
    roi: Box
    variance: float | None
    passes: bool


@dataclass(frozen=True)
class FrameResult:
    boxes: tuple[SmokeScore, ...]
    consecutive: int
    longest: int
    video_passes: bool

    @property
    def passes(self) -> bool:
        return any(box.passes for box in self.boxes)


@dataclass(frozen=True)
class VideoResult:
    frames_processed: int
    fps: float
    frames_with_bbox: int
    frames_passing: int
    longest_streak: int
    decision: str | None
    complete: bool
    output: str | None


def _orange_mask(frame: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(frame.astype(np.int16))
    return (
        (red >= 215) & (green >= 116) & (green <= 209) & (blue <= 78)
        & (red - green >= 40) & (green - blue >= 67)
    )


def _label_present(mask: np.ndarray, x: int, y: int) -> bool:
    strip = mask[max(0, y - 13):max(1, y - 2), x:x + 52]
    return np.count_nonzero(strip) >= 29


def _near(first: Box, second: Box, tolerance: int) -> bool:
    return max(abs(a - b) for a, b in zip(first, second)) <= tolerance


def _edge_support(mask: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = box
    area = mask[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3]
    return min(
        area[:7].any(0).mean(), area[-7:].any(0).mean(),
        area[:, :7].any(1).mean(), area[:, -7:].any(1).mean(),
    )


def _contour_boxes(mask: np.ndarray, binary: np.ndarray) -> list[Box]:
    boxes: list[Box] = []
    for kernel, min_side, min_support in CONTOUR_KERNELS:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
            if parent < 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            box = (x, y, x + width, y + height)
            if min(width, height) < min_side or _edge_support(mask, box) < min_support:
                continue
            if not any(_near(box, other, 4) for other in boxes):
                boxes.append(box)
    return sorted(box for box in boxes if _label_present(mask, box[0], box[1]))


def _interrupted_boxes(
    mask: np.ndarray, binary: np.ndarray, contour_boxes: list[Box],
) -> list[Box]:
    # A crossing overlay can break a contour. Pair horizontal edges only if
    # the original orange pixels also support both vertical edges.
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, HORIZONTAL_CLOSE)
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, HORIZONTAL_OPEN)
    candidates = []
    for y in np.flatnonzero(lines.any(1)):
        xs = np.flatnonzero(lines[y])
        for run in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
            if len(run) >= 20:
                candidates.append((int(run[0]) - 1, int(y), int(run[-1]) + 1))

    recovered: list[Box] = []
    for left, top, right in candidates:
        if left < 0 or right >= mask.shape[1] or not _label_present(mask, left, top + 1):
            continue
        for other_left, bottom, other_right in candidates:
            if not 20 <= bottom - top <= 690:
                continue
            if max(abs(other_left - left), abs(other_right - right)) > 3:
                continue
            box = (left, top + 1, right, bottom)
            if any(_near(box, other, 6) for other in contour_boxes + recovered):
                continue
            between = mask[top + 1:bottom]
            supports = (
                between[:, max(0, left - 2):left + 3].any(1).mean(),
                between[:, right - 2:right + 3].any(1).mean(),
                mask[max(0, top - 1):top + 2, left:right].any(0).mean(),
                mask[bottom - 1:bottom + 2, left:right].any(0).mean(),
            )
            if min(supports) >= 0.72:
                recovered.append(box)
    return recovered


def detect_smoke_boxes(frame: np.ndarray) -> list[Box]:
    """Estimate the smoke bbox coordinates from rendered amber YOLOX overlays."""
    mask = _orange_mask(frame)
    binary = mask.astype(np.uint8)
    contours = _contour_boxes(mask, binary)
    return sorted(contours + _interrupted_boxes(mask, binary, contours))


def _overlapping_graphics(gray: np.ndarray, boxes: list[Box]) -> np.ndarray:
    excluded = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(excluded, (x0, y0), (x1 - 1, y1 - 1), 1, 5)
        cv2.rectangle(
            excluded, (x0, max(0, y0 - 15)), (min(gray.shape[1] - 1, x0 + 62), y0 + 2), 1, -1,
        )
    # Exclude adjacent Laplacian centers whose four-neighbor kernel touches the overlay.
    return cv2.dilate(excluded, CROSS)


class SmokeLaplacianFilter:
    """Maintain consecutive passing frames for one video or stream."""

    def __init__(self, threshold: float = 300.0, min_consecutive: int = 5) -> None:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be at least 1")
        self.threshold = threshold
        self.min_consecutive = min_consecutive
        self.consecutive = 0
        self.longest = 0

    def process_frame(self, frame: np.ndarray, smoke_boxes: list[Box] | None = None) -> FrameResult:
        """Score a uint8 BGR frame. Supply smoke_boxes for a clean, unrendered frame."""
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
            raise ValueError("frame must be a non-empty uint8 BGR image")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        has_overlay = smoke_boxes is None
        boxes = detect_smoke_boxes(frame) if has_overlay else list(smoke_boxes)
        margin = (5 if len(boxes) > 1 else 4) if has_overlay else 0
        excluded = _overlapping_graphics(gray, boxes) if has_overlay and len(boxes) > 1 else None

        results = []
        for box in boxes:
            x0, y0, x1, y1 = box
            left, top = max(0, x0 + margin), max(0, y0 + margin)
            right, bottom = min(width, x1 - margin), min(height, y1 - margin)
            score = None
            if right - left >= 2 and bottom - top >= 2:
                crop = np.s_[top:bottom, left:right]
                lap = cv2.Laplacian(gray[crop], cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
                if excluded is not None:
                    lap = lap[excluded[crop] == 0]
                if lap.size:
                    score = float(lap.var())
            results.append(SmokeScore(box, (left, top, right, bottom), score,
                                      score is not None and score <= self.threshold))

        self.consecutive = self.consecutive + 1 if any(box.passes for box in results) else 0
        self.longest = max(self.longest, self.consecutive)
        return FrameResult(tuple(results), self.consecutive, self.longest,
                           self.longest >= self.min_consecutive)


def _annotate(
    frame: np.ndarray, result: FrameResult, index: int, fps: float, required: int, final: bool,
) -> np.ndarray:
    image = frame.copy()
    for box in result.boxes:
        color = (85, 235, 80) if box.passes else (65, 75, 255)
        x0, y0, x1, y1 = box.roi
        if x0 >= x1 or y0 >= y1:
            continue
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        label = "unscored" if box.variance is None else f"{box.variance:.1f} {'KEEP' if box.passes else 'REJECT'}"
        cv2.putText(image, label, (x0, min(y0 + 20, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (x0, min(y0 + 20, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)

    frame_state = "KEEP" if result.passes else ("REJECT" if result.boxes else "NO BBOX")
    clip_state = "KEEP" if result.video_passes else ("REJECT" if final else "PENDING")
    hud = (f"Frame {index + 1} | {index / fps:.2f}s | {frame_state} | "
           f"Run {result.consecutive}/{required} | Max {result.longest} | Clip {clip_state}")
    height = image.shape[0]
    cv2.rectangle(image, (0, max(0, height - 30)), (image.shape[1] - 1, height - 1), (20, 20, 20), -1)
    cv2.putText(image, hud, (8, max(15, height - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def process_video(
    source: str | Path, output: str | Path | None = None, *, show: bool = False,
    threshold: float = 300.0, min_consecutive: int = 5,
) -> VideoResult:
    """Process every source frame and optionally write an annotated MP4 or preview it."""
    source_path = Path(source)
    output_path = Path(output) if output is not None else None
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path is not None:
        if output_path.resolve() == source_path.resolve():
            raise ValueError("Input and output must be different files")
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if show and hasattr(cv2, "currentUIFramework") and not cv2.currentUIFramework():
        raise RuntimeError("--show needs an OpenCV build with a GUI backend")

    smoke_filter = SmokeLaplacianFilter(threshold, min_consecutive)
    capture = cv2.VideoCapture(str(source_path))
    writer = None
    temporary = None
    window = "Smoke Laplacian | Q or Esc to stop"
    window_open = False
    complete = True
    count = detected = passing = 0
    try:
        try:
            if not capture.isOpened():
                raise OSError(f"Cannot open video: {source_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise OSError(f"Invalid FPS for {source_path}")
            expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if show:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                window_open = True
            ok, frame = capture.read()
            while ok:
                start = time.perf_counter()
                next_ok, next_frame = capture.read()
                result = smoke_filter.process_frame(frame)
                detected += bool(result.boxes)
                passing += result.passes
                if show or output_path is not None:
                    image = _annotate(frame, result, count, fps, min_consecutive, final=not next_ok)
                if output_path is not None:
                    if writer is None:
                        temporary = output_path.with_name(f".{output_path.stem}-{uuid4().hex}.mp4")
                        size = (image.shape[1], image.shape[0])
                        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                        if not writer.isOpened():
                            raise OSError(f"Cannot encode MP4: {output_path}")
                    writer.write(image)
                count += 1
                if show:
                    cv2.imshow(window, image)
                    delay = max(1, round(1000 * (1 / fps - (time.perf_counter() - start))))
                    key = cv2.waitKey(delay) & 0xFF
                    try:
                        visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
                    except cv2.error:
                        visible = False
                    if next_ok and (key in (27, ord("q"), ord("Q")) or not visible):
                        complete = False
                        break
                ok, frame = next_ok, next_frame
            if count == 0:
                raise OSError(f"Video has no decodable frames: {source_path}")
            if complete and math.isfinite(expected) and expected > 0 and count < round(expected):
                raise OSError(f"Video ended after {count} of {round(expected)} expected frames")
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if window_open:
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    pass  # Closing via the title bar may already have destroyed it.
        if temporary is not None:
            verification = cv2.VideoCapture(str(temporary))
            try:
                if not verification.isOpened() or round(verification.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                    raise OSError("Exported MP4 could not be verified")
            finally:
                verification.release()
            assert output_path is not None
            temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    decision = ("KEEP" if smoke_filter.longest >= min_consecutive else "REJECT") if complete else None
    return VideoResult(count, fps, detected, passing, smoke_filter.longest,
                       decision, complete, str(output_path) if output_path is not None else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input video with smoke overlays")
    parser.add_argument("--output", type=Path, help="Write annotated frames to an MP4 file")
    parser.add_argument("--show", action="store_true", help="Preview via OpenCV. Press Q or Esc to stop.")
    parser.add_argument("--threshold", type=float, default=300.0, help="Max Laplacian variance (default 300)")
    parser.add_argument("--min-consecutive", type=int, default=5, help="Required passing frames (default 5)")
    args = parser.parse_args()
    try:
        result = process_video(args.source, args.output, show=args.show, threshold=args.threshold,
                               min_consecutive=args.min_consecutive)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(vars(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

