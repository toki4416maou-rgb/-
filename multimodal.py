"""
Akane multimodal adapters v1.1
==============================

Image / audio / video are reduced into lightweight measurable features and
normalized to CanonicalIR.  These are sensory adapters, not claims of semantic
vision/audio understanding.

Optional dependencies:
- Pillow + numpy: richer image features
- OpenCV + numpy: video frame sampling
All adapters degrade gracefully when optional packages are unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import math
import wave

from codec import CanonicalIR

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2
except Exception:
    cv2 = None


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ImageAdapter:
    @staticmethod
    def to_ir(path: str | Path) -> CanonicalIR:
        p = Path(path)
        features: Dict[str, Any] = {
            "filename": p.name,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }

        confidence = 0.45
        if Image is not None and np is not None:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                features.update({
                    "width": rgb.width,
                    "height": rgb.height,
                    "format": im.format or p.suffix.lstrip("."),
                })

                sample = rgb.copy()
                sample.thumbnail((128, 128))
                arr = np.asarray(sample, dtype=np.float32)
                mean = arr.mean(axis=(0, 1))
                std = arr.std(axis=(0, 1))
                lum = arr.mean(axis=2)

                gx = np.abs(np.diff(lum, axis=1)).mean() if lum.shape[1] > 1 else 0.0
                gy = np.abs(np.diff(lum, axis=0)).mean() if lum.shape[0] > 1 else 0.0

                hist, _ = np.histogram(lum, bins=32, range=(0, 255), density=False)
                prob = hist / max(1, hist.sum())
                entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())

                features.update({
                    "mean_rgb": [float(x) for x in mean],
                    "std_rgb": [float(x) for x in std],
                    "mean_brightness": float(lum.mean()),
                    "edge_energy": float((gx + gy) / 2.0),
                    "luminance_entropy": entropy,
                })
                confidence = 1.0

        return CanonicalIR(
            modality="image",
            intent="perceive",
            content="",
            features=features,
            payload={"path": str(p)},
            source=str(p),
            confidence=confidence,
        )


class AudioAdapter:
    @staticmethod
    def to_ir(path: str | Path) -> CanonicalIR:
        p = Path(path)
        features: Dict[str, Any] = {
            "filename": p.name,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }
        confidence = 0.4

        try:
            with wave.open(str(p), "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                width = wf.getsampwidth()
                frames = wf.getnframes()
                raw = wf.readframes(frames)

            features.update({
                "channels": channels,
                "sample_rate": rate,
                "sample_width": width,
                "frame_count": frames,
                "duration_sec": frames / rate if rate else 0.0,
            })

            if np is not None and raw:
                dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
                if dtype is not None:
                    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                    if width == 1:
                        x -= 128.0
                    peak = float(np.max(np.abs(x))) if len(x) else 0.0
                    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
                    zcr = float(np.mean((x[:-1] * x[1:]) < 0)) if len(x) > 1 else 0.0

                    features.update({
                        "peak": peak,
                        "rms": rms,
                        "zero_crossing_rate": zcr,
                    })
            confidence = 1.0
        except (wave.Error, EOFError):
            features["decoder"] = "metadata-only; WAV decoder could not read this format"

        return CanonicalIR(
            modality="audio",
            intent="perceive",
            features=features,
            payload={"path": str(p)},
            source=str(p),
            confidence=confidence,
        )


class VideoAdapter:
    @staticmethod
    def to_ir(path: str | Path, sample_frames: int = 12) -> CanonicalIR:
        p = Path(path)
        features: Dict[str, Any] = {
            "filename": p.name,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }
        confidence = 0.35

        if cv2 is not None and np is not None:
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                duration = frame_count / fps if fps > 0 else 0.0

                features.update({
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration_sec": duration,
                })

                samples = []
                motions = []
                prev = None

                if frame_count > 0:
                    indices = np.linspace(0, max(0, frame_count - 1), num=min(sample_frames, frame_count), dtype=int)
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ok, frame = cap.read()
                        if not ok:
                            continue
                        small = cv2.resize(frame, (64, 64))
                        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        samples.append(float(gray.mean()))
                        if prev is not None:
                            motions.append(float(np.mean(np.abs(gray - prev))))
                        prev = gray

                features.update({
                    "sampled_frames": len(samples),
                    "mean_brightness": float(np.mean(samples)) if samples else None,
                    "mean_frame_motion": float(np.mean(motions)) if motions else 0.0,
                })
                confidence = 1.0
            cap.release()

        return CanonicalIR(
            modality="video",
            intent="perceive",
            features=features,
            payload={"path": str(p)},
            source=str(p),
            confidence=confidence,
        )
