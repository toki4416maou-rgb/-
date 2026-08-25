# Akane / Lāma-X Complete Prototype v1.1

Compact research prototype that connects the validated numerical Lāma-X core to:

- text / Japanese toy-command Codec
- source-code structural Codec
- image adapter
- WAV audio adapter
- video frame adapter
- live web HTML ingestion
- Common Crawl index + WARC range retrieval
- Canonical IR
- residual memory / repeated-structure X-candidate marking

## Core research boundary

The Lāma-X executable reasoning core is currently validated only on its finite Toy World primitive set.
Arbitrary web/image/audio/video input is normalized and stored as evidence, but **v1.1 does not pretend that it has learned executable new cross-modal primitives automatically**.

That is the next empirical research problem.

## Run

```bash
python trainer.py
python app.py toy
python app.py text "青いcircleのvalueを合計"
python app.py code example.py
python app.py image photo.png
python app.py audio sample.wav
python app.py video clip.mp4
python app.py url https://example.com
python app.py cc example.com --limit 5
```

Optional richer multimedia support uses Pillow, NumPy, and OpenCV. The adapters degrade to metadata-only mode where possible.
