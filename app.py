"""
Akane Complete Prototype v1.1
=============================

Unified CLI for:
- language
- code
- image
- audio
- video
- live web
- Common Crawl
- Toy World Lāma-X inference

Examples
--------
python app.py text "青いcircleのvalueを合計"
python app.py code example.py
python app.py image photo.png
python app.py audio sample.wav
python app.py video clip.mp4
python app.py url https://example.com
python app.py cc example.com
python app.py toy
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse
import json
import sys

from codec import CanonicalIR, TextCodec, CodeCodec, render_ir
from multimodal import ImageAdapter, AudioAdapter, VideoAdapter
from web_ingest import CommonCrawlClient, KnowledgeIngestor, fetch_url
from learning import LamaSystem
from memory import XStore
from trainer import ToyWorldGenerator


MEMORY_PATH = Path(__file__).with_name("akane_memory.json")


def make_system() -> LamaSystem:
    store = XStore.load(MEMORY_PATH) if MEMORY_PATH.exists() else XStore()
    return LamaSystem(store=store, seed=42)


def bootstrap(system: LamaSystem, episodes: int = 500) -> None:
    gen = ToyWorldGenerator(42)
    for _ in range(episodes):
        w, q, e = gen.train_task()
        system.learn_episode(w, q, e)


def print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def handle_ir(system: LamaSystem, ir: CanonicalIR) -> None:
    ing = KnowledgeIngestor(system.store)
    result = ing.ingest(ir)
    print_json({
        "canonical_ir": ir.to_dict(),
        "memory_result": result,
    })


def cmd_text(system: LamaSystem, text: str) -> None:
    ir = TextCodec.to_ir(text)
    toy_query = ir.payload.get("toy_query")

    out: Dict[str, Any] = {"canonical_ir": ir.to_dict()}
    if toy_query:
        # Demonstration world for the natural-language bridge.
        world = {
            "objects": [
                {"id": "a", "shape": "circle", "color": "red", "value": 4},
                {"id": "b", "shape": "square", "color": "blue", "value": 7},
                {"id": "c", "shape": "circle", "color": "blue", "value": 2},
            ]
        }
        bootstrap(system, 500)
        r = system.infer(world, toy_query)
        out["toy_world_demo"] = {
            "world": world,
            "query": toy_query,
            "result": r.value,
            "lama_x": str(r.lama_x),
        }
    print_json(out)


def cmd_code(system: LamaSystem, path: str) -> None:
    handle_ir(system, CodeCodec.from_file(path))


def cmd_image(system: LamaSystem, path: str) -> None:
    handle_ir(system, ImageAdapter.to_ir(path))


def cmd_audio(system: LamaSystem, path: str) -> None:
    handle_ir(system, AudioAdapter.to_ir(path))


def cmd_video(system: LamaSystem, path: str) -> None:
    handle_ir(system, VideoAdapter.to_ir(path))


def cmd_url(system: LamaSystem, url: str) -> None:
    ir = fetch_url(url)
    handle_ir(system, ir)


def cmd_cc(system: LamaSystem, pattern: str, limit: int) -> None:
    client = CommonCrawlClient()
    records = client.search(pattern, limit=limit, match_type="domain")
    out = {
        "index_id": client.resolve_index_id(),
        "records_found": len(records),
        "records": [r.__dict__ for r in records],
    }

    # Fetch only the first result by default; this keeps the CLI conservative.
    if records:
        ir = client.fetch_record_ir(records[0])
        ing = KnowledgeIngestor(system.store)
        out["first_record_ir"] = ir.to_dict()
        out["memory_result"] = ing.ingest(ir)

    print_json(out)


def cmd_toy(system: LamaSystem) -> None:
    bootstrap(system, 900)
    gen = ToyWorldGenerator(123)
    world, query, expected = gen.heldout_task()
    r = system.infer(world, query)
    print_json({
        "world": world,
        "query": query,
        "expected": expected,
        "result": r.value,
        "correct": r.value == expected,
        "lama_x": str(r.lama_x),
        "primitive_codes": [str(x) for x in r.primitive_codes],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description="Akane Complete Prototype v1.1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("text")
    p.add_argument("text")

    p = sub.add_parser("code")
    p.add_argument("path")

    p = sub.add_parser("image")
    p.add_argument("path")

    p = sub.add_parser("audio")
    p.add_argument("path")

    p = sub.add_parser("video")
    p.add_argument("path")

    p = sub.add_parser("url")
    p.add_argument("url")

    p = sub.add_parser("cc")
    p.add_argument("pattern")
    p.add_argument("--limit", type=int, default=5)

    sub.add_parser("toy")

    args = parser.parse_args(argv)
    system = make_system()

    if args.cmd == "text":
        cmd_text(system, args.text)
    elif args.cmd == "code":
        cmd_code(system, args.path)
    elif args.cmd == "image":
        cmd_image(system, args.path)
    elif args.cmd == "audio":
        cmd_audio(system, args.path)
    elif args.cmd == "video":
        cmd_video(system, args.path)
    elif args.cmd == "url":
        cmd_url(system, args.url)
    elif args.cmd == "cc":
        cmd_cc(system, args.pattern, args.limit)
    elif args.cmd == "toy":
        cmd_toy(system)

    system.store.save(MEMORY_PATH)


if __name__ == "__main__":
    main()
