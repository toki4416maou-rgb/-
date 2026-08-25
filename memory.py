"""
Akane / Lāma-X v1.0 persistent memory.

Stores:
- Primitive X statistics
- Discovered recipe X statistics
- feature -> Primitive routing weights
- feature -> Recipe routing weights
- successful trace recurrence
- explicit residual facts

Does NOT store generated NeuroSpec as model state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple
import json
import time


@dataclass
class XRecord:
    lama_x: int
    kind: str = "recipe"     # primitive | recipe
    name: str = ""
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    reward_mean: float = 0.0
    confidence: float = 0.5
    created_at: float = 0.0
    route_key: str = ""

    def update(self, success: bool, reward: float) -> None:
        self.use_count += 1
        self.success_count += int(success)
        self.failure_count += int(not success)
        self.reward_mean += (reward - self.reward_mean) / max(1, self.use_count)
        self.confidence = (self.success_count + 1) / (self.use_count + 2)


class XStore:
    def __init__(self) -> None:
        self.records: Dict[int, XRecord] = {}
        self.primitive_weights: Dict[str, Dict[int, float]] = {}
        self.recipe_weights: Dict[str, Dict[int, float]] = {}
        self.trace_success: Dict[str, int] = {}
        self.recipe_index: Dict[str, set[int]] = {}
        self.residual: list[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(
        self,
        lama_x: int,
        kind: str,
        name: str = "",
        route_key: str = "",
    ) -> XRecord:
        lama_x = int(lama_x)
        if lama_x not in self.records:
            self.records[lama_x] = XRecord(
                lama_x=lama_x,
                kind=kind,
                name=name,
                created_at=time.time(),
                route_key=route_key,
            )
        rec = self.records[lama_x]
        if route_key and not rec.route_key:
            rec.route_key = route_key
        if rec.kind == "recipe" and rec.route_key:
            self.recipe_index.setdefault(rec.route_key, set()).add(lama_x)
        return rec

    def recipes_for(self, route_key: str) -> Tuple[int, ...]:
        return tuple(self.recipe_index.get(route_key, ()))

    def update(self, lama_x: int, success: bool, reward: float) -> None:
        rec = self.records.get(int(lama_x))
        if rec is not None:
            rec.update(success, reward)

    def primitive_weight(self, feature: str, code: int) -> float:
        return self.primitive_weights.get(feature, {}).get(int(code), 0.0)

    def add_primitive_weight(self, feature: str, code: int, delta: float) -> None:
        b = self.primitive_weights.setdefault(feature, {})
        code = int(code)
        b[code] = b.get(code, 0.0) + float(delta)

    def recipe_weight(self, feature: str, lama_x: int) -> float:
        return self.recipe_weights.get(feature, {}).get(int(lama_x), 0.0)

    def add_recipe_weight(self, feature: str, lama_x: int, delta: float) -> None:
        b = self.recipe_weights.setdefault(feature, {})
        lama_x = int(lama_x)
        b[lama_x] = b.get(lama_x, 0.0) + float(delta)

    def note_trace(self, primitive_codes: Sequence[int]) -> int:
        key = ",".join(str(int(x)) for x in primitive_codes)
        self.trace_success[key] = self.trace_success.get(key, 0) + 1
        return self.trace_success[key]

    def remember_residual(self, item: Mapping[str, Any], max_items: int = 1000) -> None:
        self.residual.append(dict(item))
        if len(self.residual) > max_items:
            del self.residual[:len(self.residual) - max_items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [asdict(r) for r in self.records.values()],
            "primitive_weights": {
                f: {str(k): v for k, v in b.items()}
                for f, b in self.primitive_weights.items()
            },
            "recipe_weights": {
                f: {str(k): v for k, v in b.items()}
                for f, b in self.recipe_weights.items()
            },
            "trace_success": self.trace_success,
            "recipe_index": {
                k: [str(x) for x in sorted(v)]
                for k, v in self.recipe_index.items()
            },
            "residual": self.residual,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "XStore":
        s = cls()
        for raw in data.get("records", []):
            r = XRecord(**raw)
            s.records[int(r.lama_x)] = r
        s.primitive_weights = {
            f: {int(k): float(v) for k, v in b.items()}
            for f, b in data.get("primitive_weights", {}).items()
        }
        s.recipe_weights = {
            f: {int(k): float(v) for k, v in b.items()}
            for f, b in data.get("recipe_weights", {}).items()
        }
        s.trace_success = {
            str(k): int(v) for k, v in data.get("trace_success", {}).items()
        }
        s.recipe_index = {
            str(k): {int(x) for x in v}
            for k, v in data.get("recipe_index", {}).items()
        }
        # Rebuild if loading a file made before the explicit index existed.
        if not s.recipe_index:
            for x, rec in s.records.items():
                if rec.kind == "recipe" and rec.route_key:
                    s.recipe_index.setdefault(rec.route_key, set()).add(x)
        s.residual = list(data.get("residual", []))
        return s

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "XStore":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
