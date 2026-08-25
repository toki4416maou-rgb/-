"""
Akane / Lāma-X v1.0 experiment and embedded self-tests.

What this version tests
-----------------------
1. No teacher X sequence is supplied.
2. A bounded candidate search can discover a correct execution structure by
   final-answer verification only.
3. Recurrent successful traces become persistent numerical X recipes.
4. Lāma routing learns to select the correct primitive composition directly.
5. A held-out composition (FILTER -> COMPARE -> SUM) can be built from known
   primitive X without having appeared as a whole training recipe.
6. Ordered recipe integers are exactly reversible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple
import json
import random
import statistics
import time
import tracemalloc

from core import (
    CODE_TO_PRIMITIVE,
    REGISTRY,
    pack_recipe,
    recipe_bytes,
    unpack_recipe,
    run_lama_x,
)
from learning import LamaSystem


COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle", "star"]


class ToyWorldGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def world(self, n: int | None = None) -> Dict[str, Any]:
        n = n or self.rng.randint(5, 16)
        return {
            "objects": [
                {
                    "id": chr(ord("a") + i),
                    "shape": self.rng.choice(SHAPES),
                    "color": self.rng.choice(COLORS),
                    "value": self.rng.randint(0, 30),
                }
                for i in range(n)
            ]
        }

    @staticmethod
    def oracle(world: Mapping[str, Any], query: Mapping[str, Any]) -> Any:
        data = list(world["objects"])

        for f in query.get("filters") or []:
            op = f.get("op", "eq")
            if op == "eq":
                data = [o for o in data if o[f["field"]] == f["value"]]
            elif op == "gt":
                data = [o for o in data if o[f["field"]] > f["value"]]
            elif op == "lt":
                data = [o for o in data if o[f["field"]] < f["value"]]

        cmpq = query.get("compare")
        if cmpq:
            field, op = cmpq["field"], cmpq.get("op", "gt")
            if "reference" in cmpq:
                ref = cmpq["reference"]
                ref_obj = next(o for o in world["objects"] if o["id"] == ref["id"])
                threshold = ref_obj[ref.get("field", field)]
            else:
                threshold = cmpq["value"]

            if op == "gt":
                data = [o for o in data if o[field] > threshold]
            elif op == "lt":
                data = [o for o in data if o[field] < threshold]
            elif op == "eq":
                data = [o for o in data if o[field] == threshold]

        extreme = query.get("extreme")
        if extreme:
            field = query.get("field", "value")
            if not data:
                obj = None
            elif extreme == "max":
                obj = max(data, key=lambda o: o[field])
            else:
                obj = min(data, key=lambda o: o[field])

            if query.get("select"):
                return None if obj is None else obj[query["select"]]
            return obj

        op = query.get("operation")
        field = query.get("field", "value")
        if op == "count":
            return len(data)
        if op == "sum":
            return sum(o[field] for o in data)
        if op == "min":
            return min((o[field] for o in data), default=None)
        if op == "max":
            return max((o[field] for o in data), default=None)
        return data

    def train_task(self):
        """
        Whole recipe FILTER -> COMPARE -> SUM is never produced here.
        """
        w = self.world()
        kind = self.rng.choice([
            "filter_count",
            "filter_sum",
            "compare_count",
            "extreme_select",
        ])

        if kind == "filter_count":
            f = self.rng.choice([
                {"field": "color", "op": "eq", "value": self.rng.choice(COLORS)},
                {"field": "shape", "op": "eq", "value": self.rng.choice(SHAPES)},
            ])
            q = {"filters": [f], "operation": "count"}

        elif kind == "filter_sum":
            f = self.rng.choice([
                {"field": "color", "op": "eq", "value": self.rng.choice(COLORS)},
                {"field": "shape", "op": "eq", "value": self.rng.choice(SHAPES)},
            ])
            q = {"filters": [f], "operation": "sum", "field": "value"}

        elif kind == "compare_count":
            if self.rng.random() < 0.5:
                ref = self.rng.choice(w["objects"])
                q = {
                    "compare": {
                        "field": "value",
                        "op": self.rng.choice(["gt", "lt"]),
                        "reference": {"id": ref["id"], "field": "value"},
                    },
                    "operation": "count",
                }
            else:
                q = {
                    "compare": {
                        "field": "value",
                        "op": self.rng.choice(["gt", "lt"]),
                        "value": self.rng.randint(4, 24),
                    },
                    "operation": "count",
                }

        else:
            q = {
                "extreme": self.rng.choice(["max", "min"]),
                "field": "value",
                "select": "color",
            }

        return w, q, self.oracle(w, q)

    def heldout_task(self):
        """
        Never appears as a whole in training:
            categorical FILTER -> numeric COMPARE -> SUM(value)
        """
        w = self.world()
        filt = self.rng.choice([
            {"field": "color", "op": "eq", "value": self.rng.choice(COLORS)},
            {"field": "shape", "op": "eq", "value": self.rng.choice(SHAPES)},
        ])
        q = {
            "filters": [filt],
            "compare": {
                "field": "value",
                "op": self.rng.choice(["gt", "lt"]),
                "value": self.rng.randint(4, 24),
            },
            "operation": "sum",
            "field": "value",
        }
        return w, q, self.oracle(w, q)


def evaluate(system: LamaSystem, gen: ToyWorldGenerator, n: int, heldout: bool):
    correct = 0
    active = []
    expanded = []
    latency = []
    known_recipe_use = 0

    for _ in range(n):
        w, q, exp = gen.heldout_task() if heldout else gen.train_task()
        t0 = time.perf_counter()
        r = system.infer(w, q)
        latency.append((time.perf_counter() - t0) * 1000.0)
        correct += int(r.value == exp)
        active.append(r.active_x)
        expanded.append(r.expanded_primitives)
        known_recipe_use += int(r.used_known_recipe)

    return {
        "accuracy": correct / max(1, n),
        "avg_active_x": statistics.fmean(active) if active else 0.0,
        "avg_expanded_primitives": statistics.fmean(expanded) if expanded else 0.0,
        "avg_latency_ms": statistics.fmean(latency) if latency else 0.0,
        "known_recipe_use_rate": known_recipe_use / max(1, n),
    }


def self_test() -> Dict[str, bool]:
    from core import REGISTRY, pack_recipe, unpack_recipe

    # Use genuinely non-commutative stages:
    # EXTREME_MAX(value) -> SELECT(color) is valid,
    # SELECT(color) -> EXTREME_MAX(value) is not equivalent.
    a = REGISTRY["EXTREME_MAX_VALUE"].code
    b = REGISTRY["SELECT_COLOR"].code

    x1 = pack_recipe([a, b])
    x2 = pack_recipe([b, a])

    w = {
        "objects": [
            {"id": "a", "shape": "circle", "color": "red", "value": 4},
            {"id": "b", "shape": "square", "color": "blue", "value": 7},
            {"id": "c", "shape": "circle", "color": "green", "value": 2},
        ]
    }
    q = {
        "extreme": "max",
        "field": "value",
        "select": "color",
    }

    correct = run_lama_x(x1, w, q).value
    wrong = run_lama_x(x2, w, q).value

    return {
        "recipe_roundtrip": unpack_recipe(x1) == (a, b),
        "order_is_distinct": x1 != x2,
        "correct_order_result": correct == "blue",
        "wrong_order_is_not_equivalent": wrong != correct,
    }



def indexed_scaling_check(system: LamaSystem, seed: int = 999, decoys: int = 5000):
    """
    Add thousands of valid recipe X to unrelated route buckets and verify that
    target-bucket retrieval remains correct.  This is a functional scaling check,
    not a hardware-independent complexity proof.
    """
    from itertools import product
    from core import REGISTRY, pack_recipe

    gen = ToyWorldGenerator(seed)
    w, q, exp = gen.train_task()

    # Warm target once.
    baseline_samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        r = system.infer(w, q)
        baseline_samples.append((time.perf_counter() - t0) * 1000.0)
    baseline_value = r.value

    primitive_codes = [p.code for p in REGISTRY.values()]
    added = 0
    # Valid ordered recipe integers, but indexed under unrelated buckets.
    for i, combo in enumerate(product(primitive_codes, repeat=3)):
        if added >= decoys:
            break
        x = pack_recipe(combo)
        if x in system.store.records:
            continue
        system.store.add(
            x,
            kind="recipe",
            name=f"decoy-{added}",
            route_key=f"DECOY_BUCKET_{added % 97}",
        )
        added += 1

    after_samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        r2 = system.infer(w, q)
        after_samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "decoy_recipes_added": added,
        "correct_before": baseline_value == exp,
        "correct_after": r2.value == exp,
        "baseline_latency_ms": statistics.fmean(baseline_samples),
        "after_latency_ms": statistics.fmean(after_samples),
        "latency_ratio": (
            statistics.fmean(after_samples) /
            max(1e-12, statistics.fmean(baseline_samples))
        ),
    }


def run_experiment(
    train_episodes: int = 1800,
    validation_episodes: int = 400,
    heldout_episodes: int = 400,
    seed: int = 42,
):
    system = LamaSystem(seed=seed)

    pre = evaluate(
        system,
        ToyWorldGenerator(seed + 50),
        200,
        heldout=False,
    )

    gen = ToyWorldGenerator(seed)

    discover_flags = []
    promotion_flags = []
    tried = []
    reward = []
    active = []
    expanded = []

    # first/last quarter: inference-style selection after each learning episode
    first_probe = []
    last_probe = []

    tracemalloc.start()
    t0 = time.perf_counter()

    for i in range(train_episodes):
        w, q, exp = gen.train_task()
        r = system.learn_episode(w, q, exp)

        discover_flags.append(int(r.discovered))
        promotion_flags.append(int(r.promoted))
        tried.append(r.tried_plans)
        reward.append(r.reward)
        active.append(r.active_x)
        expanded.append(r.expanded_primitives)

        probe = system.infer(w, q)
        probe_ok = int(probe.value == exp)

        quarter = max(1, train_episodes // 4)
        if i < quarter:
            first_probe.append(probe_ok)
        if i >= train_episodes - quarter:
            last_probe.append(probe_ok)

    train_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    validation = evaluate(
        system,
        ToyWorldGenerator(seed + 1),
        validation_episodes,
        heldout=False,
    )
    heldout = evaluate(
        system,
        ToyWorldGenerator(seed + 2),
        heldout_episodes,
        heldout=True,
    )

    stored_recipe_count = sum(r.kind == "recipe" for r in system.store.records.values())
    stored_primitive_count = sum(r.kind == "primitive" for r in system.store.records.values())
    avg_active = statistics.fmean(active) if active else 0.0
    total_stored = len(system.store.records)
    active_ratio = avg_active / max(1, total_stored)

    recipe_sizes = [
        recipe_bytes(x)
        for x, rec in system.store.records.items()
        if rec.kind == "recipe"
    ]

    scaling = indexed_scaling_check(system, seed=seed + 1000, decoys=5000)

    return {
        "version": "Akane Lāma-X Numerical Generative Architecture v1.0",
        "seed": seed,
        "self_test": self_test(),
        "teacher_plan_present": False,
        "train_episodes": train_episodes,
        "pretrain_inference_accuracy": pre["accuracy"],
        "first_quarter_probe_accuracy": statistics.fmean(first_probe),
        "last_quarter_probe_accuracy": statistics.fmean(last_probe),
        "validation_accuracy": validation["accuracy"],
        "heldout_composition_accuracy": heldout["accuracy"],
        "stored_x_total": total_stored,
        "stored_primitive_x": stored_primitive_count,
        "stored_discovered_recipe_x": stored_recipe_count,
        "avg_active_x": avg_active,
        "active_x_over_stored_x": active_ratio,
        "avg_expanded_primitives": statistics.fmean(expanded) if expanded else 0.0,
        "avg_search_plans_tried": statistics.fmean(tried) if tried else 0.0,
        "discovery_rate": statistics.fmean(discover_flags) if discover_flags else 0.0,
        "promotion_count": sum(promotion_flags),
        "avg_reward": statistics.fmean(reward) if reward else 0.0,
        "avg_stored_recipe_bytes": statistics.fmean(recipe_sizes) if recipe_sizes else 0.0,
        "peak_python_memory_bytes": peak,
        "train_time_sec": train_time,
        "validation_avg_latency_ms": validation["avg_latency_ms"],
        "heldout_avg_latency_ms": heldout["avg_latency_ms"],
        "validation_known_recipe_use_rate": validation["known_recipe_use_rate"],
        "heldout_known_recipe_use_rate": heldout["known_recipe_use_rate"],
        "indexed_scaling_check": scaling,
        "gates": {
            "all_self_tests": all(self_test().values()),
            "validation_ge_95": validation["accuracy"] >= 0.95,
            "heldout_ge_80": heldout["accuracy"] >= 0.80,
            "active_ratio_le_25": active_ratio <= 0.25,
            "routing_improved": statistics.fmean(last_probe) >= statistics.fmean(first_probe),
            "indexed_decoys_preserve_correctness": scaling["correct_before"] and scaling["correct_after"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))
