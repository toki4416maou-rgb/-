"""
Akane / Lāma-X Numerical Generative Architecture v1.0
=====================================================

Research-complete minimal core.

Key representation
------------------
PrimitiveX:
    fixed 64-bit hierarchical code
    [flags:4][beta:12][alpha:12][omega:12][macro:12][micro:12]

Lāma X (recipe X):
    ONE arbitrary-precision integer encoding an ordered sequence of PrimitiveX codes.

    sentinel = 1
    X = (((1 << 64 | p0) << 64 | p1) ...)

Thus:
    FILTER -> COMPARE -> SUM
and:
    COMPARE -> FILTER -> SUM
are different, exactly reversible numerical X values.

The numerical code is a compact representation of a generative recipe; generated
NeuroSpec is temporary and is not persistent model state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------- 64-bit primitive fields ----------

FIELD_BITS = 12
FIELD_MASK = (1 << FIELD_BITS) - 1
FLAGS_BITS = 4
FLAGS_MASK = (1 << FLAGS_BITS) - 1
PRIMITIVE_BITS = 64
PRIMITIVE_MASK = (1 << PRIMITIVE_BITS) - 1
RECIPE_SENTINEL = 1


class Domain(IntEnum):
    TOY_OBJECTS = 1


class Stage(IntFlag):
    NONE = 0
    FILTER = 1 << 0
    COMPARE = 1 << 1
    EXTREME = 1 << 2
    AGGREGATE = 1 << 3
    SELECT = 1 << 4


class Relation(IntFlag):
    NONE = 0
    EQ = 1 << 0
    GT = 1 << 1
    LT = 1 << 2
    REFERENCE = 1 << 3
    EXTREME_MAX = 1 << 4
    EXTREME_MIN = 1 << 5


class MacroOp(IntEnum):
    NONE = 0
    COUNT = 1
    SUM = 2
    MIN = 3
    MAX = 4
    SELECT = 5


class Field(IntEnum):
    NONE = 0
    ID = 1
    SHAPE = 2
    COLOR = 3
    VALUE = 4


class XFlag(IntFlag):
    NONE = 0
    VERIFY = 1 << 0
    STOP = 1 << 1


FIELD_NAME_TO_CODE = {
    "id": Field.ID,
    "shape": Field.SHAPE,
    "color": Field.COLOR,
    "value": Field.VALUE,
}
FIELD_CODE_TO_NAME = {int(v): k for k, v in FIELD_NAME_TO_CODE.items()}


@dataclass(frozen=True)
class PrimitiveParts:
    beta: int
    alpha: int
    omega: int
    macro: int
    micro: int
    flags: int


@dataclass(frozen=True)
class ObservationSignature:
    """
    Input/context features.
    This is NOT the answer X and does not encode a complete execution recipe.
    """
    features: FrozenSet[str]

    def has(self, token: str) -> bool:
        return token in self.features


@dataclass(frozen=True)
class PrimitiveX:
    name: str
    code: int
    role: str
    capabilities: FrozenSet[str]


@dataclass(frozen=True)
class NeuroSpec:
    opcode: str
    args: Tuple[Any, ...] = ()


@dataclass
class ExecutionResult:
    value: Any
    verified: Optional[bool]
    lama_x: int
    primitive_codes: Tuple[int, ...]
    specs: List[NeuroSpec]
    trace: List[Dict[str, Any]]
    stopped: bool

    @property
    def active_x_count(self) -> int:
        # One recipe X is logically active, even if it expands into multiple primitives.
        return 1 if self.primitive_codes else 0

    @property
    def expanded_primitive_count(self) -> int:
        return len(self.primitive_codes)

    @property
    def steps(self) -> int:
        return len(self.specs)


def _check12(name: str, value: int) -> int:
    value = int(value)
    if not 0 <= value <= FIELD_MASK:
        raise ValueError(f"{name} must fit in 12 bits: {value}")
    return value


def pack_primitive(
    beta: int,
    alpha: int,
    omega: int,
    macro: int,
    micro: int,
    flags: int = 0,
) -> int:
    beta = _check12("beta", beta)
    alpha = _check12("alpha", alpha)
    omega = _check12("omega", omega)
    macro = _check12("macro", macro)
    micro = _check12("micro", micro)
    flags = int(flags)
    if not 0 <= flags <= FLAGS_MASK:
        raise ValueError("flags must fit in 4 bits")
    return (
        micro
        | (macro << 12)
        | (omega << 24)
        | (alpha << 36)
        | (beta << 48)
        | (flags << 60)
    )


def unpack_primitive(code: int) -> PrimitiveParts:
    code = int(code)
    return PrimitiveParts(
        beta=(code >> 48) & FIELD_MASK,
        alpha=(code >> 36) & FIELD_MASK,
        omega=(code >> 24) & FIELD_MASK,
        macro=(code >> 12) & FIELD_MASK,
        micro=code & FIELD_MASK,
        flags=(code >> 60) & FLAGS_MASK,
    )


def pack_micro(target_field: int = 0, select_field: int = 0) -> int:
    return (int(target_field) & 0xF) | ((int(select_field) & 0xF) << 4)


def unpack_micro(micro: int) -> Tuple[int, int]:
    return micro & 0xF, (micro >> 4) & 0xF


# ---------- reversible ordered numerical Lāma X ----------

def pack_recipe(primitive_codes: Sequence[int]) -> int:
    """
    Ordered sequence -> one integer.

    The sentinel makes length recoverable without a separate field.
    Empty recipe is represented by 1.
    """
    x = RECIPE_SENTINEL
    for code in primitive_codes:
        code = int(code)
        if not 0 <= code <= PRIMITIVE_MASK:
            raise ValueError("primitive code must fit in 64 bits")
        x = (x << PRIMITIVE_BITS) | code
    return x


def unpack_recipe(lama_x: int) -> Tuple[int, ...]:
    x = int(lama_x)
    if x < RECIPE_SENTINEL:
        raise ValueError("invalid Lāma X")
    out: List[int] = []
    while x != RECIPE_SENTINEL:
        if x < RECIPE_SENTINEL:
            raise ValueError("corrupt recipe sentinel")
        out.append(x & PRIMITIVE_MASK)
        x >>= PRIMITIVE_BITS
    out.reverse()
    return tuple(out)


def recipe_bytes(lama_x: int) -> int:
    return max(1, (int(lama_x).bit_length() + 7) // 8)


def hierarchy_view(lama_x: int) -> Dict[str, Any]:
    codes = unpack_recipe(lama_x)
    parts = [unpack_primitive(c) for c in codes]
    return {
        "lama_x_decimal": str(lama_x),
        "lama_x_hex": hex(lama_x),
        "recipe_bytes": recipe_bytes(lama_x),
        "recipe_length": len(codes),
        "beta": [p.beta for p in parts],
        "alpha": [p.alpha for p in parts],
        "omega": [p.omega for p in parts],
        "macro": [p.macro for p in parts],
        "micro": [p.micro for p in parts],
        "flags": [p.flags for p in parts],
    }


# ---------- observation ----------

def observe(query: Mapping[str, Any]) -> ObservationSignature:
    """
    Extract neutral task evidence.

    It says things such as "there is a comparison on value" and
    "the requested terminal operation is sum".  It never returns the final X.
    """
    t = set()

    filters = query.get("filters") or []
    if filters:
        t.add("has_filter")
        t.add(f"filter_count:{len(filters)}")
        for item in filters:
            t.add(f"filter_field:{item['field']}")
            t.add(f"filter_op:{item.get('op', 'eq')}")

    cmpq = query.get("compare")
    if cmpq:
        t.add("has_compare")
        t.add(f"compare_field:{cmpq.get('field', 'value')}")
        t.add(f"compare_op:{cmpq.get('op', 'gt')}")
        if "reference" in cmpq:
            t.add("compare_reference")
        else:
            t.add("compare_literal")

    extreme = query.get("extreme")
    if extreme:
        t.add("has_extreme")
        t.add(f"extreme:{extreme}")
        t.add(f"extreme_field:{query.get('field', 'value')}")

    operation = query.get("operation")
    if operation:
        t.add("has_aggregate")
        t.add(f"operation:{operation}")
        if query.get("field"):
            t.add(f"aggregate_field:{query['field']}")

    select = query.get("select")
    if select:
        t.add("has_select")
        t.add(f"select_field:{select}")

    return ObservationSignature(frozenset(t))


# ---------- Primitive X registry ----------

def _field(name: str) -> int:
    return int(FIELD_NAME_TO_CODE[name])


def primitive_registry() -> Dict[str, PrimitiveX]:
    V = _field("value")
    C = _field("color")
    S = _field("shape")

    def px(
        name: str,
        role: str,
        alpha: Stage,
        omega: Relation = Relation.NONE,
        macro: MacroOp = MacroOp.NONE,
        target: int = 0,
        select: int = 0,
        capabilities: Iterable[str] = (),
    ) -> PrimitiveX:
        return PrimitiveX(
            name=name,
            role=role,
            code=pack_primitive(
                int(Domain.TOY_OBJECTS),
                int(alpha),
                int(omega),
                int(macro),
                pack_micro(target, select),
                0,
            ),
            capabilities=frozenset(capabilities),
        )

    r: Dict[str, PrimitiveX] = {}

    # filter
    r["FILTER_COLOR_EQ"] = px(
        "FILTER_COLOR_EQ", "filter", Stage.FILTER, Relation.EQ, target=C,
        capabilities={"filter_field:color", "filter_op:eq"})
    r["FILTER_SHAPE_EQ"] = px(
        "FILTER_SHAPE_EQ", "filter", Stage.FILTER, Relation.EQ, target=S,
        capabilities={"filter_field:shape", "filter_op:eq"})

    # comparison
    for op, relation in (("gt", Relation.GT), ("lt", Relation.LT), ("eq", Relation.EQ)):
        r[f"COMPARE_VALUE_{op.upper()}"] = px(
            f"COMPARE_VALUE_{op.upper()}", "compare", Stage.COMPARE, relation, target=V,
            capabilities={"compare_field:value", f"compare_op:{op}", "compare_literal"})
        r[f"COMPARE_REF_VALUE_{op.upper()}"] = px(
            f"COMPARE_REF_VALUE_{op.upper()}", "compare", Stage.COMPARE,
            relation | Relation.REFERENCE, target=V,
            capabilities={"compare_field:value", f"compare_op:{op}", "compare_reference"})

    # extreme
    r["EXTREME_MAX_VALUE"] = px(
        "EXTREME_MAX_VALUE", "extreme", Stage.EXTREME, Relation.EXTREME_MAX, target=V,
        capabilities={"extreme:max", "extreme_field:value"})
    r["EXTREME_MIN_VALUE"] = px(
        "EXTREME_MIN_VALUE", "extreme", Stage.EXTREME, Relation.EXTREME_MIN, target=V,
        capabilities={"extreme:min", "extreme_field:value"})

    # terminal
    r["COUNT"] = px(
        "COUNT", "terminal", Stage.AGGREGATE, macro=MacroOp.COUNT,
        capabilities={"operation:count"})
    r["SUM_VALUE"] = px(
        "SUM_VALUE", "terminal", Stage.AGGREGATE, macro=MacroOp.SUM, target=V,
        capabilities={"operation:sum", "aggregate_field:value"})
    r["MIN_VALUE"] = px(
        "MIN_VALUE", "terminal", Stage.AGGREGATE, macro=MacroOp.MIN, target=V,
        capabilities={"operation:min", "aggregate_field:value"})
    r["MAX_VALUE"] = px(
        "MAX_VALUE", "terminal", Stage.AGGREGATE, macro=MacroOp.MAX, target=V,
        capabilities={"operation:max", "aggregate_field:value"})
    r["SELECT_COLOR"] = px(
        "SELECT_COLOR", "terminal", Stage.SELECT, macro=MacroOp.SELECT, select=C,
        capabilities={"select_field:color"})

    return r


REGISTRY = primitive_registry()
CODE_TO_PRIMITIVE = {p.code: p for p in REGISTRY.values()}


# ---------- X -> temporary Neuro ----------

def compile_recipe(lama_x: int, query: Mapping[str, Any]) -> List[NeuroSpec]:
    """
    Recipe X + Context -> ephemeral NeuroSpec[].

    The code says which generative operations exist.
    Concrete values ("blue", threshold=12, reference id="a") remain in Context.
    """
    specs: List[NeuroSpec] = []

    for code in unpack_recipe(lama_x):
        p = unpack_primitive(code)
        stage = Stage(p.alpha)
        relation = Relation(p.omega)
        target_code, select_code = unpack_micro(p.micro)
        target_field = FIELD_CODE_TO_NAME.get(target_code)

        if stage & Stage.FILTER:
            for f in query.get("filters") or []:
                if f.get("field") != target_field:
                    continue
                op = f.get("op", "eq")
                specs.append(NeuroSpec(
                    f"FILTER_{ {'eq':'EQ','gt':'GT','lt':'LT'}[op] }",
                    (f["field"], f["value"]),
                ))

        if stage & Stage.COMPARE:
            cmpq = query.get("compare")
            if cmpq and cmpq.get("field", "value") == target_field:
                op = cmpq.get("op", "gt")
                rel_name = {"eq": "EQ", "gt": "GT", "lt": "LT"}[op]
                if (relation & Relation.REFERENCE) and "reference" in cmpq:
                    ref = cmpq["reference"]
                    specs.append(NeuroSpec(
                        f"COMPARE_REF_{rel_name}",
                        (target_field, ref["id"], ref.get("field", target_field)),
                    ))
                elif not (relation & Relation.REFERENCE) and "value" in cmpq:
                    specs.append(NeuroSpec(
                        f"COMPARE_{rel_name}",
                        (target_field, cmpq["value"]),
                    ))

        if stage & Stage.EXTREME:
            field = query.get("field") or target_field or "value"
            if relation & Relation.EXTREME_MAX:
                specs.append(NeuroSpec("EXTREME_MAX", (field,)))
            elif relation & Relation.EXTREME_MIN:
                specs.append(NeuroSpec("EXTREME_MIN", (field,)))

        if stage & Stage.SELECT and p.macro == int(MacroOp.SELECT):
            field = query.get("select") or FIELD_CODE_TO_NAME.get(select_code)
            specs.append(NeuroSpec("SELECT_FIELD", (field,)))

        if stage & Stage.AGGREGATE:
            if p.macro == int(MacroOp.COUNT):
                specs.append(NeuroSpec("AGG_COUNT"))
            elif p.macro == int(MacroOp.SUM):
                specs.append(NeuroSpec(
                    "AGG_SUM", (query.get("field") or target_field or "value",)))
            elif p.macro == int(MacroOp.MIN):
                specs.append(NeuroSpec(
                    "AGG_MIN", (query.get("field") or target_field or "value",)))
            elif p.macro == int(MacroOp.MAX):
                specs.append(NeuroSpec(
                    "AGG_MAX", (query.get("field") or target_field or "value",)))

    specs.append(NeuroSpec("VERIFY"))
    specs.append(NeuroSpec("STOP"))
    return specs


def _filter(data: Sequence[Mapping[str, Any]], field: str, op: str, value: Any):
    if op == "EQ":
        return [o for o in data if o.get(field) == value]
    if op == "GT":
        return [o for o in data if o.get(field) is not None and o.get(field) > value]
    if op == "LT":
        return [o for o in data if o.get(field) is not None and o.get(field) < value]
    raise ValueError(op)


def execute_neuro(
    specs: Sequence[NeuroSpec],
    world: Mapping[str, Any],
) -> Tuple[Any, List[Dict[str, Any]], bool]:
    original = list(world.get("objects") or [])
    state: Any = list(original)
    trace: List[Dict[str, Any]] = []
    stopped = False

    def snap(v: Any) -> Any:
        if isinstance(v, list):
            return {"type": "list", "count": len(v)}
        if isinstance(v, dict):
            return {"type": "object", "id": v.get("id")}
        return v

    for spec in specs:
        before = snap(state)
        op, a = spec.opcode, spec.args

        if op.startswith("FILTER_"):
            state = _filter(state, a[0], op.rsplit("_", 1)[-1], a[1]) \
                if isinstance(state, list) else []

        elif op.startswith("COMPARE_REF_"):
            relation = op.rsplit("_", 1)[-1]
            field, ref_id, ref_field = a
            ref_obj = next((o for o in original if o.get("id") == ref_id), None)
            state = _filter(state, field, relation, ref_obj[ref_field]) \
                if isinstance(state, list) and ref_obj is not None else []

        elif op.startswith("COMPARE_"):
            state = _filter(state, a[0], op.rsplit("_", 1)[-1], a[1]) \
                if isinstance(state, list) else []

        elif op == "EXTREME_MAX":
            state = max(state, key=lambda o: o[a[0]]) \
                if isinstance(state, list) and state else None

        elif op == "EXTREME_MIN":
            state = min(state, key=lambda o: o[a[0]]) \
                if isinstance(state, list) and state else None

        elif op == "SELECT_FIELD":
            state = state.get(a[0]) if isinstance(state, dict) else None

        elif op == "AGG_COUNT":
            state = len(state) if isinstance(state, list) else None

        elif op == "AGG_SUM":
            state = sum(o[a[0]] for o in state) if isinstance(state, list) else None

        elif op == "AGG_MIN":
            state = min((o[a[0]] for o in state), default=None) \
                if isinstance(state, list) else None

        elif op == "AGG_MAX":
            state = max((o[a[0]] for o in state), default=None) \
                if isinstance(state, list) else None

        elif op == "VERIFY":
            pass

        elif op == "STOP":
            stopped = True

        else:
            raise ValueError(f"Unknown opcode: {op}")

        trace.append({
            "opcode": op,
            "args": list(a),
            "before": before,
            "after": snap(state),
        })
        if stopped:
            break

    return state, trace, stopped


def run_lama_x(
    lama_x: int,
    world: Mapping[str, Any],
    query: Mapping[str, Any],
    expected: Any = None,
    has_expected: bool = False,
) -> ExecutionResult:
    specs = compile_recipe(lama_x, query)
    value, trace, stopped = execute_neuro(specs, world)
    verified = (value == expected) if has_expected else None
    codes = unpack_recipe(lama_x)
    return ExecutionResult(
        value=value,
        verified=verified,
        lama_x=lama_x,
        primitive_codes=codes,
        specs=list(specs),
        trace=trace,
        stopped=stopped,
    )
