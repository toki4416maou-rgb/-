"""
Akane Canonical IR + text/code codecs v1.1
===========================================

All external modalities normalize into CanonicalIR before being handed to memory
or to a domain-specific Lāma-X executor.

This module intentionally separates:
    external representation != internal structural representation
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import ast
import hashlib
import json
import re


@dataclass
class CanonicalIR:
    modality: str
    intent: str
    content: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalIR":
        return cls(**dict(data))

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


JP_COLOR = {
    "赤": "red", "赤い": "red",
    "青": "blue", "青い": "blue",
    "緑": "green", "緑の": "green",
    "黄": "yellow", "黄色": "yellow", "黄色い": "yellow",
}
JP_SHAPE = {
    "円": "circle", "丸": "circle", "丸い": "circle", "circle": "circle",
    "四角": "square", "正方形": "square", "square": "square",
    "三角": "triangle", "三角形": "triangle", "triangle": "triangle",
    "星": "star", "star": "star",
}


class TextCodec:
    TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[\u3040-\u30ff\u3400-\u9fff]+|[^\s]")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return cls.TOKEN_RE.findall(text)

    @staticmethod
    def detect_intent(text: str) -> str:
        low = text.lower()
        if any(k in low for k in ["sum", "合計", "総和"]):
            return "aggregate_sum"
        if any(k in low for k in ["count", "数えて", "何個", "いくつ"]):
            return "aggregate_count"
        if any(k in low for k in ["最大", "max", "largest", "maximum"]):
            return "extreme_max"
        if any(k in low for k in ["最小", "min", "smallest", "minimum"]):
            return "extreme_min"
        if any(k in low for k in ["比較", "compare", "違い"]):
            return "compare"
        if any(k in low for k in ["検索", "調べ", "search", "find"]):
            return "search"
        if any(k in low for k in ["コード", "code", "python", "javascript"]):
            return "code"
        return "conversation"

    @classmethod
    def parse_toy_command(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Tiny natural-language bridge for the Toy World benchmark.
        It is a Codec, not the intelligence core.

        Examples:
            青いcircleのvalueを合計
            赤いものを数えて
            valueが10より大きいものを数えて
            最大valueのobjectのcolor
        """
        low = text.lower().replace("　", " ")
        query: Dict[str, Any] = {}

        filters = []
        color = None
        for jp, en in JP_COLOR.items():
            if jp in text or en in low:
                color = en
                break
        if color:
            filters.append({"field": "color", "op": "eq", "value": color})

        shape = None
        for jp, en in JP_SHAPE.items():
            if jp in text or jp in low:
                shape = en
                break
        if shape:
            filters.append({"field": "shape", "op": "eq", "value": shape})

        # numeric comparisons
        m = re.search(r"(?:value|値)[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)[^。,.]{0,8}(?:より)?(大き|以上|greater|>|超)", low)
        if m:
            query["compare"] = {"field": "value", "op": "gt", "value": float(m.group(1))}
        else:
            m = re.search(r"(?:value|値)[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)[^。,.]{0,8}(?:より)?(小さ|以下|less|<|未満)", low)
            if m:
                query["compare"] = {"field": "value", "op": "lt", "value": float(m.group(1))}

        if filters:
            query["filters"] = filters

        if any(k in low for k in ["合計", "sum", "総和"]):
            query["operation"] = "sum"
            query["field"] = "value"
        elif any(k in low for k in ["数えて", "何個", "いくつ", "count"]):
            query["operation"] = "count"
        elif any(k in low for k in ["最大", "maximum", "max"]):
            query["extreme"] = "max"
            query["field"] = "value"
            if "color" in low or "色" in text:
                query["select"] = "color"
        elif any(k in low for k in ["最小", "minimum", "min"]):
            query["extreme"] = "min"
            query["field"] = "value"
            if "color" in low or "色" in text:
                query["select"] = "color"

        return query or None

    @classmethod
    def to_ir(cls, text: str, source: str = "text") -> CanonicalIR:
        tokens = cls.tokenize(text)
        intent = cls.detect_intent(text)
        toy = cls.parse_toy_command(text)

        features = {
            "char_count": len(text),
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
            "has_number": any(any(ch.isdigit() for ch in t) for t in tokens),
            "language_hint": "ja" if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text) else "latin",
        }
        payload: Dict[str, Any] = {"tokens": tokens}
        if toy:
            payload["toy_query"] = toy

        return CanonicalIR(
            modality="text",
            intent=intent,
            content=text,
            features=features,
            payload=payload,
            source=source,
            confidence=1.0,
        )


class CodeCodec:
    @staticmethod
    def detect_language(text: str, filename: str = "") -> str:
        ext = Path(filename).suffix.lower()
        if ext == ".py":
            return "python"
        if ext in {".js", ".mjs", ".cjs"}:
            return "javascript"
        if ext in {".ts", ".tsx"}:
            return "typescript"
        if ext in {".rs"}:
            return "rust"
        if ext in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
            return "cpp"
        if "def " in text or "import " in text:
            return "python"
        if "function " in text or "=>" in text or "const " in text:
            return "javascript"
        return "unknown"

    @staticmethod
    def _python_features(text: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            return {
                "parse_ok": False,
                "syntax_error": str(e),
            }

        funcs = []
        classes = []
        imports = []
        calls = []
        loops = 0
        conditionals = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend(a.name for a in node.names)
                else:
                    imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                loops += 1
            elif isinstance(node, ast.If):
                conditionals += 1

        return {
            "parse_ok": True,
            "functions": funcs,
            "classes": classes,
            "imports": imports,
            "calls": calls[:200],
            "loop_count": loops,
            "conditional_count": conditionals,
            "ast_node_count": sum(1 for _ in ast.walk(tree)),
        }

    @classmethod
    def to_ir(cls, text: str, filename: str = "", source: str = "code") -> CanonicalIR:
        language = cls.detect_language(text, filename)
        if language == "python":
            details = cls._python_features(text)
        else:
            details = {
                "parse_ok": None,
                "function_like_count": len(re.findall(r"\b(?:function|fn|def)\b|=>", text)),
                "class_like_count": len(re.findall(r"\b(?:class|struct|enum)\b", text)),
                "loop_like_count": len(re.findall(r"\b(?:for|while|loop)\b", text)),
                "conditional_like_count": len(re.findall(r"\b(?:if|else|match|switch)\b", text)),
            }

        features = {
            "language": language,
            "line_count": text.count("\n") + 1,
            "char_count": len(text),
            **details,
        }

        return CanonicalIR(
            modality="code",
            intent="analyze_code",
            content=text,
            features=features,
            payload={"filename": filename},
            source=source or filename,
            confidence=1.0 if language != "unknown" else 0.6,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> CanonicalIR:
        p = Path(path)
        return cls.to_ir(
            p.read_text(encoding="utf-8", errors="replace"),
            filename=p.name,
            source=str(p),
        )


def render_ir(ir: CanonicalIR) -> str:
    return json.dumps(ir.to_dict(), ensure_ascii=False, indent=2, default=str)
