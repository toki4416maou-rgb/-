"""
Akane Web / Common Crawl ingestion v1.1
=======================================

Live web:
    URL -> HTML -> text/link/code extraction -> CanonicalIR

Common Crawl:
    collinfo.json -> current index id
    index query -> WARC filename/offset/length
    HTTP Range fetch from data.commoncrawl.org
    -> response body -> HTML/text CanonicalIR

This module does NOT assert that fetched content is legally reusable.
Callers remain responsible for source terms, copyright, privacy and filtering.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urljoin, quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import gzip
import io
import json
import re

from codec import CanonicalIR, TextCodec


USER_AGENT = "Akane-LamaX-Research/1.1 (+local research client)"


class HTMLExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip = 0
        self._in_code = 0
        self.text_parts: List[str] = []
        self.code_parts: List[str] = []
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "title":
            self._in_title = True
        if t in {"script", "style", "noscript", "svg"}:
            self._skip += 1
        if t in {"code", "pre"}:
            self._in_code += 1
        if t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t == "title":
            self._in_title = False
        if t in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        if t in {"code", "pre"} and self._in_code:
            self._in_code -= 1

    def handle_data(self, data):
        if self._skip:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        if self._in_title:
            self.title += (" " if self.title else "") + cleaned
        if self._in_code:
            self.code_parts.append(cleaned)
        else:
            self.text_parts.append(cleaned)


def extract_html(html: str, base_url: str = "") -> Dict[str, Any]:
    p = HTMLExtractor()
    p.feed(html)
    text = "\n".join(p.text_parts)
    links = []
    seen = set()
    for href in p.links:
        full = urljoin(base_url, href) if base_url else href
        if full not in seen:
            seen.add(full)
            links.append(full)
    return {
        "title": p.title.strip(),
        "text": text,
        "links": links[:1000],
        "code_blocks": p.code_parts[:200],
    }


def fetch_bytes(url: str, headers: Optional[Mapping[str, str]] = None, timeout: int = 20) -> bytes:
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if headers:
        h.update(dict(headers))
    req = Request(url, headers=h)
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_url(url: str, timeout: int = 20, max_bytes: int = 8_000_000) -> CanonicalIR:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urlopen(req, timeout=timeout) as r:
        content_type = r.headers.get("Content-Type", "")
        raw = r.read(max_bytes + 1)

    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    charset = "utf-8"
    m = re.search(r"charset=([\w\-]+)", content_type, re.I)
    if m:
        charset = m.group(1)

    text = raw.decode(charset, errors="replace")
    if "html" in content_type.lower() or "<html" in text[:1000].lower():
        ext = extract_html(text, url)
        body = ext["text"]
        features = {
            "title": ext["title"],
            "link_count": len(ext["links"]),
            "code_block_count": len(ext["code_blocks"]),
            "content_type": content_type,
            "truncated": truncated,
        }
        payload = {
            "links": ext["links"],
            "code_blocks": ext["code_blocks"],
        }
    else:
        body = text
        features = {
            "content_type": content_type,
            "truncated": truncated,
        }
        payload = {}

    text_ir = TextCodec.to_ir(body, source=url)
    text_ir.modality = "web"
    text_ir.intent = "ingest"
    text_ir.features.update(features)
    text_ir.payload.update(payload)
    return text_ir


@dataclass
class CommonCrawlRecord:
    url: str
    filename: str
    offset: int
    length: int
    mime: str = ""
    status: str = ""
    digest: str = ""

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> "CommonCrawlRecord":
        return cls(
            url=str(d.get("url", "")),
            filename=str(d["filename"]),
            offset=int(d["offset"]),
            length=int(d["length"]),
            mime=str(d.get("mime", "")),
            status=str(d.get("status", "")),
            digest=str(d.get("digest", "")),
        )


class CommonCrawlClient:
    COLLINFO = "https://index.commoncrawl.org/collinfo.json"
    DATA_ROOT = "https://data.commoncrawl.org/"

    def __init__(self, index_id: Optional[str] = None, timeout: int = 25):
        self.index_id = index_id
        self.timeout = timeout

    def latest_index_id(self) -> str:
        raw = fetch_bytes(self.COLLINFO, timeout=self.timeout)
        info = json.loads(raw.decode("utf-8"))
        if not info:
            raise RuntimeError("Common Crawl returned no index collections")
        return str(info[0]["id"])

    def resolve_index_id(self) -> str:
        if not self.index_id:
            self.index_id = self.latest_index_id()
        return self.index_id

    def search(
        self,
        url_pattern: str,
        limit: int = 20,
        match_type: str = "domain",
    ) -> List[CommonCrawlRecord]:
        index_id = self.resolve_index_id()
        endpoint = (
            f"https://index.commoncrawl.org/{quote(index_id)}-index"
            f"?url={quote(url_pattern)}&output=json&matchType={quote(match_type)}"
        )
        raw = fetch_bytes(endpoint, timeout=self.timeout)
        records = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "filename" in obj and "offset" in obj and "length" in obj:
                    records.append(CommonCrawlRecord.from_mapping(obj))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                break
        return records

    def fetch_record_bytes(self, rec: CommonCrawlRecord) -> bytes:
        start = rec.offset
        end = rec.offset + rec.length - 1
        url = self.DATA_ROOT + rec.filename
        return fetch_bytes(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=self.timeout,
        )

    @staticmethod
    def _decode_warc_segment(segment: bytes) -> bytes:
        # Common Crawl range segments are normally gzip-compressed WARC records.
        try:
            segment = gzip.decompress(segment)
        except (OSError, EOFError):
            pass

        # WARC headers, then encapsulated HTTP response headers, then body.
        first = segment.find(b"\r\n\r\n")
        if first < 0:
            return segment
        payload = segment[first + 4:]

        if payload.startswith(b"HTTP/"):
            second = payload.find(b"\r\n\r\n")
            if second >= 0:
                return payload[second + 4:]
        return payload

    def fetch_record_ir(self, rec: CommonCrawlRecord) -> CanonicalIR:
        segment = self.fetch_record_bytes(rec)
        body = self._decode_warc_segment(segment)
        text = body.decode("utf-8", errors="replace")

        if "<html" in text[:2000].lower() or "html" in rec.mime.lower():
            ext = extract_html(text, rec.url)
            content = ext["text"]
            payload = {
                "links": ext["links"],
                "code_blocks": ext["code_blocks"],
                "cc_record": rec.__dict__,
            }
            features = {
                "title": ext["title"],
                "link_count": len(ext["links"]),
                "code_block_count": len(ext["code_blocks"]),
                "mime": rec.mime,
                "status": rec.status,
                "common_crawl": True,
            }
        else:
            content = text
            payload = {"cc_record": rec.__dict__}
            features = {
                "mime": rec.mime,
                "status": rec.status,
                "common_crawl": True,
            }

        ir = TextCodec.to_ir(content, source=rec.url)
        ir.modality = "web"
        ir.intent = "ingest"
        ir.features.update(features)
        ir.payload.update(payload)
        return ir


class KnowledgeIngestor:
    """
    Minimal bridge from external IR into Akane memory.

    For v1.1, arbitrary world facts are stored as Residual evidence and repeated
    structural signatures are marked as X-candidates.  Automatic creation of new
    executable cross-modal primitives remains a research boundary rather than
    being faked here.
    """
    def __init__(self, store, repeat_threshold: int = 3):
        self.store = store
        self.repeat_threshold = int(repeat_threshold)
        self.signature_count: Dict[str, int] = {}
        self.seen_fingerprints = set()

    @staticmethod
    def structural_signature(ir: CanonicalIR) -> str:
        keys = sorted(str(k) for k in ir.features.keys())
        return f"{ir.modality}|{ir.intent}|" + ",".join(keys)

    def ingest(self, ir: CanonicalIR) -> Dict[str, Any]:
        fp = ir.fingerprint()
        if fp in self.seen_fingerprints:
            return {"stored": False, "duplicate": True, "x_candidate": False}

        self.seen_fingerprints.add(fp)
        sig = self.structural_signature(ir)
        n = self.signature_count.get(sig, 0) + 1
        self.signature_count[sig] = n

        x_candidate = n >= self.repeat_threshold
        self.store.remember_residual({
            "fingerprint": fp,
            "modality": ir.modality,
            "intent": ir.intent,
            "source": ir.source,
            "features": ir.features,
            "content_excerpt": ir.content[:2000],
            "structural_signature": sig,
            "x_candidate": x_candidate,
        })

        return {
            "stored": True,
            "duplicate": False,
            "x_candidate": x_candidate,
            "signature_count": n,
        }
