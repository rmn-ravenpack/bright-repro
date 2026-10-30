"""Search a private Bigdata index of one BRIGHT passage corpus."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
load_dotenv(_REPO / ".env")

API_BASE = os.getenv("BIGDATA_API_BASE_URL", "https://api.bigdata.com").rstrip("/")
SEARCH_URL = f"{API_BASE}/v1/search"
POST_ATTEMPTS = 6

RANKING_PARAMS = {
    "source_boost": 0,
    "freshness_boost": 0,
    "content_diversification": {"enabled": False},
    "reranker": {"enabled": True, "threshold": 0},
}

DOCUMENT_RANKING_PARAMS = {
    "source_boost": 0,
    "freshness_boost": 0,
    "content_diversification": {"enabled": False},
    "reranker": {"enabled": False},
}


@dataclass
class Chunk:
    cnum: int | None
    text: str
    relevance: float


@dataclass
class Hit:
    content_id: str
    docid: str | None
    headline: str
    score: float
    chunks: list[Chunk] = field(default_factory=list)

    def to_agent_dict(self) -> dict[str, Any]:
        """Fields the model is allowed to see. No search scores."""
        return {
            "docid": self.docid,
            "headline": self.headline,
            "chunks": [{"cnum": chunk.cnum, "text": chunk.text} for chunk in self.chunks],
        }


@dataclass
class SearchResult:
    status: int
    hits: list[Hit]
    unmapped_ids: list[str]
    error: str = ""


@dataclass
class DocumentResult:
    docid: str
    content_id: str | None
    status: int
    headline: str = ""
    text: str = ""
    n_chunks: int = 0
    truncated: bool = False
    error: str = ""
    cnums: list[int] = field(default_factory=list)


class CorpusSearcher:
    def __init__(
        self,
        *,
        tag: str,
        state_path: Path,
        api_key: str | None = None,
        timeout: int = 90,
        ranking_params: dict[str, Any] | None = None,
    ):
        self.api_key = (api_key or os.getenv("BIGDATA_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("BIGDATA_API_KEY is not set")
        self.tag = str(tag or "").strip()
        if not self.tag:
            raise RuntimeError("A corpus tag is required")
        self.timeout = timeout
        self.ranking_params = dict(ranking_params or RANKING_PARAMS)
        self.content_id_to_docid = self._load_mapping(state_path)
        self.docid_to_content_id = {
            docid: cid for cid, docid in self.content_id_to_docid.items()
        }

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, str]:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            """
            SELECT content_id, doc_id FROM documents
            WHERE status='uploaded' AND TRIM(content_id) != ''
            """
        ).fetchall()
        conn.close()
        return {str(cid).upper(): str(docid) for cid, docid in rows}

    def _filters(self) -> dict[str, Any]:
        return {
            "category": {"mode": "INCLUDE", "values": ["my_files"]},
            "tag": {"any_of": [self.tag]},
        }

    def _post(self, payload: dict) -> tuple[int, dict]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        for attempt in range(POST_ATTEMPTS):
            last = attempt == POST_ATTEMPTS - 1
            try:
                resp = requests.post(
                    SEARCH_URL, json=payload, headers=headers, timeout=self.timeout
                )
            except requests.ConnectionError:
                if last:
                    raise
            else:
                if resp.status_code != 429 or last:
                    break
            time.sleep(min(30, 2 ** attempt))
        try:
            data = resp.json() if resp.text else {}
        except ValueError:
            data = {"raw": (resp.text or "")[:500]}
        if not isinstance(data, dict):
            data = {"raw": data}
        return resp.status_code, data

    def _collapse(self, results: list[dict]) -> tuple[list[Hit], list[str]]:
        """One hit per passage, keeping API order, with chunks by relevance."""
        best: dict[str, Hit] = {}
        order: list[str] = []
        unmapped: list[str] = []
        for doc in results:
            cid = str(doc.get("id") or "").upper()
            if not cid:
                continue
            chunks = [
                Chunk(
                    cnum=chunk.get("cnum"),
                    text=str(chunk.get("text") or ""),
                    relevance=float(chunk.get("relevance") or 0.0),
                )
                for chunk in doc.get("chunks") or []
            ]
            chunks.sort(key=lambda c: c.relevance, reverse=True)
            hit = Hit(
                content_id=cid,
                docid=self.content_id_to_docid.get(cid),
                headline=str(doc.get("headline") or ""),
                score=max((chunk.relevance for chunk in chunks), default=0.0),
                chunks=chunks,
            )
            if cid not in best:
                order.append(cid)
                best[cid] = hit
            elif hit.score > best[cid].score:
                best[cid] = hit
            if hit.docid is None:
                unmapped.append(cid)
        return [best[cid] for cid in order if best[cid].docid], unmapped

    def search_payload(self, query: str, *, max_chunks: int = 20) -> dict[str, Any]:
        return {
            "search_mode": "fast",
            "query": {
                "text": query,
                "filters": self._filters(),
                "ranking_params": dict(self.ranking_params),
                "max_chunks": max_chunks,
                "auto_enrich_filters": False,
            },
        }

    def search(self, query: str, *, max_chunks: int = 20) -> SearchResult:
        status, data = self._post(self.search_payload(query, max_chunks=max_chunks))
        if status >= 400:
            return SearchResult(
                status=status, hits=[], unmapped_ids=[], error=json.dumps(data)[:400]
            )
        hits, unmapped = self._collapse(data.get("results") or [])
        return SearchResult(status=status, hits=hits, unmapped_ids=unmapped)

    def document_payload(self, content_id: str, *, max_chunks: int = 200) -> dict[str, Any]:
        """Chunk retrieval within one known passage, not a relevance search."""
        filters = self._filters()
        filters["document"] = {"mode": "INCLUDE", "values": [content_id]}
        filters["chunk"] = {"from": 1, "to": max_chunks}
        return {
            "search_mode": "fast",
            "query": {
                "filters": filters,
                "ranking_params": dict(DOCUMENT_RANKING_PARAMS),
                "max_chunks": max_chunks,
                "auto_enrich_filters": False,
            },
        }

    def fetch_document(
        self, docid: str, *, max_chunks: int = 200, max_chars: int = 0
    ) -> DocumentResult:
        """Reassemble one passage from its chunks, each labelled [cnum N]."""
        content_id = self.docid_to_content_id.get(str(docid))
        if not content_id:
            return DocumentResult(
                docid=str(docid), content_id=None, status=0, error="unknown docid"
            )
        status, data = self._post(self.document_payload(content_id, max_chunks=max_chunks))
        if status >= 400:
            return DocumentResult(
                docid=str(docid),
                content_id=content_id,
                status=status,
                error=json.dumps(data)[:400],
            )
        results = data.get("results") or []
        if not results:
            return DocumentResult(
                docid=str(docid),
                content_id=content_id,
                status=status,
                error="document not found in index",
            )
        doc = results[0]
        chunks = sorted(
            (c for c in (doc.get("chunks") or []) if c.get("text")),
            key=lambda c: c.get("cnum") or 0,
        )
        parts: list[str] = []
        shown_cnums: list[int] = []
        chars_used = 0
        truncated = False
        for chunk in chunks:
            cnum = int(chunk.get("cnum") or 0)
            separator = "\n\n" if parts else ""
            addition = f"{separator}[cnum {cnum}]\n{str(chunk.get('text') or '').strip()}"
            if max_chars and chars_used + len(addition) > max_chars:
                remaining = max_chars - chars_used
                if remaining > 0:
                    parts.append(addition[:remaining])
                    shown_cnums.append(cnum)
                truncated = True
                break
            parts.append(addition)
            shown_cnums.append(cnum)
            chars_used += len(addition)
        if len(chunks) >= max_chunks:
            truncated = True
        return DocumentResult(
            docid=str(docid),
            content_id=content_id,
            status=status,
            headline=str(doc.get("headline") or ""),
            text="".join(parts),
            n_chunks=len(shown_cnums),
            truncated=truncated,
            cnums=shown_cnums,
        )
