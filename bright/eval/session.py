"""Per-query session: search log, seen passages, and the two rankings."""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from paths import atomic_write_json

MAX_RANKED_DOCUMENTS = 10


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@dataclass
class EvalSession:
    root: Path
    query_id: str = ""
    max_searches: int = 80
    max_fetches: int = 12
    retrieval_wait_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        meta = _read_json(self.meta_path, {})
        if isinstance(meta, dict) and meta.get("retrieval_wait_s") is not None:
            try:
                self.retrieval_wait_s = float(meta["retrieval_wait_s"])
            except (TypeError, ValueError):
                self.retrieval_wait_s = 0.0

    @property
    def search_log_path(self) -> Path:
        return self.root / "search_calls.jsonl"

    @property
    def fetch_log_path(self) -> Path:
        return self.root / "fetch_calls.jsonl"

    @property
    def seen_path(self) -> Path:
        return self.root / "seen_docids.json"

    @property
    def seen_chunks_path(self) -> Path:
        return self.root / "seen_chunks.json"

    @property
    def evidence_catalog_path(self) -> Path:
        return self.root / "retrieved_evidence.json"

    @property
    def ranking_draft_path(self) -> Path:
        return self.root / "ranking_draft.json"

    @property
    def ranking_path(self) -> Path:
        return self.root / "ranking.json"

    @property
    def ranking_attempts_path(self) -> Path:
        return self.root / "ranking_attempts.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.root / "session.json"

    def n_searches(self) -> int:
        return len(_read_jsonl(self.search_log_path))

    def n_fetches(self) -> int:
        return len(_read_jsonl(self.fetch_log_path))

    def remaining_searches(self) -> int:
        return max(0, self.max_searches - self.n_searches())

    def remaining_fetches(self) -> int:
        return max(0, self.max_fetches - self.n_fetches())

    def add_retrieval_wait(self, seconds: float) -> None:
        """Accumulate the wall time of one turn's parallel retrieval calls."""
        with self._lock:
            self.retrieval_wait_s += max(0.0, float(seconds))
            self._write_meta()

    def search_result_lists(self) -> list[list[str]]:
        """Docids of each search call in result order, oldest call first."""
        return [
            [str(docid) for docid in record.get("docids") or []]
            for record in _read_jsonl(self.search_log_path)
        ]

    def seen_docids(self) -> list[str]:
        data = _read_json(self.seen_path, [])
        if isinstance(data, list):
            return [str(item) for item in data if str(item)]
        return []

    def seen_chunk_map(self) -> dict[str, set[int]]:
        """Every (docid, cnum) pair the tools have shown in this session."""
        data = _read_json(self.seen_chunks_path, {})
        if not isinstance(data, dict):
            return {}
        return {
            str(docid): {int(cnum) for cnum in cnums or [] if str(cnum).isdigit()}
            for docid, cnums in data.items()
        }

    def seen_chunks(self, docid: str) -> set[int]:
        return self.seen_chunk_map().get(str(docid), set())

    def evidence_catalog(self) -> dict[str, dict[str, Any]]:
        data = _read_json(self.evidence_catalog_path, {})
        if not isinstance(data, dict):
            return {}
        return {
            str(docid): document
            for docid, document in data.items()
            if isinstance(document, dict)
        }

    def ranking(self) -> dict[str, Any] | None:
        data = _read_json(self.ranking_path, None)
        return data if isinstance(data, dict) else None

    def ranking_draft(self) -> dict[str, Any] | None:
        data = _read_json(self.ranking_draft_path, None)
        return data if isinstance(data, dict) else None

    def retrieval_closed(self) -> bool:
        """The first submit_ranking stops retrieval."""
        return self.ranking_draft_path.is_file() or self.ranking_path.is_file()

    def _merge_seen_chunks(self, chunks_by_doc: dict[str, list[int]]) -> None:
        data = _read_json(self.seen_chunks_path, {})
        if not isinstance(data, dict):
            data = {}
        for docid, cnums in chunks_by_doc.items():
            merged = {int(c) for c in data.get(docid, []) if str(c).isdigit()}
            merged.update(int(c) for c in cnums)
            data[docid] = sorted(merged)
        atomic_write_json(self.seen_chunks_path, data)

    def _merge_evidence_documents(self, documents: list[dict[str, Any]]) -> None:
        """Keep the text of every chunk shown to the model for the ranking pass."""
        data = _read_json(self.evidence_catalog_path, {})
        if not isinstance(data, dict):
            data = {}
        for document in documents:
            if not isinstance(document, dict):
                continue
            docid = str(document.get("docid") or "").strip()
            if not docid:
                continue
            current = data.get(docid)
            if not isinstance(current, dict):
                current = {"headline": "", "url": "", "chunks": {}}
            headline = str(document.get("headline") or "").strip()
            if headline:
                current["headline"] = headline
            url = str(document.get("url") or "").strip()
            if url:
                current["url"] = url
            chunks = current.get("chunks")
            if not isinstance(chunks, dict):
                chunks = {}
            for chunk in document.get("chunks") or []:
                if not isinstance(chunk, dict):
                    continue
                cnum = chunk.get("cnum")
                if isinstance(cnum, bool) or not str(cnum).isdigit():
                    continue
                chunks[str(int(cnum))] = str(chunk.get("text") or "")
            current["chunks"] = chunks
            data[docid] = current
        atomic_write_json(self.evidence_catalog_path, data)

    def record_search(self, payload: dict[str, Any], result: dict[str, Any]) -> list[str]:
        with self._lock:
            if self.retrieval_closed():
                raise RuntimeError("Retrieval closed when submit_ranking was called.")
            if self.n_searches() >= self.max_searches:
                raise RuntimeError(
                    f"Search limit reached ({self.max_searches}). Call submit_ranking."
                )
            seen = self.seen_docids()
            seen_set = set(seen)
            new_ids: list[str] = []
            for docid in result.get("docids") or []:
                text = str(docid)
                if text and text not in seen_set:
                    seen.append(text)
                    seen_set.add(text)
                    new_ids.append(text)
            atomic_write_json(self.seen_path, seen)
            self._merge_seen_chunks(result.get("cnums_by_docid") or {})
            self._merge_evidence_documents(result.get("evidence_documents") or [])
            record = {
                "ts": _utc_now(),
                "call": self.n_searches() + 1,
                "request": payload,
                "n_hits": result.get("n_hits"),
                "n_chunks": result.get("n_chunks"),
                "raw_n_hits": result.get("raw_n_hits"),
                "raw_n_chunks": result.get("raw_n_chunks"),
                "deduped_seen_chunks": result.get("deduped_seen_chunks"),
                "docids": result.get("docids") or [],
                "new_docids": new_ids,
                "unmapped_ids": result.get("unmapped_ids") or [],
                "status": result.get("status"),
                "error": result.get("error") or "",
                "elapsed_ms": result.get("elapsed_ms"),
            }
            with self.search_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._write_meta()
            return new_ids

    def record_fetch(self, payload: dict[str, Any], result: dict[str, Any]) -> None:
        with self._lock:
            if self.retrieval_closed():
                raise RuntimeError("Retrieval closed when submit_ranking was called.")
            if self.n_fetches() >= self.max_fetches:
                raise RuntimeError(
                    f"get_document limit reached ({self.max_fetches}). "
                    "Use search or submit_ranking."
                )
            record = {
                "ts": _utc_now(),
                "call": self.n_fetches() + 1,
                "request": payload,
                "n_chunks": result.get("n_chunks"),
                "n_chars": result.get("n_chars"),
                "chunks_new": result.get("chunks_new"),
                "chunks_already_seen": result.get("chunks_already_seen"),
                "status": result.get("status"),
                "error": result.get("error") or "",
                "elapsed_ms": result.get("elapsed_ms"),
            }
            with self.fetch_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            docid = str(payload.get("docid") or "")
            if docid:
                self._merge_seen_chunks({docid: result.get("cnums") or []})
            self._merge_evidence_documents(result.get("evidence_documents") or [])
            self._write_meta()

    def _validate_ranked_documents(
        self, documents: list[Any] | None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Parse ranked ids against what the tools actually showed."""
        seen_docs = set(self.seen_docids())
        items = list(documents or [])
        if not items:
            return [], ["documents is empty; rank the documents you judged relevant"]

        def score_key(item: Any) -> float:
            if not isinstance(item, dict):
                return -1.0
            try:
                return float(item.get("relevance_score"))
            except (TypeError, ValueError):
                return -1.0

        parsed: list[dict[str, Any]] = []
        errors: list[str] = []
        used: set[str] = set()
        unknown: list[str] = []
        for item in sorted(items, key=score_key, reverse=True)[:MAX_RANKED_DOCUMENTS]:
            if not isinstance(item, dict):
                errors.append(
                    "each documents entry must be an object with docid and relevance_score"
                )
                continue
            docid = str(item.get("docid") or "").strip()
            if not docid:
                errors.append("docid is empty")
                continue
            if docid not in seen_docs:
                unknown.append(docid)
                continue
            if docid in used:
                # Items are sorted by score, so a repeat is the weaker copy.
                # Rejecting the call over it would cost a round.
                continue
            used.add(docid)
            try:
                score = float(item.get("relevance_score"))
            except (TypeError, ValueError):
                errors.append(f"relevance_score for {docid} must be a number from 0 to 1")
                continue
            if not 0.0 <= score <= 1.0:
                errors.append(f"relevance_score for {docid} must be between 0 and 1")
                continue
            parsed.append({"docid": docid, "relevance_score": score})
        if unknown:
            errors.append(
                "docid was never shown by search or get_document in this query: "
                + ", ".join(unknown)
            )
        if not parsed and not errors:
            errors.append("documents is empty; rank the documents you judged relevant")
        return parsed, errors

    def submit_ranking(
        self, *, documents: list[Any] | None, ranking_strategy: str = ""
    ) -> dict[str, Any]:
        """The first call records the draft. The second records the scored ranking."""
        with self._lock:
            if self.ranking_path.is_file():
                return {
                    "ok": False,
                    "errors": ["both submit_ranking passes have already been called"],
                }
            is_final = self.ranking_draft_path.is_file()
            parsed, errors = self._validate_ranked_documents(documents)
            if not str(ranking_strategy or "").strip():
                errors.append("ranking_strategy is empty")
            attempt = {
                "ts": _utc_now(),
                "pass": "final" if is_final else "draft",
                "documents": documents if isinstance(documents, list) else [],
                "ranking_strategy": ranking_strategy,
                "ok": not errors,
                "errors": errors,
            }
            with self.ranking_attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            if errors:
                return {"ok": False, "errors": errors}
            parsed.sort(key=lambda item: item["relevance_score"], reverse=True)
            ranked_docids = [item["docid"] for item in parsed]
            atomic_write_json(
                self.ranking_path if is_final else self.ranking_draft_path,
                {
                    "query_id": self.query_id,
                    "documents": parsed,
                    "ranked_docids": ranked_docids,
                    "ranking_strategy": str(ranking_strategy).strip(),
                    "submitted_at": _utc_now(),
                },
            )
            self._write_meta()
            return {
                "ok": True,
                "phase": "final" if is_final else "draft",
                "n_documents": len(ranked_docids),
                "ranked_docids": ranked_docids,
            }

    def promote_draft_ranking(self) -> bool:
        """Score the draft when the ranking pass could not complete."""
        with self._lock:
            if self.ranking_path.is_file():
                return False
            draft = _read_json(self.ranking_draft_path, None)
            if not isinstance(draft, dict):
                return False
            promoted = dict(draft)
            docs = list(promoted.get("documents") or [])[:MAX_RANKED_DOCUMENTS]
            promoted["documents"] = docs
            promoted["ranked_docids"] = [
                str(item.get("docid") or "")
                for item in docs
                if isinstance(item, dict) and item.get("docid")
            ]
            atomic_write_json(self.ranking_path, promoted)
            self._write_meta()
            return True

    def _write_meta(self) -> None:
        atomic_write_json(
            self.meta_path,
            {
                "query_id": self.query_id,
                "n_searches": self.n_searches(),
                "n_fetches": self.n_fetches(),
                "n_seen": len(self.seen_docids()),
                "has_ranking_draft": self.ranking_draft_path.is_file(),
                "has_ranking": self.ranking_path.is_file(),
                "retrieval_wait_s": round(self.retrieval_wait_s, 3),
                "updated_at": _utc_now(),
            },
        )
