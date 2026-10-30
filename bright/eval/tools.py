"""Tool handlers for the ranking agent."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from session import EvalSession

_CNUM_HEADER_RE = re.compile(r"(?:^|\n)\[cnum (\d+)\]\n")


class SearcherLike(Protocol):
    def search(self, query: str, *, max_chunks: int = 20) -> Any: ...

    def fetch_document(
        self, docid: str, *, max_chunks: int = 200, max_chars: int = 0
    ) -> Any: ...


def budget_line(session: EvalSession) -> str:
    return (
        f"search {session.n_searches()} of max {session.max_searches}, "
        f"get_document {session.n_fetches()} of max {session.max_fetches}."
    )


def _drop_seen_chunks(
    session: EvalSession, raw_hits: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    """Omit chunks this session has already shown."""
    selected: list[dict[str, Any]] = []
    raw_chunk_count = 0
    deduped = 0
    for hit in raw_hits:
        docid = str(hit["docid"])
        seen = session.seen_chunks(docid)
        kept: list[dict[str, Any]] = []
        for chunk in hit.get("chunks") or []:
            raw_chunk_count += 1
            try:
                cnum = int(chunk["cnum"]) if chunk.get("cnum") is not None else None
            except (TypeError, ValueError):
                cnum = None
            if cnum is not None and cnum in seen:
                deduped += 1
                continue
            kept.append(chunk)
        if kept:
            selected.append(
                {"docid": docid, "headline": hit.get("headline") or "", "chunks": kept}
            )
    return selected, raw_chunk_count, deduped


def handle_search(
    session: EvalSession, searcher: SearcherLike, *, text: str = "", max_chunks: int = 20
) -> dict[str, Any]:
    query = str(text or "").strip()
    if not query:
        return {"ok": False, "error": "Provide a text query."}
    if session.retrieval_closed():
        return {"ok": False, "error": "Retrieval closed when the draft ranking was submitted."}
    if session.remaining_searches() <= 0:
        return {
            "ok": False,
            "error": f"Search limit reached ({session.max_searches}). Call submit_ranking.",
        }
    n_chunks = max(1, min(int(max_chunks), 100))
    request = {"tool": "search", "search_mode": "fast", "text": query, "max_chunks": n_chunks}
    started = time.perf_counter()
    result = searcher.search(query, max_chunks=n_chunks)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    raw_hits = [hit.to_agent_dict() for hit in result.hits if hit.docid]
    hits, raw_chunk_count, deduped = _drop_seen_chunks(session, raw_hits)
    try:
        session.record_search(
            request,
            {
                "status": result.status,
                "error": result.error,
                "n_hits": len(hits),
                "n_chunks": sum(len(hit["chunks"]) for hit in hits),
                "raw_n_hits": len(raw_hits),
                "raw_n_chunks": raw_chunk_count,
                "deduped_seen_chunks": deduped,
                "docids": [hit["docid"] for hit in hits],
                "unmapped_ids": result.unmapped_ids,
                "elapsed_ms": elapsed_ms,
                "cnums_by_docid": {
                    hit["docid"]: [
                        chunk["cnum"] for chunk in hit["chunks"] if chunk.get("cnum") is not None
                    ]
                    for hit in hits
                },
                "evidence_documents": hits,
            },
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if result.error or int(result.status or 0) >= 400:
        return {
            "ok": False,
            "error": result.error or f"Search API returned HTTP {result.status}.",
            "status": result.status,
            "budget": budget_line(session),
        }
    return {"ok": True, "budget": budget_line(session), "documents": hits}


def handle_get_document(
    session: EvalSession,
    searcher: SearcherLike,
    *,
    docid: str,
    max_chunks: int = 200,
    max_chars: int = 0,
) -> dict[str, Any]:
    if session.retrieval_closed():
        return {"ok": False, "error": "Retrieval closed when the draft ranking was submitted."}
    if session.remaining_fetches() <= 0:
        return {
            "ok": False,
            "error": (
                f"get_document limit reached ({session.max_fetches}). "
                "Use search, or call submit_ranking."
            ),
        }
    wanted = str(docid or "").strip()
    if not wanted:
        return {"ok": False, "error": "docid is required."}
    if wanted not in set(session.seen_docids()):
        return {
            "ok": False,
            "error": (
                f"docid {wanted} has not appeared in a search result. "
                "Only fetch a passage that search has already returned."
            ),
        }
    already_seen = session.seen_chunks(wanted)
    started = time.perf_counter()
    doc = searcher.fetch_document(wanted, max_chunks=max_chunks, max_chars=max_chars)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    cnums = list(getattr(doc, "cnums", []) or [])
    n_already = len([cnum for cnum in cnums if cnum in already_seen])
    text = str(doc.text or "")
    matches = list(_CNUM_HEADER_RE.finditer(text))
    chunks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunks.append({"cnum": int(match.group(1)), "text": text[match.end() : end].strip()})
    if not chunks and len(cnums) == 1:
        chunks = [{"cnum": cnums[0], "text": text}]
    documents = [{"docid": wanted, "headline": str(doc.headline or ""), "chunks": chunks}]
    try:
        session.record_fetch(
            {"docid": wanted, "max_chunks": max_chunks},
            {
                "n_chunks": doc.n_chunks,
                "n_chars": len(text),
                "chunks_new": len(cnums) - n_already,
                "chunks_already_seen": n_already,
                "cnums": cnums,
                "status": doc.status,
                "error": doc.error,
                "elapsed_ms": elapsed_ms,
                "evidence_documents": documents,
            },
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if doc.error:
        return {"ok": False, "error": doc.error, "budget": budget_line(session)}
    return {
        "ok": True,
        "docid": doc.docid,
        "headline": doc.headline,
        "truncated": doc.truncated,
        "text": text,
        "documents": documents,
        "budget": budget_line(session),
    }


def handle_submit_ranking(
    session: EvalSession, *, documents: list[Any] | None = None, ranking_strategy: str = ""
) -> dict[str, Any]:
    result = session.submit_ranking(documents=documents, ranking_strategy=ranking_strategy)
    if not result.get("ok"):
        return result
    if result.get("phase") == "draft":
        result["next"] = "Draft ranking recorded and retrieval is now closed."
    else:
        result["next"] = "Final ranking recorded."
    return result


def dispatch_tool(
    session: EvalSession,
    searcher: SearcherLike,
    name: str,
    arguments: dict[str, Any] | None,
    *,
    max_chunks: int = 20,
    document_max_chunks: int = 200,
    document_max_chars: int = 0,
) -> dict[str, Any]:
    """Route one tool call to its handler."""
    args = arguments or {}
    if name == "search":
        return handle_search(
            session, searcher, text=str(args.get("text") or ""), max_chunks=max_chunks
        )
    if name == "get_document":
        return handle_get_document(
            session,
            searcher,
            docid=str(args.get("docid") or ""),
            max_chunks=document_max_chunks,
            max_chars=document_max_chars,
        )
    if name == "submit_ranking":
        return handle_submit_ranking(
            session,
            documents=list(args.get("documents") or []),
            ranking_strategy=str(args.get("ranking_strategy") or ""),
        )
    return {"ok": False, "error": f"Unknown tool: {name}"}
