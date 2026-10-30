"""Bedrock tool loop for ranking passages.

The model sees search, get_document, and submit_ranking. The first
submit_ranking call stops retrieval. A fresh conversation then reviews the
collected passages and records the scored ranking.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from paths import atomic_write_json
from schemas import RETRIEVAL_TOOL_NAMES, bedrock_tool_config, search_phase_tools
from session import EvalSession
from tools import SearcherLike, dispatch_tool

RETRIEVED_CHUNKS_PREFIX = "RETRIEVED_CHUNKS (deduplicated and grouped by document):\n"
RANKING_MUST_BE_ALONE = (
    "submit_ranking must be the only tool call in its turn. "
    "Retrieval and ranking cannot share a turn."
)
CONTINUE_WITH_TOOL = "Continue. Use a tool. Do not answer in plain text."
ROUND_CAP_KEYS = {
    "search": "max_search_calls_per_round",
    "get_document": "max_fetch_calls_per_round",
}
ROUND_CAP_DEFAULTS = {"search": 5, "get_document": 5}
_ARTICLE_SECTION = re.compile(r"_(\d+)(?=\.txt$)")


def article_stem(docid: str) -> str:
    """Group passage ids such as folder/name_12.txt with folder/name_4.txt.

    Used only to order the ranking-review bundle. It never adds a passage.
    """
    return _ARTICLE_SECTION.sub("", str(docid))


def article_section_sort_key(docid: str) -> tuple[int, int, str]:
    match = _ARTICLE_SECTION.search(str(docid))
    if match:
        return (0, int(match.group(1)), str(docid))
    return (1, 0, str(docid))


def round_caps(cfg: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(cfg.get(ROUND_CAP_KEYS[name]) or ROUND_CAP_DEFAULTS[name])
        for name in ROUND_CAP_DEFAULTS
    }


def over_round_cap(uses: list["ToolUse"], caps: dict[str, int]) -> dict[int, str]:
    """Index the calls past this turn's per-tool ceiling, keeping the earlier ones."""
    used: dict[str, int] = {}
    rejected: dict[int, str] = {}
    for index, use in enumerate(uses):
        cap = caps.get(use.name)
        if cap is None:
            continue
        used[use.name] = used.get(use.name, 0) + 1
        if used[use.name] > cap:
            rejected[index] = (
                f"One turn accepts {cap} {use.name} calls; this was number "
                f"{used[use.name]}. The earlier ones ran. Send this one in the "
                "next turn if the results still leave the gap open."
            )
    return rejected


def transcript_json_safe(value: Any) -> Any:
    """Convert Bedrock binary response fields into JSON values."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_base64__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): transcript_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [transcript_json_safe(item) for item in value]
    return value


def fused_order(result_lists: list[list[str]], k: int = 60) -> list[str]:
    """Docids by reciprocal-rank fusion over result lists, best first."""
    scores: dict[str, float] = {}
    for docids in result_lists:
        for rank, docid in enumerate(docids):
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda docid: -scores[docid])


def ranking_review_prompt(query: str, session: EvalSession, *, max_documents: int = 0) -> str:
    """Build the evidence bundle for the scored ranking pass.

    max_documents > 0 keeps the draft plus the seen passages ranked highest by
    reciprocal-rank fusion of the search result lists, up to that many in total.
    Every chunk shown during retrieval is included for the kept passages.
    """
    draft = session.ranking_draft() or {}
    catalog = session.evidence_catalog()
    seen_map = session.seen_chunk_map()
    draft_ids: list[str] = []
    for item in draft.get("documents") or []:
        if isinstance(item, dict):
            docid = str(item.get("docid") or "").strip()
            if docid and docid not in draft_ids:
                draft_ids.append(docid)
    candidate_ids = session.seen_docids()
    if max_documents > 0 and len(candidate_ids) > max_documents:
        seen_set = set(candidate_ids)
        kept = [docid for docid in draft_ids if docid in seen_set]
        for docid in fused_order(session.search_result_lists()):
            if len(kept) >= max_documents:
                break
            if docid in seen_set and docid not in kept:
                kept.append(docid)
        keep = set(kept)
        candidate_ids = [docid for docid in candidate_ids if docid in keep]
    visible = [
        (docid, sorted(seen_map[docid])) for docid in candidate_ids if seen_map.get(docid)
    ]
    stem_rank: dict[str, tuple[int, int]] = {}
    for docid in draft_ids:
        stem_rank.setdefault(article_stem(docid), (0, len(stem_rank)))
    for docid, _cnums in visible:
        stem_rank.setdefault(article_stem(docid), (1, len(stem_rank)))
    visible.sort(
        key=lambda item: (stem_rank[article_stem(item[0])], article_section_sort_key(item[0]))
    )
    lines = [
        "FINAL_RANKING_REVIEW",
        "",
        "ORIGINAL_QUERY:",
        str(query).strip(),
        "",
        "DRAFT_RANKING:",
        json.dumps(draft, ensure_ascii=False, indent=2),
        "",
        "RETRIEVED_DOCUMENTS:",
    ]
    for docid, cnums in visible:
        document = catalog.get(docid) or {}
        chunks = document.get("chunks") if isinstance(document.get("chunks"), dict) else {}
        lines.append("")
        lines.append(f"DOCUMENT {docid}")
        lines.append(f"TITLE: {str(document.get('headline') or '')}")
        lines.append(f"URL: {str(document.get('url') or '')}")
        for cnum in cnums:
            text = str(chunks.get(str(cnum)) or "")
            lines.append(f"[{docid}:{cnum}]")
            lines.append(text if text else "(text unavailable in persisted catalog)")
    return "\n".join(lines)


class GenerateFn(Protocol):
    def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[str],
        tool_choice: str | None,
    ) -> "ModelTurn": ...


@dataclass
class BedrockRetryWait:
    """Backoff sleeps between throttled Converse attempts. Not model time."""

    seconds: float = 0.0


@dataclass
class ToolUse:
    tool_use_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    assistant_message: dict[str, Any]
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    text: str = ""


@dataclass
class LoopResult:
    status: str
    error: str = ""
    n_turns: int = 0
    forced_ranking: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None


def estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    cache_read_multiplier: float = 0.1,
    cache_write_multiplier: float = 1.25,
) -> float:
    """Bedrock reports uncached input separately from cache reads and writes."""
    mtok = 1_000_000.0
    input_rate = float(input_usd_per_mtok)
    return (
        input_tokens / mtok * input_rate
        + cache_read_tokens / mtok * input_rate * float(cache_read_multiplier)
        + cache_write_tokens / mtok * input_rate * float(cache_write_multiplier)
        + output_tokens / mtok * float(output_usd_per_mtok)
    )


def aws_session_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """boto3 Session kwargs. Access keys in the environment beat AWS_PROFILE."""
    env = os.environ
    region = str(
        cfg.get("aws_region")
        or env.get("BEDROCK_AWS_REGION_NAME")
        or env.get("AWS_REGION")
        or env.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    kwargs: dict[str, Any] = {"region_name": region}
    if env.get("AWS_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        if env.get("AWS_SECRET_ACCESS_KEY"):
            kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
        if env.get("AWS_SESSION_TOKEN"):
            kwargs["aws_session_token"] = env["AWS_SESSION_TOKEN"]
    else:
        profile = str(cfg.get("aws_profile") or env.get("AWS_PROFILE") or "").strip()
        if profile:
            kwargs["profile_name"] = profile
    return kwargs


def make_bedrock_generate(
    cfg: dict[str, Any],
    *,
    client: Any | None = None,
    retry_wait: BedrockRetryWait | None = None,
) -> GenerateFn:
    """Bedrock Converse generate function with throttle retries."""
    if client is None:
        import boto3
        from botocore.config import Config as BotoConfig

        kwargs = aws_session_kwargs(cfg)
        timeout = int(cfg.get("timeout_s") or 900)
        client = boto3.Session(**kwargs).client(
            "bedrock-runtime",
            region_name=kwargs["region_name"],
            config=BotoConfig(
                read_timeout=min(timeout, 300),
                connect_timeout=20,
                retries={"max_attempts": 1},
            ),
        )
    model_id = str(cfg.get("model") or "us.openai.gpt-5.6-luna")
    max_tokens = int(cfg.get("max_tokens") or 8192)
    reasoning_effort = str(cfg.get("reasoning_effort") or "").strip().lower()
    wait = retry_wait if retry_wait is not None else BedrockRetryWait()

    def generate(
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[str],
        tool_choice: str | None,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "modelId": model_id,
            "messages": [ensure_nonempty_message(msg) for msg in messages],
            "system": [{"text": system}],
            "inferenceConfig": {"maxTokens": max_tokens},
            "toolConfig": bedrock_tool_config(tools, tool_choice=tool_choice),
        }
        if reasoning_effort:
            request["additionalModelRequestFields"] = {
                "reasoning": {"effort": reasoning_effort}
            }
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                return parse_converse_response(client.converse(**request))
            except Exception as exc:
                last_error = exc
                if type(exc).__name__ not in {
                    "ThrottlingException",
                    "ModelTimeoutException",
                    "ServiceUnavailableException",
                } and "Throttling" not in str(exc):
                    raise
                delay = min(2 ** attempt, 20)
                time.sleep(delay)
                wait.seconds += delay
        raise last_error or RuntimeError("Bedrock converse failed")

    return generate


def parse_converse_response(response: dict[str, Any]) -> ModelTurn:
    message = ((response.get("output") or {}).get("message")) or {}
    content: list[Any] = []
    uses: list[ToolUse] = []
    texts: list[str] = []
    for block in list(message.get("content") or []):
        content.append(block)
        if not isinstance(block, dict):
            continue
        if block.get("text"):
            texts.append(str(block["text"]))
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            raw_input = tool_use.get("input") or {}
            if not isinstance(raw_input, dict):
                # Echoing a non-object input back fails the whole transcript.
                # An empty object lets the tool report the missing arguments.
                raw_input = {}
                content[-1] = {**block, "toolUse": {**tool_use, "input": raw_input}}
            uses.append(
                ToolUse(
                    tool_use_id=str(tool_use.get("toolUseId") or ""),
                    name=str(tool_use.get("name") or ""),
                    arguments=raw_input,
                )
            )
    usage = response.get("usage") or {}
    return ModelTurn(
        assistant_message={"role": "assistant", "content": content},
        tool_uses=uses,
        stop_reason=str(response.get("stopReason") or ""),
        input_tokens=int(usage.get("inputTokens") or 0),
        output_tokens=int(usage.get("outputTokens") or 0),
        cache_read_tokens=int(usage.get("cacheReadInputTokens") or 0),
        cache_write_tokens=int(usage.get("cacheWriteInputTokens") or 0),
        text="\n".join(texts).strip(),
    )


def ensure_nonempty_message(message: dict[str, Any]) -> dict[str, Any]:
    """Bedrock rejects a message whose content is empty."""
    out = dict(message)
    if out.get("content"):
        return out
    if out.get("role") == "assistant":
        out["content"] = [{"text": "(no output)"}]
    else:
        out["content"] = [{"text": CONTINUE_WITH_TOOL}]
    return out


def _tool_result_block(tool_use_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "status": "success" if payload.get("ok", True) else "error",
            "content": [{"json": payload}],
        }
    }


def prepare_retrieval_results(
    payloads: list[dict[str, Any] | None], retrieval_indexes: list[int]
) -> list[dict[str, Any]]:
    """Shrink each retrieval result to chunk ids and return one grouped bundle.

    The same chunk can come back from several calls in one turn. The model
    reads its text once, in the bundle, grouped by passage.
    """
    documents: dict[str, dict[str, Any]] = {}
    seen_keys: dict[str, set[tuple[str, Any]]] = {}
    for index in retrieval_indexes:
        payload = payloads[index]
        if not isinstance(payload, dict):
            continue
        matched: list[str] = []
        for document in payload.get("documents") or []:
            if not isinstance(document, dict):
                continue
            docid = str(document.get("docid") or "")
            if not docid:
                continue
            grouped = documents.setdefault(
                docid,
                {
                    "docid": docid,
                    "title": document.get("title") or document.get("headline") or "",
                    "url": document.get("url") or "",
                    "chunks": [],
                },
            )
            keys = seen_keys.setdefault(docid, set())
            for chunk in document.get("chunks") or []:
                if not isinstance(chunk, dict):
                    continue
                try:
                    cnum = int(chunk["cnum"]) if chunk.get("cnum") is not None else None
                except (TypeError, ValueError):
                    cnum = None
                key = ("cnum", cnum) if cnum is not None else ("text", str(chunk.get("text") or ""))
                if cnum is not None:
                    matched.append(f"{docid}:{cnum}")
                if key not in keys:
                    keys.add(key)
                    grouped["chunks"].append({"cnum": cnum, "text": str(chunk.get("text") or "")})
        compact: dict[str, Any] = {
            "ok": bool(payload.get("ok", True)),
            "matched_chunk_ids": list(dict.fromkeys(matched)),
        }
        if not compact["ok"]:
            compact["error"] = str(payload.get("error") or "retrieval failed")
        if "truncated" in payload:
            compact["truncated"] = bool(payload["truncated"])
        payload.clear()
        payload.update(compact)

    def chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, str]:
        cnum = chunk.get("cnum")
        if isinstance(cnum, int):
            return cnum, ""
        return 2**31 - 1, str(chunk.get("text") or "")

    for document in documents.values():
        document["chunks"].sort(key=chunk_sort_key)
    return list(documents.values())


def execute_turn_tools(
    uses: list[ToolUse],
    *,
    session: EvalSession,
    searcher: SearcherLike,
    cfg: dict[str, Any],
    allowed: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run this turn's tool calls. Retrieval calls in one turn run in parallel."""
    uses = list(uses)
    ranking_mixed = any(use.name == "submit_ranking" for use in uses) and len(uses) > 1
    capped = over_round_cap(uses, round_caps(cfg))
    dispatch_kwargs = {
        "max_chunks": int(cfg.get("max_chunks") or 20),
        "document_max_chunks": int(cfg.get("document_max_chunks") or 200),
        "document_max_chars": int(cfg.get("document_max_chars") or 0),
    }

    def run_one(index: int) -> dict[str, Any]:
        use = uses[index]
        skip_reason = capped.get(index, "")
        if not skip_reason and use.name not in allowed:
            skip_reason = f"Tool {use.name} is not available in this phase."
        if not skip_reason and ranking_mixed and use.name == "submit_ranking":
            skip_reason = RANKING_MUST_BE_ALONE
        if skip_reason:
            return {"ok": False, "error": skip_reason}
        return dispatch_tool(session, searcher, use.name, use.arguments, **dispatch_kwargs)

    payloads: list[dict[str, Any] | None] = [None] * len(uses)
    parallel_idxs = [i for i, use in enumerate(uses) if use.name in RETRIEVAL_TOOL_NAMES]
    if parallel_idxs:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(parallel_idxs)) as pool:
            futures = {pool.submit(run_one, i): i for i in parallel_idxs}
            for future, index in futures.items():
                payloads[index] = future.result()
        session.add_retrieval_wait(time.monotonic() - started)
    for index in range(len(uses)):
        if payloads[index] is None:
            payloads[index] = run_one(index)
    evidence_documents = prepare_retrieval_results(payloads, parallel_idxs)

    blocks: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for use, payload in zip(uses, payloads):
        assert payload is not None
        blocks.append(_tool_result_block(use.tool_use_id, payload))
        log.append(
            {
                "name": use.name,
                "tool_use_id": use.tool_use_id,
                "ok": bool(payload.get("ok", True)),
                "arguments": use.arguments,
            }
        )
    if parallel_idxs:
        blocks.append(
            {
                "text": RETRIEVED_CHUNKS_PREFIX
                + json.dumps(evidence_documents, ensure_ascii=False, indent=2)
            }
        )
    return blocks, log


def run_agent_loop(
    *,
    session: EvalSession,
    searcher: SearcherLike,
    query: str,
    generate: GenerateFn,
    cfg: dict[str, Any],
    system_prompt: str,
    final_ranking_system_prompt: str,
    transcript_path: Path | None = None,
) -> LoopResult:
    max_rounds = int(cfg.get("max_rounds") or 16)
    timeout_s = int(cfg.get("timeout_s") or 900)
    max_final_attempts = int(cfg.get("max_final_ranking_attempts") or 2)
    review_max_documents = int(cfg.get("ranking_review_max_documents") or 0)
    started = time.monotonic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": f"USER_QUERY:\n{query.strip()}"}]}
    ]
    review_messages: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    n_turns = 0
    forced_ranking = False

    def check_time() -> None:
        if timeout_s - (time.monotonic() - started) <= 1:
            raise TimeoutError(f"timeout after {timeout_s}s")

    def record_turn(kind: str, turn: ModelTurn, tool_log: list[dict[str, Any]]) -> None:
        nonlocal n_turns
        n_turns += 1
        totals["input"] += turn.input_tokens
        totals["output"] += turn.output_tokens
        totals["cache_read"] += turn.cache_read_tokens
        totals["cache_write"] += turn.cache_write_tokens
        trace.append(
            {
                "turn": n_turns,
                "kind": kind,
                "stop_reason": turn.stop_reason,
                "text": turn.text,
                "tools": tool_log,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "cache_read_tokens": turn.cache_read_tokens,
                "cache_write_tokens": turn.cache_write_tokens,
            }
        )

    def retrieval_turn(kind: str, tools: list[str], tool_choice: str | None = None) -> None:
        check_time()
        turn = generate(
            messages=messages, system=system_prompt, tools=tools, tool_choice=tool_choice
        )
        messages.append(ensure_nonempty_message(turn.assistant_message))
        blocks, tool_log = execute_turn_tools(
            turn.tool_uses, session=session, searcher=searcher, cfg=cfg, allowed=set(tools)
        )
        messages.append({"role": "user", "content": blocks or [{"text": CONTINUE_WITH_TOOL}]})
        record_turn(kind, turn, tool_log)

    def final_ranking_turn() -> None:
        check_time()
        prompt = {
            "role": "user",
            "content": [
                {
                    "text": ranking_review_prompt(
                        query, session, max_documents=review_max_documents
                    )
                }
            ],
        }
        turn = generate(
            messages=[prompt],
            system=final_ranking_system_prompt,
            tools=["submit_ranking"],
            tool_choice="submit_ranking",
        )
        blocks, tool_log = execute_turn_tools(
            turn.tool_uses,
            session=session,
            searcher=searcher,
            cfg=cfg,
            allowed={"submit_ranking"},
        )
        review_messages.extend(
            [
                prompt,
                ensure_nonempty_message(turn.assistant_message),
                {
                    "role": "user",
                    "content": blocks or [{"text": "Final ranking was not submitted; try again."}],
                },
            ]
        )
        record_turn("final_ranking_review", turn, tool_log)

    try:
        # max_rounds bounds retrieval. The forced draft and the ranking pass
        # run after that budget so a late draft is never discarded.
        final_attempts = 0
        force_attempts = 0
        while not session.ranking():
            if session.ranking_draft():
                if final_attempts < max_final_attempts:
                    final_attempts += 1
                    final_ranking_turn()
                    continue
                session.promote_draft_ranking()
                break
            if n_turns >= max_rounds:
                if force_attempts >= 2:
                    break
                force_attempts += 1
                forced_ranking = True
                retrieval_turn("force_draft_ranking", ["submit_ranking"], "submit_ranking")
                continue
            retrieval_turn(
                "search", search_phase_tools(can_rank=bool(session.seen_chunk_map()))
            )
        if session.ranking():
            status, error = "completed", ""
        elif session.ranking_draft():
            status, error = "no_submit", "final submit_ranking pass was not called"
        else:
            status, error = "no_submit", "draft submit_ranking pass was not called"
    except TimeoutError as exc:
        status, error = "timeout", str(exc)
    except Exception as exc:
        status, error = "error", str(exc)

    if transcript_path is not None:
        atomic_write_json(
            transcript_path,
            transcript_json_safe(
                {
                    "messages": messages,
                    "ranking_review_messages": review_messages,
                    "turns": trace,
                    "status": status,
                    "error": error,
                }
            ),
        )

    return LoopResult(
        status=status,
        error=error,
        n_turns=n_turns,
        forced_ranking=forced_ranking,
        input_tokens=totals["input"],
        output_tokens=totals["output"],
        cache_read_tokens=totals["cache_read"],
        cache_write_tokens=totals["cache_write"],
        cost_usd=estimate_cost_usd(
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            cache_read_tokens=totals["cache_read"],
            cache_write_tokens=totals["cache_write"],
            input_usd_per_mtok=float(cfg.get("input_usd_per_mtok") or 0.0),
            output_usd_per_mtok=float(cfg.get("output_usd_per_mtok") or 0.0),
            cache_read_multiplier=float(cfg.get("cache_read_multiplier") or 0.1),
            cache_write_multiplier=float(cfg.get("cache_write_multiplier") or 1.25),
        ),
    )


def check_bedrock(cfg: dict[str, Any]) -> dict[str, str]:
    """Cheap credential check: STS caller identity, not a model call."""
    import boto3

    kwargs = aws_session_kwargs(cfg)
    region = kwargs.get("region_name") or "us-east-1"
    ident = boto3.Session(**kwargs).client("sts", region_name=region).get_caller_identity()
    return {
        "account": str(ident.get("Account") or ""),
        "arn": str(ident.get("Arn") or ""),
        "region": region,
        "model": str(cfg.get("model") or ""),
    }
