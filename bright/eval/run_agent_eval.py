#!/usr/bin/env python3
"""Rank BRIGHT passages with Bigdata search and GPT-5.6 Luna.

The agent searches, reads passages search has already returned, and submits a
draft ranking. A second conversation then ranks at most 40 of those passages.
The scored list is that second ranking. No answer is generated.

    python -m bright.eval.run_agent_eval --check
    python -m bright.eval.run_agent_eval biology --limit 1 --dry-run
    python -m bright.eval.run_agent_eval biology
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_PKG = _EVAL_DIR.parent
for path in (_EVAL_DIR, _PKG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loop import BedrockRetryWait, check_bedrock, make_bedrock_generate, run_agent_loop
from dotenv import load_dotenv
from metrics import mean, ndcg_at, recall_at
from paths import (
    CONFIG_PATH,
    DATASETS,
    FINAL_RANKING_PROMPT_PATH,
    PROMPT_PATH,
    atomic_write_json,
    dataset_dir,
    docids_from_field,
    load_config,
    load_jsonl,
)
from searcher import CorpusSearcher
from session import EvalSession

load_dotenv(_PKG.parent / ".env")


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def rendered_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt is empty: {path}")
    return text


def score_session(
    *,
    row: dict,
    session: EvalSession,
    status: str,
    error: str,
    wall_s: float,
    cost_usd: float | None,
    model_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranking = session.ranking() or {}
    gold = docids_from_field(row.get("gold_docs"))
    ranked = list(ranking.get("ranked_docids") or [])
    seen = set(session.seen_docids())
    packed = {
        "query_id": str(row.get("query_id")),
        "status": status,
        "error": error,
        "wall_s": round(wall_s, 3),
        "cost_usd": cost_usd,
        "model_usage": model_usage or {},
        "n_searches": session.n_searches(),
        "n_fetches": session.n_fetches(),
        "ranking_strategy": ranking.get("ranking_strategy"),
        "ndcg@10": ndcg_at(ranked, gold, 10),
        "recall@10": recall_at(ranked, gold, 10),
        "n_ranked": len(ranked),
        "n_seen": len(seen),
        "gold_n": len(set(gold)),
        "gold_seen": len(seen & set(gold)),
    }
    atomic_write_json(session.root / "metrics.json", packed)
    return packed


def run_one_query(
    *,
    row: dict,
    run_dir: Path,
    cfg: dict,
    dry_run: bool,
    force: bool,
    searcher: CorpusSearcher | None,
) -> dict[str, Any]:
    query_id = str(row.get("query_id"))
    qdir = run_dir / "queries" / query_id
    metrics_path = qdir / "metrics.json"
    if metrics_path.is_file() and not force:
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    if not force and (qdir / "ranking.json").is_file():
        recovered = EvalSession(
            qdir,
            query_id=query_id,
            max_searches=int(cfg.get("max_searches") or 80),
            max_fetches=int(cfg.get("max_fetches") or 12),
        )
        if recovered.ranking():
            return score_session(
                row=row,
                session=recovered,
                status="completed",
                error="",
                wall_s=0.0,
                cost_usd=None,
                model_usage={"recovered_existing_ranking": True},
            )
    if qdir.exists():
        shutil.rmtree(qdir)
    qdir.mkdir(parents=True, exist_ok=True)
    session = EvalSession(
        qdir,
        query_id=query_id,
        max_searches=int(cfg.get("max_searches") or 80),
        max_fetches=int(cfg.get("max_fetches") or 12),
    )
    (qdir / "query.txt").write_text(str(row.get("query") or ""), encoding="utf-8")
    if dry_run:
        return score_session(
            row=row,
            session=session,
            status="dry_run",
            error="",
            wall_s=0.0,
            cost_usd=None,
        )
    if searcher is None:
        searcher = CorpusSearcher(
            tag=str(cfg["tag"]),
            ranking_params=cfg.get("ranking_params"),
            state_path=Path(cfg["state_path"]),
        )
    retry_wait = BedrockRetryWait()
    generate = make_bedrock_generate(cfg, retry_wait=retry_wait)
    started = datetime.now(tz=timezone.utc)
    result = run_agent_loop(
        session=session,
        searcher=searcher,
        query=str(row.get("query") or ""),
        generate=generate,
        cfg=cfg,
        system_prompt=rendered_prompt(PROMPT_PATH),
        final_ranking_system_prompt=rendered_prompt(FINAL_RANKING_PROMPT_PATH),
        transcript_path=qdir / "transcript.json",
    )
    wall_s = (datetime.now(tz=timezone.utc) - started).total_seconds()
    return score_session(
        row=row,
        session=session,
        status=result.status,
        error=result.error,
        wall_s=wall_s,
        cost_usd=result.cost_usd,
        model_usage={
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cache_read_tokens": result.cache_read_tokens,
            "cache_write_tokens": result.cache_write_tokens,
            "n_turns": result.n_turns,
            "bedrock_wait_s": round(retry_wait.seconds, 3),
        },
    )


def run_batch(
    rows: list[dict],
    *,
    run_dir: Path,
    cfg: dict,
    dry_run: bool,
    force: bool,
    searcher: CorpusSearcher | None,
    concurrency: int,
) -> list[dict]:
    workers = max(1, min(int(concurrency), len(rows) or 1))
    packed: list[dict | None] = [None] * len(rows)

    def run_at(index: int) -> tuple[int, dict]:
        return index, run_one_query(
            row=rows[index],
            run_dir=run_dir,
            cfg=cfg,
            dry_run=dry_run,
            force=force,
            searcher=None if workers > 1 else searcher,
        )

    if workers == 1:
        for index in range(len(rows)):
            print(f"[{index + 1}/{len(rows)}] query_id={rows[index].get('query_id')}", flush=True)
            _, result = run_at(index)
            packed[index] = result
            print(
                f"status={result.get('status')} ndcg@10={result.get('ndcg@10')} "
                f"cost={result.get('cost_usd')}",
                flush=True,
            )
        return [item for item in packed if item is not None]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_at, index) for index in range(len(rows))]
        for future in as_completed(futures):
            index, result = future.result()
            packed[index] = result
            print(
                f"[{index + 1}/{len(rows)}] query_id={rows[index].get('query_id')} "
                f"status={result.get('status')} ndcg@10={result.get('ndcg@10')}",
                flush=True,
            )
    return [item for item in packed if item is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=DATASETS)
    parser.add_argument("--check", action="store_true", help="Verify AWS credentials via STS")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query-id", action="append", default=[], dest="query_ids")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)

    if args.check:
        try:
            ident = check_bedrock(cfg)
        except Exception as exc:
            print(f"bedrock check failed: {exc}", file=sys.stderr)
            return 1
        print("runtime=bedrock_converse")
        print(f"model={ident['model']}")
        print(f"region={ident['region']}")
        print(f"account={ident['account']}")
        return 0

    if not args.task:
        raise SystemExit("Name a BRIGHT split, for example: biology")
    folder = dataset_dir(args.task)
    jsonl = folder / "agent_queries.jsonl"
    state_path = folder / "upload_state.sqlite"
    if not jsonl.is_file():
        raise SystemExit(f"No queries at {jsonl}. Run: python -m bright.prepare {args.task}")
    if not args.dry_run and not state_path.is_file():
        raise SystemExit(f"No upload index at {state_path}. Upload {args.task} first.")

    rows = load_jsonl(jsonl)
    by_id = {str(row["query_id"]): row for row in rows}
    if args.query_ids:
        missing = [qid for qid in args.query_ids if qid not in by_id]
        if missing:
            raise SystemExit("Unknown query ids: " + ", ".join(missing))
        rows = [by_id[qid] for qid in args.query_ids]
    else:
        rows = rows[args.offset :]
        if args.limit is not None:
            rows = rows[: args.limit]
    if not rows:
        print("No queries selected.", file=sys.stderr)
        return 1

    cfg["tag"] = f"bright-{args.task}"
    cfg["state_path"] = str(state_path)
    if args.concurrency is not None:
        if args.concurrency < 1:
            raise SystemExit("--concurrency must be at least 1")
        cfg["query_concurrency"] = args.concurrency

    run_id = args.run_id or f"{args.task}-bright"
    run_dir = folder / "agent_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "config.json",
        {
            "created_at": _utc_now(),
            "task": args.task,
            "model": cfg.get("model"),
            "tag": cfg["tag"],
            "ranking_review_max_documents": cfg.get("ranking_review_max_documents"),
            "query_ids": [str(row["query_id"]) for row in rows],
        },
    )
    searcher = None
    if not args.dry_run:
        searcher = CorpusSearcher(
            tag=cfg["tag"],
            ranking_params=cfg.get("ranking_params"),
            state_path=state_path,
        )
    print(
        f"run={run_dir} queries={len(rows)} model={cfg.get('model')} "
        f"concurrency={cfg.get('query_concurrency')}",
        flush=True,
    )
    packed = run_batch(
        rows,
        run_dir=run_dir,
        cfg=cfg,
        dry_run=args.dry_run,
        force=args.force,
        searcher=searcher,
        concurrency=int(cfg.get("query_concurrency") or 1),
    )
    summary = {
        "run_id": run_id,
        "task": args.task,
        "updated_at": _utc_now(),
        "n_queries": len(packed),
        "n_completed": sum(1 for row in packed if row.get("status") == "completed"),
        "ndcg@10": mean(row.get("ndcg@10") for row in packed),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in packed),
        "queries": packed,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print(
        f"n={summary['n_queries']} completed={summary['n_completed']} "
        f"ndcg@10={summary['ndcg@10']} cost_usd={summary['total_cost_usd']}"
    )
    print(f"Wrote {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
