"""Paths and config for the BRIGHT ranking harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PKG_DIR = EVAL_DIR.parent
REPO_DIR = PKG_DIR.parent
DATA_DIR = PKG_DIR / "data"
PROMPT_PATH = EVAL_DIR / "agent_prompt.txt"
FINAL_RANKING_PROMPT_PATH = EVAL_DIR / "final_ranking_prompt.txt"
CONFIG_PATH = EVAL_DIR / "eval_config.json"

DATASETS = (
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
)


def dataset_dir(name: str) -> Path:
    return DATA_DIR / name


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid eval config: {cfg_path}")
    for key, minimum, maximum in (
        ("max_search_calls_per_round", 1, 20),
        ("max_fetch_calls_per_round", 1, 20),
        ("query_concurrency", 1, 20),
        ("ranking_review_max_documents", 0, 500),
    ):
        if key not in data:
            continue
        value = data[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
    effort = data.get("reasoning_effort", "high")
    if effort not in ("", "low", "medium", "high"):
        raise ValueError('reasoning_effort must be "", low, medium, or high')
    return data


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def docids_from_field(field: Any) -> list[str]:
    if not field:
        return []
    if isinstance(field, list):
        out: list[str] = []
        for item in field:
            if isinstance(item, dict) and item.get("docid") is not None:
                out.append(str(item["docid"]))
            else:
                out.append(str(item))
        return out
    return [str(field)]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
