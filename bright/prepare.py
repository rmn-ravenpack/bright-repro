#!/usr/bin/env python3
"""Download one BRIGHT split and upload its passages to a private Bigdata index.

    python -m bright.prepare biology
    python -m bright.prepare biology upload --concurrency 12

The dataset is downloaded at runtime and is not stored in this repository.
Uploads are private. Passages are tagged bright-<split>.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from bigdata_client import BigdataContentClient, RateLimiter  # noqa: E402
from eval.paths import DATASETS, dataset_dir  # noqa: E402

load_dotenv(PKG.parent / ".env")


def clean_title(doc_id: str) -> str:
    title = doc_id.replace("\\", " ").replace("/", " ")
    title = re.sub(r"\.txt$", "", title, flags=re.IGNORECASE)
    title = title.replace("_", " ").replace("\u2013", "-").replace("\u2014", "-")
    title = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" .")
    return (title or doc_id)[:180]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare(task: str) -> Path:
    from datasets import load_dataset

    folder = dataset_dir(task)
    folder.mkdir(parents=True, exist_ok=True)
    documents = list(load_dataset("xlangai/BRIGHT", "documents", split=task))
    examples = list(load_dataset("xlangai/BRIGHT", "examples", split=task))
    used: dict[str, int] = {}
    with (folder / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for row in documents:
            title = clean_title(str(row["id"]))
            seen = used.get(title, 0) + 1
            used[title] = seen
            file_name = title if seen == 1 else f"{title[:170]} ({seen})"
            text = str(row["content"] or "").replace("\u2013", "-").replace("\u2014", "-").strip()
            handle.write(
                json.dumps(
                    {"doc_id": str(row["id"]), "file_name": file_name, "text": text + "\n"},
                    ensure_ascii=False,
                )
                + "\n"
            )
    gold: dict[str, list[str]] = {}
    with (folder / "queries.jsonl").open("w", encoding="utf-8") as queries:
        for row in examples:
            query_id = str(row["id"])
            gold[query_id] = [str(doc_id) for doc_id in row["gold_ids"]]
            queries.write(
                json.dumps({"query_id": query_id, "query": row["query"]}, ensure_ascii=False) + "\n"
            )
    with (folder / "agent_queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            query_id = str(row["id"])
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "query": row["query"],
                        "gold_docs": gold[query_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(documents)} passages and {len(examples)} queries to {folder}")
    return folder


def connect(folder: Path) -> sqlite3.Connection:
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(folder / "upload_state.sqlite")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT,
            content_id TEXT,
            status TEXT
        )
        """
    )
    return conn


def upload(task: str, concurrency: int) -> None:
    folder = dataset_dir(task)
    documents_path = folder / "documents.jsonl"
    if not documents_path.is_file():
        raise SystemExit(f"No passages at {documents_path}. Prepare {task} first.")
    api_key = os.getenv("BIGDATA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("BIGDATA_API_KEY is required")
    limit = max(1, int(os.getenv("BIGDATA_RATE_LIMIT_PER_MINUTE", "500")) - 20)
    client = BigdataContentClient(api_key, RateLimiter(limit))
    tag = f"bright-{task}"
    documents = load_rows(documents_path)
    conn = connect(folder)
    done = {
        row[0]
        for row in conn.execute(
            "SELECT doc_id FROM documents WHERE status = 'uploaded' AND content_id != ''"
        )
    }
    pending = [doc for doc in documents if doc["doc_id"] not in done]

    def one(document: dict) -> tuple[str, str, str, str]:
        data, status = client.create_document(
            document["file_name"], tags=[tag], share_with_org=False
        )
        if status != 200 or not data or "url" not in data or "id" not in data:
            return document["doc_id"], document["file_name"], "", f"post_{status}"
        body = document["text"].replace("\u2013", "-").replace("\u2014", "-").encode("utf-8")
        ok, put_status = client.put_bytes(data["url"], body)
        if not ok:
            return document["doc_id"], document["file_name"], str(data["id"]), f"put_{put_status}"
        return document["doc_id"], document["file_name"], str(data["id"]), "uploaded"

    uploaded = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, document) for document in pending]
        for future in as_completed(futures):
            doc_id, file_name, content_id, status = future.result()
            conn.execute(
                """
                INSERT INTO documents (doc_id, file_name, content_id, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    content_id = excluded.content_id,
                    status = excluded.status
                """,
                (doc_id, file_name, content_id, status),
            )
            conn.commit()
            if status == "uploaded":
                uploaded += 1
            else:
                failed += 1
                print(f"failed {doc_id} {status}", file=sys.stderr)
            if (uploaded + failed) % 200 == 0:
                print(f"progress uploaded={uploaded} failed={failed}", flush=True)
    print(f"uploaded={uploaded} failed={failed} already={len(done)} tag={tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=DATASETS)
    parser.add_argument("command", nargs="?", choices=["prepare", "upload"], default="prepare")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.task)
        return 0
    upload(args.task, args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
