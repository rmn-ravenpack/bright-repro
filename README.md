# BRIGHT ranking eval

Search one BRIGHT passage corpus with Bigdata.com and rank evidence with
GPT-5.6 Luna on AWS Bedrock. The harness downloads a BRIGHT split, uploads
the passages to a private Bigdata index, runs the questions, and reports
binary nDCG@10 against the official gold passages.

No benchmark plaintext, corpus data, credentials, or run artifacts are
included. Generated files under `bright/data/` are gitignored.

## What it runs

- Corpus and questions: [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT).
- Upload: one private file per passage, tag `bright-<split>`. The filename is
  a cleaned passage id. The harness does not fetch further passages from a
  filename.
- Retrieval: Bigdata Search API `fast` mode over `my_files` plus that tag.
  Each search returns up to 20 chunks. The reranker is on with threshold 0,
  so it reorders hits and does not drop them.
- Tools: `search` and `get_document`. `get_document` reads a passage that
  search has already returned.
- Agent: `us.openai.gpt-5.6-luna`, reasoning effort `high`, at most 80
  searches, 12 document reads, 16 rounds, and three concurrent questions.
- Ranking: the agent calls `submit_ranking` to stop search. A second Bedrock
  call then ranks the draft plus the strongest retrieved passages, at most 40,
  and submits the ten that are scored.

Prompts and budgets are in `bright/eval/`.

## Setup

Python 3.10 is recommended. `requirements-lock.txt` pins the installed
dependency graph; `requirements.txt` lists the direct dependencies. Run the
commands below from this directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env
```

Set `BIGDATA_API_KEY` in `.env`. Prefer the normal AWS credential chain or an
SSO/profile. Never commit `.env`.

## 1. Download and upload one split

```bash
python -m bright.prepare biology
python -m bright.prepare biology upload --concurrency 12
```

Uploads are private. A successful upload means Bigdata accepted the bytes;
indexing continues asynchronously. Repeat `upload` to retry passages that
failed. Do not start the eval until the upload command reports no new
failures.

The prompts were written for the seven StackExchange splits: `biology`,
`earth_science`, `economics`, `psychology`, `robotics`, `stackoverflow`,
`sustainable_living`. The coding and math splits can be prepared and run, but
the prompts were not tuned for them.

## 2. Run the eval

```bash
python -m bright.eval.run_agent_eval --check
python -m bright.eval.run_agent_eval biology --limit 1 --dry-run
python -m bright.eval.run_agent_eval biology
```

`--check` calls AWS STS, not the model. To resume, repeat the same command
with the original `--run-id`. Completed questions are skipped. Search and
generation are stochastic, so scores will vary across runs.

Outputs go to `bright/data/<split>/agent_runs/<run-id>/`. Each question has
its transcript, search log, draft and final ranking, and `metrics.json`.
`summary.json` reports binary nDCG@10 against BRIGHT's gold passages,
averaged over every selected question. A question that ends without a ranking
scores 0.

## Tests

```bash
python -m unittest discover -s bright/eval/tests -p 'test_*.py'
```

Dataset attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
