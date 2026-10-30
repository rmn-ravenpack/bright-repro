"""Checks for the published BRIGHT harness."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1]
_PKG = _EVAL.parent
for path in (_EVAL, _PKG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loop import ModelTurn, ToolUse, ranking_review_prompt, run_agent_loop  # noqa: E402
from metrics import ndcg_at  # noqa: E402
from schemas import search_phase_tools, tool_schemas  # noqa: E402
from searcher import Chunk, DocumentResult, Hit, SearchResult  # noqa: E402
from session import EvalSession  # noqa: E402
from tools import dispatch_tool, handle_submit_ranking  # noqa: E402


class FakeSearcher:
    def __init__(self) -> None:
        self.fetches: list[str] = []

    def search(self, query: str, *, max_chunks: int = 20) -> SearchResult:
        return SearchResult(
            status=200,
            hits=[
                Hit(
                    content_id="C1",
                    docid="1234",
                    headline="Example",
                    score=0.9,
                    chunks=[Chunk(cnum=0, text="hello", relevance=0.9)],
                )
            ],
            unmapped_ids=[],
        )

    def fetch_document(self, docid: str, **kwargs) -> DocumentResult:
        self.fetches.append(docid)
        return DocumentResult(
            docid=docid,
            content_id="C1",
            status=200,
            headline="Example",
            text="[cnum 0]\nhello",
            n_chunks=1,
            cnums=[0],
        )


def _turn(*uses: ToolUse) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": use.tool_use_id, "name": use.name, "input": use.arguments}}
                for use in uses
            ],
        },
        tool_uses=list(uses),
        input_tokens=100,
        output_tokens=10,
    )


class HarnessTests(unittest.TestCase):
    def test_published_tools_are_search_read_and_rank(self) -> None:
        self.assertEqual(
            [tool["name"] for tool in tool_schemas()],
            ["search", "get_document", "submit_ranking"],
        )
        self.assertEqual(search_phase_tools(can_rank=False), ["search", "get_document"])

    def test_get_document_reads_only_a_passage_search_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = EvalSession(tmp, query_id="1", max_searches=3, max_fetches=2)
            searcher = FakeSearcher()
            unseen = dispatch_tool(session, searcher, "get_document", {"docid": "1234"})
            self.assertFalse(unseen["ok"])
            self.assertEqual(searcher.fetches, [])
            dispatch_tool(session, searcher, "search", {"text": "what explains this?"})
            seen = dispatch_tool(session, searcher, "get_document", {"docid": "1234"})
            self.assertTrue(seen["ok"])
            self.assertEqual(searcher.fetches, ["1234"])

    def test_ranking_review_keeps_the_draft_then_fused_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = EvalSession(tmp, query_id="1")
            for docids in (["a", "b", "c"], ["b", "d", "e"]):
                session.record_search(
                    {"text": "q"},
                    {
                        "docids": docids,
                        "cnums_by_docid": {docid: [1] for docid in docids},
                        "evidence_documents": [
                            {
                                "docid": docid,
                                "headline": docid,
                                "chunks": [{"cnum": 1, "text": f"body {docid}"}],
                            }
                            for docid in docids
                        ],
                    },
                )
            self.assertTrue(
                handle_submit_ranking(
                    session,
                    documents=[{"docid": "e", "relevance_score": 0.9}],
                    ranking_strategy="draft",
                )["ok"]
            )
            prompt = ranking_review_prompt("Who?", session, max_documents=3)
            for kept in ("body e", "body b", "body a"):
                self.assertIn(kept, prompt)
            for dropped in ("body c", "body d"):
                self.assertNotIn(dropped, prompt)

    def test_loop_scores_the_second_ranking(self) -> None:
        turns = iter(
            [
                _turn(ToolUse("t1", "search", {"text": "why?"})),
                _turn(
                    ToolUse(
                        "t2",
                        "submit_ranking",
                        {"ranking_strategy": "draft", "documents": [{"docid": "1234", "relevance_score": 0.5}]},
                    )
                ),
                _turn(
                    ToolUse(
                        "t3",
                        "submit_ranking",
                        {"ranking_strategy": "final", "documents": [{"docid": "1234", "relevance_score": 0.9}]},
                    )
                ),
            ]
        )
        calls: list[dict] = []

        def generate(**kwargs) -> ModelTurn:
            calls.append(kwargs)
            return next(turns)

        with tempfile.TemporaryDirectory() as tmp:
            session = EvalSession(tmp, query_id="1")
            result = run_agent_loop(
                session=session,
                searcher=FakeSearcher(),
                query="why?",
                generate=generate,
                cfg={"max_rounds": 4},
                system_prompt="search",
                final_ranking_system_prompt="rank",
                transcript_path=Path(tmp) / "transcript.json",
            )
            self.assertEqual(result.status, "completed", result.error)
            self.assertEqual(session.ranking()["ranking_strategy"], "final")
            self.assertEqual(calls[0]["tools"], ["search", "get_document"])
            self.assertEqual(calls[1]["tools"], ["search", "get_document", "submit_ranking"])
            self.assertEqual(calls[2]["system"], "rank")
            self.assertEqual(len(calls[2]["messages"]), 1)
            self.assertIn("FINAL_RANKING_REVIEW", calls[2]["messages"][0]["content"][0]["text"])

    def test_binary_ndcg(self) -> None:
        self.assertEqual(ndcg_at(["a", "b"], ["a"]), 1.0)
        self.assertEqual(ndcg_at([], ["a"]), 0.0)
        self.assertIsNone(ndcg_at(["a"], []))


if __name__ == "__main__":
    unittest.main()
