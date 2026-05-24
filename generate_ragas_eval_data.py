import argparse
import contextlib
import io
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from eval_rag import load_qa_pairs, search_top_k
from app.query_process.nodes.node_rerank import node_rerank
from app.query_process.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.nodes.node_answer_output import (
    step_2_construct_prompt,
    step_3_generate_response,
)


DEFAULT_PAPER_TITLE = "Test-Time Adaptation by Causal Trimming"


def _doc_text(doc: Dict[str, Any]) -> str:
    text = doc.get("content") or doc.get("text") or ""
    entity = doc.get("entity") or {}
    if not text and isinstance(entity, dict):
        text = entity.get("content") or ""
    return (text or "").strip()


def _build_contexts(docs: List[Dict[str, Any]], max_contexts: int) -> List[str]:
    contexts = []
    for doc in docs[:max_contexts]:
        text = _doc_text(doc)
        if text:
            contexts.append(text)
    return contexts


def _retrieve_and_rerank(
    question: str,
    retrieve_limit: int,
    use_web: bool,
    idx: int,
) -> List[Dict[str, Any]]:
    retrieved = search_top_k(question, limit=retrieve_limit)
    state = {
        "session_id": f"ragas_eval_retrieval_{idx}",
        "user_id": "ragas_eval",
        "is_stream": False,
        "original_query": question,
        "rewritten_query": question,
        "rrf_chunks": retrieved,
        "web_search_docs": [],
    }
    if use_web:
        state.update(node_web_search_mcp(state) or {})

    with contextlib.redirect_stdout(io.StringIO()):
        result = node_rerank(state)
    return result.get("reranked_docs") or []


def _generate_answer(question: str, reranked_docs: List[Dict[str, Any]]) -> str:
    state = {
        "session_id": f"ragas_eval_answer_{uuid.uuid4()}",
        "user_id": "ragas_eval",
        "is_stream": False,
        "original_query": question,
        "rewritten_query": question,
        "paper_titles": [DEFAULT_PAPER_TITLE],
        "history": [],
        "reranked_docs": reranked_docs,
    }
    prompt = step_2_construct_prompt(state)
    state["prompt"] = prompt
    step_3_generate_response(state, prompt)
    return (state.get("answer") or "").strip()


def generate_dataset(
    qa_file: str,
    output: str,
    retrieve_limit: int,
    max_contexts: int,
    use_web: bool,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    qa_pairs = load_qa_pairs(qa_file)
    if limit and limit > 0:
        qa_pairs = qa_pairs[:limit]

    dataset = []
    for idx, qa in enumerate(qa_pairs, start=1):
        question = (qa.get("question") or qa.get("query") or "").strip()
        ground_truth = (
            qa.get("ground_truth")
            or qa.get("gold_answer")
            or qa.get("evidence")
            or qa.get("answer")
            or ""
        ).strip()
        if not question:
            continue

        print(f"[{idx}/{len(qa_pairs)}] {question[:80]}")
        reranked_docs = _retrieve_and_rerank(
            question=question,
            retrieve_limit=retrieve_limit,
            use_web=use_web,
            idx=idx,
        )
        answer = _generate_answer(question, reranked_docs)
        contexts = _build_contexts(reranked_docs, max_contexts)

        dataset.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

        Path(output).write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Ragas eval dataset from eval.json")
    parser.add_argument("--qa_file", default="eval.json")
    parser.add_argument("--output", default="ragas_eval_dataset.json")
    parser.add_argument("--retrieve_limit", type=int, default=20)
    parser.add_argument("--max_contexts", type=int, default=10)
    parser.add_argument("--no_web", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only generate first N examples")
    args = parser.parse_args()

    dataset = generate_dataset(
        qa_file=args.qa_file,
        output=args.output,
        retrieve_limit=args.retrieve_limit,
        max_contexts=args.max_contexts,
        use_web=not args.no_web,
        limit=args.limit,
    )
    print(f"saved {len(dataset)} examples to {args.output}")


if __name__ == "__main__":
    main()
