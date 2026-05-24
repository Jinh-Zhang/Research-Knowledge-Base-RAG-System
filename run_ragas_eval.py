import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness


load_dotenv()


DEFAULT_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
]

ALL_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def _load_ragas_records(input_path: str, limit: int = 0) -> List[Dict[str, Any]]:
    raw_items = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if limit and limit > 0:
        raw_items = raw_items[:limit]

    records = []
    for item in raw_items:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        ground_truth = (item.get("ground_truth") or "").strip()
        contexts = item.get("contexts") or []
        contexts = [str(context).strip() for context in contexts if str(context).strip()]

        if not question or not answer or not ground_truth:
            continue

        records.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            }
        )
    return records


def _build_metrics(metric_names: List[str]):
    if metric_names == ["all"]:
        metric_names = ALL_METRICS

    metric_map = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "answer_correctness": AnswerCorrectness,
    }

    metrics = []
    for name in metric_names:
        if name not in metric_map:
            raise ValueError(f"未知 Ragas 指标: {name}")
        metrics.append(metric_map[name]())
    return metrics


def _build_llm(model: str, temperature: float):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("请先在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        extra_body={"enable_thinking": False},
    )


def _build_embeddings(model: str):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("请先在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL")

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def _save_result(result, output_path: str, summary_path: str):
    df = result.to_pandas()
    Path(output_path).write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    numeric_cols = [
        col
        for col in df.columns
        if col
        not in {
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        }
        and str(df[col].dtype) != "object"
    ]
    summary = {
        "count": int(len(df)),
        "metrics": {
            col: (
                None
                if math.isnan(float(df[col].mean()))
                else float(df[col].mean())
            )
            for col in numeric_cols
        },
    }
    Path(summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on generated RAG data")
    parser.add_argument("--input", default="ragas_eval_dataset.json")
    parser.add_argument("--output", default="ragas_eval_result.json")
    parser.add_argument("--summary", default="ragas_eval_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--llm_model", default=os.getenv("LLM_DEFAULT_MODEL") or "qwen3.5-flash")
    parser.add_argument("--embedding_model", default=os.getenv("RAGAS_EMBEDDING_MODEL") or "text-embedding-v4")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"默认跑稳定指标: {' '.join(DEFAULT_METRICS)}；可传 all 尝试全部指标。",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    records = _load_ragas_records(args.input, args.limit)
    if not records:
        raise ValueError(f"没有可评估的数据: {args.input}")

    dataset = Dataset.from_list(records)
    llm = _build_llm(args.llm_model, args.temperature)
    embeddings = _build_embeddings(args.embedding_model)
    metrics = _build_metrics(args.metrics)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        batch_size=args.batch_size,
    )
    summary = _save_result(result, args.output, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
