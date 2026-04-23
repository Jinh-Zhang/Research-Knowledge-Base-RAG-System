"""
RAG 检索质量评估脚本
指标：Recall@K、MRR、Rerank 提升对比

问答对格式（JSON 数组或 JSONL，每条字段）：
  question    : 问题
  gold_answer : 标准答案（原文片段，用于子串匹配命中 chunk）
                或 evidence : 证据原文（兼容字段名）
  paper       : 所属文档标识（可选，也可从 source 字段提取）
  type        : 问题类型（可选，用于分组统计）
  difficulty  : 难度（可选，用于分组统计）

用法：
  # 端到端评估（不过滤，测真实检索能力）
  python eval_rag.py --qa_file eval_qa.json --topk 1 3 5

  # 仅测某篇文章范围内的检索（加过滤，相当于假设 paper_title 识别已正确）
  python eval_rag.py --qa_file eval_qa.json --filter_paper_title "开-TACT"

  # 跳过 Rerank，只测向量检索阶段
  python eval_rag.py --qa_file eval_qa.json --no_rerank

  # 保存结果
  python eval_rag.py --qa_file eval_qa.json --output result.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    hybrid_search,
    get_milvus_client,
)
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def load_qa_pairs(qa_file: str) -> List[Dict[str, Any]]:
    """支持 JSONL（每行一个对象）和 JSON 数组两种格式"""
    with open(qa_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def gold_hit(gold_answer: str, retrieved_chunks: List[Dict]) -> int:
    """
    判断 gold_answer 是否命中检索结果。
    取 gold_answer 前 30 个字符作为锚，在 chunk content 里做子串匹配。
    返回命中位置（1-based），未命中返回 -1。
    """
    anchor = gold_answer.strip()[:300]
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        content = chunk.get("entity", {}).get("content", "") or chunk.get("content", "")
        if anchor in content:
            return rank
    return -1


def search_top_k(
    query: str,
    limit: int = 10,
    filter_paper_title: Optional[str] = None,
) -> List[Dict]:
    """执行混合检索，返回原始结果列表。filter_paper_title 为空则不加过滤。"""
    embeddings = generate_embeddings([query])
    dense_vec = embeddings["dense"][0]
    sparse_vec = embeddings["sparse"][0]

    collection_name = os.environ.get("CHUNKS_COLLECTION", "paper_chunks")
    client = get_milvus_client()

    expr = None
    if filter_paper_title:
        escaped = filter_paper_title.replace('"', '\\"')
        expr = f'paper_title == "{escaped}"'

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=limit,
    )
    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id", "content", "paper_title", "title"],
    )
    return res[0] if res else []


def rerank_chunks(query: str, chunks: List[Dict], max_chars: int = 700) -> List[Dict]:
    """用 BGE-Reranker 对检索结果重排序"""
    if not chunks:
        return []
    reranker = get_reranker_model()
    texts = [
        (chunk.get("entity", {}).get("content", "") or chunk.get("content", ""))[:max_chars]
        for chunk in chunks
    ]
    pairs = [[query, t] for t in texts]
    scores = reranker.compute_score(pairs, max_length=256, batch_size=8)
    scored = sorted(
        [{"chunk": c, "rerank_score": float(s)} for c, s in zip(chunks, scores)],
        key=lambda x: x["rerank_score"],
        reverse=True,
    )
    return [item["chunk"] for item in scored]


# ──────────────────────────────────────────────
# 指标计算辅助
# ──────────────────────────────────────────────

def _empty_counters(topk_list):
    return {"hits": {k: 0 for k in topk_list}, "rr": []}


def _update_counters(counters, hit_rank, topk_list):
    counters["rr"].append(1.0 / hit_rank if hit_rank > 0 else 0.0)
    for k in topk_list:
        if 0 < hit_rank <= k:
            counters["hits"][k] += 1


def _calc_metrics(counters, total, topk_list):
    metrics = {}
    for k in topk_list:
        metrics[f"recall@{k}"] = counters["hits"][k] / total if total else 0.0
    metrics["mrr"] = sum(counters["rr"]) / total if total else 0.0
    return metrics


def _print_metrics(label, metrics, topk_list, total):
    print(f"\n【{label}】  (n={total})")
    for k in topk_list:
        v = metrics[f"recall@{k}"]
        hits = round(v * total)
        print(f"  Recall@{k}: {hits}/{total} = {v:.3f}")
    print(f"  MRR      : {metrics['mrr']:.3f}")


# ──────────────────────────────────────────────
# 核心评估逻辑
# ──────────────────────────────────────────────

def evaluate(
    qa_pairs: List[Dict],
    topk_list: List[int],
    use_rerank: bool,
    retrieve_limit: int,
    filter_paper_title: Optional[str],
) -> Dict[str, Any]:
    max_k = max(topk_list)

    # 全局计数器
    g_before = _empty_counters(topk_list)
    g_after  = _empty_counters(topk_list)

    # 按 paper 分组计数器
    by_paper_before: Dict[str, Any] = defaultdict(lambda: _empty_counters(topk_list))
    by_paper_after:  Dict[str, Any] = defaultdict(lambda: _empty_counters(topk_list))
    paper_totals: Dict[str, int] = defaultdict(int)

    total = len(qa_pairs)
    mode = f"过滤 paper_title={filter_paper_title!r}" if filter_paper_title else "不过滤（端到端）"
    print(f"\n开始评估，共 {total} 条问答对，模式：{mode}\n" + "─" * 60)

    for idx, qa in enumerate(qa_pairs, start=1):
        question    = qa.get("question", "").strip()
        # 兼容 evidence 和 gold_answer 两种字段名
        gold_answer = qa.get("gold_answer") or qa.get("evidence", "")
        gold_answer = gold_answer.strip()
        # 从 source 字段提取 paper 信息，或使用 paper 字段，或默认为 unknown
        paper       = qa.get("paper") or qa.get("source", "unknown")
        qa_type     = qa.get("type", "")
        difficulty  = qa.get("difficulty", "")

        # 构建显示标签
        labels = [paper] if paper != "unknown" else []
        if qa_type:
            labels.append(qa_type)
        if difficulty:
            labels.append(difficulty)
        label_str = "/".join(labels) if labels else "无标签"

        print(f"[{idx}/{total}] Q: {question[:60]}  ({label_str})")
        paper_totals[paper] += 1

        # ── 检索 ──
        retrieved = search_top_k(question, limit=retrieve_limit, filter_paper_title=filter_paper_title)

        if not retrieved:
            print(f"  ⚠️  检索结果为空！")
            continue

        # ── Before Rerank ──
        hit_before = gold_hit(gold_answer, retrieved[:max_k])
        _update_counters(g_before, hit_before, topk_list)
        _update_counters(by_paper_before[paper], hit_before, topk_list)
        print(f"  before rerank 命中位置: {hit_before if hit_before > 0 else '未命中'}")

        # 如果未命中，打印检索到的前3个chunk的部分内容用于调试
        if hit_before < 0:
            print(f"  📝 Gold anchor (前50字): {gold_answer[:50]}")
            print(f"  🔍 检索到的前3个chunk片段:")
            for i, chunk in enumerate(retrieved[:3], 1):
                content = chunk.get("entity", {}).get("content", "") or chunk.get("content", "")
                paper_title = chunk.get("entity", {}).get("paper_title", "") or chunk.get("paper_title", "")
                print(f"     [{i}] {paper_title} | {content[:80]}...")

        # ── After Rerank ──
        if use_rerank:
            reranked = rerank_chunks(question, retrieved[:max_k])
            hit_after = gold_hit(gold_answer, reranked)
            _update_counters(g_after, hit_after, topk_list)
            _update_counters(by_paper_after[paper], hit_after, topk_list)
            print(f"  after  rerank 命中位置: {hit_after if hit_after > 0 else '未命中'}")

            # 如果 rerank 导致排名下降，输出详细信息
            if hit_before > 0 and (hit_after < 0 or hit_after > hit_before):
                print(f"  ⚠️  Rerank 导致排名下降: {hit_before} -> {hit_after if hit_after > 0 else '未命中'}")
                print(f"     Gold anchor (前30字): {gold_answer[:30]}")

    # ── 汇总输出 ──
    print("\n" + "═" * 60)
    print("评估结果汇总")
    print("═" * 60)

    result: Dict[str, Any] = {"total": total, "mode": mode}

    m_before = _calc_metrics(g_before, total, topk_list)
    _print_metrics("Before Rerank（全部）", m_before, topk_list, total)
    result["before"] = m_before

    if use_rerank:
        m_after = _calc_metrics(g_after, total, topk_list)
        _print_metrics("After Rerank（全部）", m_after, topk_list, total)
        result["after"] = m_after

        print("\n【Rerank 提升 delta（after - before）】")
        delta = {}
        for k in topk_list:
            d = m_after[f"recall@{k}"] - m_before[f"recall@{k}"]
            delta[f"recall@{k}"] = d
            print(f"  Recall@{k} delta: {d:+.3f}")
        d_mrr = m_after["mrr"] - m_before["mrr"]
        delta["mrr"] = d_mrr
        print(f"  MRR delta      : {d_mrr:+.3f}")
        result["delta"] = delta

    # ── 分 paper 分组统计 ──
    if len(paper_totals) > 1:
        print("\n【按 paper 分组】")
        result["by_paper"] = {}
        for paper, n in sorted(paper_totals.items()):
            mb = _calc_metrics(by_paper_before[paper], n, topk_list)
            _print_metrics(f"Before Rerank · {paper}", mb, topk_list, n)
            entry = {"total": n, "before": mb}
            if use_rerank:
                ma = _calc_metrics(by_paper_after[paper], n, topk_list)
                _print_metrics(f"After  Rerank · {paper}", ma, topk_list, n)
                entry["after"] = ma
            result["by_paper"][paper] = entry

    print("═" * 60)
    return result


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评估")
    parser.add_argument("--qa_file", required=True, help="问答对文件路径（JSONL 或 JSON 数组）")
    parser.add_argument(
        "--filter_paper_title", default="",
        help="若指定，检索时加 paper_title 过滤（测单篇文章范围内检索）；不指定则端到端评估（推荐）",
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5], help="评估的 K 值，如 1 3 5")
    parser.add_argument("--no_rerank", action="store_true", help="跳过 Rerank 阶段")
    parser.add_argument("--retrieve_limit", type=int, default=10, help="检索召回数量，建议 >= max(topk)")
    parser.add_argument("--output", default="", help="结果 JSON 输出路径，留空则不保存")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_file)
    if not qa_pairs:
        print("问答对文件为空，退出")
        sys.exit(1)

    result = evaluate(
        qa_pairs=qa_pairs,
        topk_list=sorted(args.topk),
        use_rerank=not args.no_rerank,
        retrieve_limit=args.retrieve_limit,
        filter_paper_title=args.filter_paper_title or None,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
