"""
RAG 检索质量评估脚本
指标：Recall@K、MRR、Rerank 提升对比

问答对格式（JSON 数组或 JSONL，每条字段）：
  question/query : 问题
  gold_answer : 标准答案（原文片段，用于子串匹配命中 chunk）
                或 evidence/answer : 证据原文（兼容字段名）
  paper       : 所属文档标识（可选，也可从 source 字段提取）
  source_parent_title/source_chunk_title : 来源章节/切片标题（可选，用于分组显示）
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
import contextlib
import io
import json
import os
import re
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


def _normalize_for_match(text: str) -> str:
    """归一化空白和大小写，降低 PDF/Markdown 切分带来的格式差异。"""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _gold_anchors(gold_answer: str) -> List[str]:
    """
    从证据文本中生成多个锚点。
    长锚点更精确，短锚点用于兼容 chunk overlap、换行和 LaTeX 转写差异。
    """
    normalized = _normalize_for_match(gold_answer)
    if not normalized:
        return []
    anchors = [normalized[:300], normalized[:180], normalized[:100]]
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    anchors.extend(sentence for sentence in sentences[:2] if len(sentence) >= 50)
    return list(dict.fromkeys(anchor for anchor in anchors if len(anchor) >= 50))


def _chunk_entity(chunk: Dict) -> Dict[str, Any]:
    entity = chunk.get("entity") if isinstance(chunk, dict) else {}
    return entity if isinstance(entity, dict) else {}


def _chunk_content(chunk: Dict) -> str:
    entity = _chunk_entity(chunk)
    return entity.get("content", "") or chunk.get("content", "") or chunk.get("text", "")


def _chunk_source(chunk: Dict) -> str:
    entity = _chunk_entity(chunk)
    source = chunk.get("source") or entity.get("source") or ""
    if source:
        return str(source).lower()
    if chunk.get("url") or entity.get("url"):
        return "web"
    if _chunk_content(chunk):
        return "local"
    return "other"


def _chunk_paper_title(chunk: Dict) -> str:
    entity = _chunk_entity(chunk)
    return (
        entity.get("paper_title")
        or chunk.get("paper_title")
        or entity.get("file_title")
        or chunk.get("file_title")
        or entity.get("item_name")
        or chunk.get("item_name")
        or ""
    )


def _chunk_title(chunk: Dict) -> str:
    entity = _chunk_entity(chunk)
    return (
        entity.get("title")
        or chunk.get("title")
        or entity.get("parent_title")
        or chunk.get("parent_title")
        or ""
    )


def _title_matches(expected: str, actual: str) -> bool:
    expected_norm = _normalize_for_match(expected).lstrip("# ").strip()
    actual_norm = _normalize_for_match(actual).lstrip("# ").strip()
    if not expected_norm or not actual_norm:
        return False
    return expected_norm in actual_norm or actual_norm in expected_norm


def _paper_title_matches(expected: str, actual: str) -> bool:
    expected_norm = _normalize_for_match(expected)
    actual_norm = _normalize_for_match(actual)
    if not expected_norm:
        return True
    if not actual_norm:
        return False
    return expected_norm in actual_norm or actual_norm in expected_norm


def _chunk_title_hit(chunk: Dict, gold_chunk_title: str = "", gold_parent_title: str = "") -> bool:
    if not gold_chunk_title and not gold_parent_title:
        return False
    entity = _chunk_entity(chunk)
    candidate_titles = [
        _chunk_title(chunk),
        entity.get("parent_title") or chunk.get("parent_title") or "",
    ]
    gold_titles = [gold_chunk_title, gold_parent_title]
    return any(
        _title_matches(gold_title, candidate_title)
        for gold_title in gold_titles
        for candidate_title in candidate_titles
    )


def gold_hit(
    gold_answer: str,
    retrieved_chunks: List[Dict],
    gold_paper_title: str = "",
    gold_chunk_title: str = "",
    gold_parent_title: str = "",
    require_local: bool = False,
) -> int:
    """
    判断 gold_answer 是否命中检索结果。
    用 evidence/gold_answer 生成多个归一化锚点，在 chunk content 里做子串匹配。
    返回命中位置（1-based），未命中返回 -1。
    """
    anchors = _gold_anchors(gold_answer)
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if require_local and _chunk_source(chunk) != "local":
            continue
        if not _paper_title_matches(gold_paper_title, _chunk_paper_title(chunk)):
            continue
        if _chunk_title_hit(chunk, gold_chunk_title, gold_parent_title):
            return rank
        content = _chunk_content(chunk)
        normalized_content = _normalize_for_match(content)
        if any(anchor in normalized_content for anchor in anchors):
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
        output_fields=[
            "chunk_id",
            "content",
            "paper_title",
            "file_title",
            "title",
            "parent_title",
        ],
    )
    return res[0] if res else []


def _run_online_rerank(
    question: str,
    retrieved: List[Dict],
    idx: int,
    use_web: bool = True,
    pre_topk_local_guard: bool = False,
) -> List[Dict]:
    """
    严格复用线上节点：可选调用 Web MCP，再调用 node_rerank。
    retrieved 作为线上 RRF 后的 rrf_chunks 输入，web_search_docs 由线上 web 节点产生。
    """
    from app.query_process.nodes.node_rerank import (
        node_rerank,
        step_1_merge_docs,
        step_2_rerank_docs,
        step_3_topk,
    )
    from app.query_process.nodes.node_web_search_mcp import node_web_search_mcp

    state = {
        "session_id": f"eval_online_{idx}",
        "is_stream": False,
        "original_query": question,
        "rewritten_query": question,
        "rrf_chunks": retrieved,
        "web_search_docs": [],
    }

    if use_web:
        web_state = node_web_search_mcp(state)
        state.update(web_state or {})

    with contextlib.redirect_stdout(io.StringIO()):
        if pre_topk_local_guard:
            doc_items = step_1_merge_docs(state)
            scored_docs = step_2_rerank_docs(state, doc_items)
            guarded_docs = _apply_local_guard(scored_docs)
            return step_3_topk(guarded_docs)

        rerank_state = node_rerank(state)
        return rerank_state.get("reranked_docs") or []


def _paper_titles_for_online_eval(
    qa: Dict[str, Any],
    question: str,
    filter_paper_title: Optional[str],
) -> List[str]:
    title_aliases = {
        "TACT": "Test-Time Adaptation by Causal Trimming",
    }

    def resolve_title(title: str) -> str:
        return title_aliases.get(title, title)

    if filter_paper_title:
        return [resolve_title(filter_paper_title)]

    paper = (qa.get("paper") or qa.get("source") or "").strip()
    if not paper or paper.lower() == "unknown":
        return []

    if paper.lower() in {"cross-paper", "cross paper", "cross_paper"}:
        text = f"{question} {qa.get('answer') or qa.get('gold_answer') or ''}".lower()
        titles = []
        for candidate in ("TACT", "MonoDETR"):
            if candidate.lower() in text:
                titles.append(resolve_title(candidate))
        return titles or [resolve_title("TACT"), resolve_title("MonoDETR")]

    return [resolve_title(paper)]


def _run_online_retrieval_pipeline(
    question: str,
    qa: Dict[str, Any],
    idx: int,
    filter_paper_title: Optional[str],
    use_web: bool,
) -> Dict[str, Any]:
    """
    复用线上检索链路：
    embedding top20 + HyDE top20 -> RRF top30，并可携带 Web 结果给 rerank。
    """
    from app.query_process.nodes.node_search_embedding import node_search_embedding
    from app.query_process.nodes.node_search_embedding_hyde import node_search_embedding_hyde
    from app.query_process.nodes.node_web_search_mcp import node_web_search_mcp
    from app.query_process.nodes.node_rrf import node_rrf

    state: Dict[str, Any] = {
        "session_id": f"eval_online_pipeline_{idx}",
        "user_id": "eval_user",
        "is_stream": False,
        "original_query": question,
        "rewritten_query": question,
        "paper_titles": _paper_titles_for_online_eval(qa, question, filter_paper_title),
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "rrf_chunks": [],
    }

    with contextlib.redirect_stdout(io.StringIO()):
        embedding_state = node_search_embedding(state)
        state.update(embedding_state or {})

        hyde_state = node_search_embedding_hyde(state)
        state.update(hyde_state or {})

        if use_web:
            web_state = node_web_search_mcp(state)
            state.update(web_state or {})

        rrf_state = node_rrf(state)
        state.update(rrf_state or {})

    return state


def _rerank_online_state(
    state: Dict[str, Any],
    question: str,
    local_guard: bool,
    local_original_weight: Optional[float] = None,
    local_rerank_weight: Optional[float] = None,
    no_source_protection: bool = False,
) -> List[Dict]:
    from app.query_process.nodes import node_rerank as rerank_node

    if no_source_protection:
        return _rerank_online_state_no_source_protection(state, question, rerank_node)

    old_local_original_weight = rerank_node.LOCAL_ORIGINAL_WEIGHT
    old_local_rerank_weight = rerank_node.LOCAL_RERANK_WEIGHT

    if local_original_weight is not None:
        rerank_node.LOCAL_ORIGINAL_WEIGHT = local_original_weight
    if local_rerank_weight is not None:
        rerank_node.LOCAL_RERANK_WEIGHT = local_rerank_weight

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if local_guard:
                doc_items = rerank_node.step_1_merge_docs(state)
                scored_docs = rerank_node.step_2_rerank_docs(state, doc_items)
                guarded_docs = _apply_local_guard(scored_docs)
                return rerank_node.step_3_topk(guarded_docs)

            rerank_state = rerank_node.node_rerank(state)
            return rerank_state.get("reranked_docs") or []
    finally:
        rerank_node.LOCAL_ORIGINAL_WEIGHT = old_local_original_weight
        rerank_node.LOCAL_RERANK_WEIGHT = old_local_rerank_weight


def _rerank_online_state_no_source_protection(
    state: Dict[str, Any],
    question: str,
    rerank_node,
) -> List[Dict]:
    """
    Eval-only baseline: local/web are mixed by one reranker score only.
    This intentionally disables local source protection and source weights.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        doc_items = rerank_node.step_1_merge_docs(state)

    if not doc_items or not question:
        return []

    reranker = get_reranker_model()
    max_chars = getattr(rerank_node, "RERANK_TEXT_MAX_CHARS", 700)
    texts = [
        (f"{d.get('title', '')}: {d.get('text', '')}" if d.get("title") else d.get("text", ""))[:max_chars]
        for d in doc_items
    ]
    pairs = [[question, text] for text in texts]
    raw_scores = reranker.compute_score(pairs, max_length=256, batch_size=8)
    norm_scores = rerank_node._normalize_scores(raw_scores)

    scored_docs = []
    for item, text, raw_score, norm_score in zip(doc_items, texts, raw_scores, norm_scores):
        scored_item = item.copy()
        scored_item.update(
            {
                "text": item.get("text") or "",
                "content": item.get("content") or item.get("text") or "",
                "score": norm_score,
                "raw_rerank_score": float(raw_score),
                "rerank_score": norm_score,
                "rerank_policy": "no_source_protection",
                "source": item.get("source") or "",
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "url": item.get("url") or "",
                "title": item.get("title") or "",
                "rerank_text": text,
            }
        )
        scored_docs.append(scored_item)

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return rerank_node.step_3_topk(scored_docs)


def _source_counts(docs: List[Dict], k: int) -> Dict[str, int]:
    counts = {"local": 0, "web": 0, "other": 0}
    for doc in docs[:k]:
        source = doc.get("source") or "other"
        counts[source if source in counts else "other"] += 1
    return counts


def _apply_local_guard(docs: List[Dict]) -> List[Dict]:
    """
    评估用策略：
    local 作为主证据，web 作为补充。
      Top1 保 local，Top3 至少 2 条 local，Top5 至少 3 条 local。
    """
    local_docs = [doc for doc in docs if doc.get("source") == "local"]
    web_docs = [doc for doc in docs if doc.get("source") == "web"]
    other_docs = [doc for doc in docs if doc.get("source") not in ("local", "web")]

    if len(local_docs) < 3:
        return docs

    guarded = []
    guarded.extend(local_docs[:2])
    guarded.extend(web_docs[:1])
    guarded.extend(local_docs[2:3])
    guarded.extend(web_docs[1:2])
    guarded.extend(local_docs[3:])
    guarded.extend(web_docs[2:])
    guarded.extend(other_docs)

    seen = set()
    deduped = []
    for doc in guarded:
        key = doc.get("chunk_id") or doc.get("url") or id(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _chunk_rerank_text(chunk: Dict, max_chars: Optional[int]) -> str:
    entity = chunk.get("entity", {}) or {}
    title = entity.get("title") or entity.get("paper_title") or chunk.get("title") or chunk.get("paper_title") or ""
    content = entity.get("content", "") or chunk.get("content", "")
    text = f"{title}: {content}" if title else content
    return text[:max_chars] if max_chars and max_chars > 0 else text


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    max_chars: Optional[int] = 1500,
    fusion: bool = False,
    fusion_mode: str = "rrf",
    original_weight: float = 0.7,
    rerank_weight: float = 0.3,
    local_weight: float = 2.5,
    rrf_k: int = 60,
) -> List[Dict]:
    """用 BGE-Reranker 对检索结果重排序"""
    if not chunks:
        return []
    reranker = get_reranker_model()
    texts = [_chunk_rerank_text(chunk, max_chars) for chunk in chunks]
    pairs = [[query, t] for t in texts]
    scores = reranker.compute_score(pairs, max_length=256, batch_size=8)

    if fusion and fusion_mode == "score":
        raw_scores = [float(s) for s in scores]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score
        fused = []
        total = len(chunks)
        for original_index, (chunk, score) in enumerate(zip(chunks, raw_scores), start=1):
            original_rank_score = (total - original_index + 1) / total
            rerank_score = (score - min_score) / score_range if score_range > 1e-9 else 1.0
            final_score = (
                original_weight * original_rank_score
                + rerank_weight * rerank_score
            ) * local_weight
            fused.append(
                {
                    "chunk": chunk,
                    "original_rank_score": original_rank_score,
                    "rerank_score": rerank_score,
                    "raw_rerank_score": score,
                    "final_score": final_score,
                }
            )
        fused.sort(key=lambda x: x["final_score"], reverse=True)
        return [item["chunk"] for item in fused]

    if fusion:
        rerank_order = sorted(
            enumerate(scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        rerank_rank_by_index = {
            original_index: rerank_rank
            for rerank_rank, (original_index, _) in enumerate(rerank_order, start=1)
        }
        fused = []
        for original_index, (chunk, score) in enumerate(zip(chunks, scores), start=1):
            rerank_rank = rerank_rank_by_index[original_index - 1]
            final_score = (
                original_weight / (rrf_k + original_index)
                + rerank_weight / (rrf_k + rerank_rank)
            )
            fused.append(
                {
                    "chunk": chunk,
                    "rerank_score": float(score),
                    "original_rank": original_index,
                    "rerank_rank": rerank_rank,
                    "final_score": final_score,
                }
            )
        fused.sort(key=lambda x: x["final_score"], reverse=True)
        return [item["chunk"] for item in fused]

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
    rerank_max_chars: Optional[int],
    rerank_fusion: bool,
    rerank_fusion_mode: str,
    original_weight: float,
    rerank_weight: float,
    online_rerank: bool,
    online_pipeline: bool,
    online_use_web: bool,
    local_guard: bool,
    online_local_original_weight: Optional[float],
    online_local_rerank_weight: Optional[float],
    online_no_source_protection: bool,
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
    if online_pipeline:
        mode += " + online_pipeline(embedding20+HyDE20+RRF30)"
        if online_no_source_protection:
            mode += " + no_source_protection"
        if online_local_original_weight is not None or online_local_rerank_weight is not None:
            mode += (
                " + local_weights="
                f"{online_local_original_weight if online_local_original_weight is not None else 'default'}"
                "/"
                f"{online_local_rerank_weight if online_local_rerank_weight is not None else 'default'}"
            )
    if online_rerank:
        mode += " + 线上node_rerank"
        if online_use_web:
            mode += " + Web"
    print(f"\n开始评估，共 {total} 条问答对，模式：{mode}\n" + "─" * 60)

    for idx, qa in enumerate(qa_pairs, start=1):
        question    = (qa.get("question") or qa.get("query") or "").strip()
        # 兼容 evidence、gold_answer、answer 三种字段名
        gold_answer = qa.get("gold_answer") or qa.get("evidence") or qa.get("answer") or ""
        gold_answer = gold_answer.strip()
        # 从 source 字段提取 paper 信息，或使用章节标题，或默认为 unknown
        gold_paper_title = (qa.get("paper_title") or qa.get("paper") or qa.get("source") or "").strip()
        if gold_paper_title.lower() == "unknown":
            gold_paper_title = ""
        paper       = gold_paper_title or qa.get("source_parent_title") or "unknown"
        qa_type     = qa.get("type", "")
        difficulty  = qa.get("difficulty", "")
        chunk_title = qa.get("source_chunk_title", "")
        parent_title = qa.get("source_parent_title", "")
        require_local_hit = bool(gold_paper_title)

        # 构建显示标签
        labels = [paper] if paper != "unknown" else []
        if chunk_title:
            labels.append(chunk_title)
        if qa_type:
            labels.append(qa_type)
        if difficulty:
            labels.append(difficulty)
        label_str = "/".join(labels) if labels else "无标签"

        print(f"[{idx}/{total}] Q: {question[:60]}  ({label_str})")
        paper_totals[paper] += 1

        # ── 检索 ──
        online_state = None
        if online_pipeline:
            online_state = _run_online_retrieval_pipeline(
                question=question,
                qa=qa,
                idx=idx,
                filter_paper_title=filter_paper_title,
                use_web=online_use_web,
            )
            retrieved = online_state.get("rrf_chunks") or []
            print(
                "  online retrieval: "
                f"paper_titles={online_state.get('paper_titles')}, "
                f"embedding={len(online_state.get('embedding_chunks') or [])}, "
                f"hyde={len(online_state.get('hyde_embedding_chunks') or [])}, "
                f"rrf={len(retrieved)}, "
                f"web={len(online_state.get('web_search_docs') or [])}"
            )
        else:
            retrieved = search_top_k(question, limit=retrieve_limit, filter_paper_title=filter_paper_title)

        if not retrieved:
            print("  WARNING: 检索结果为空！")
            continue

        # ── Before Rerank ──
        hit_before = gold_hit(
            gold_answer,
            retrieved[:max_k],
            gold_paper_title=gold_paper_title,
            gold_chunk_title=chunk_title,
            gold_parent_title=parent_title,
            require_local=require_local_hit,
        )
        _update_counters(g_before, hit_before, topk_list)
        _update_counters(by_paper_before[paper], hit_before, topk_list)
        print(f"  before rerank 命中位置: {hit_before if hit_before > 0 else '未命中'}")

        # 如果未命中，打印检索到的前3个chunk的部分内容用于调试
        if hit_before < 0:
            print(f"  Gold anchor (前50字): {gold_answer[:50]}")
            print(f"  检索到的前3个chunk片段:")
            for i, chunk in enumerate(retrieved[:3], 1):
                content = chunk.get("entity", {}).get("content", "") or chunk.get("content", "")
                paper_title = _chunk_paper_title(chunk)
                print(f"     [{i}] {_chunk_source(chunk)} | {paper_title} | {content[:80]}...")

        # ── After Rerank ──
        if use_rerank:
            if online_pipeline and online_state is not None:
                reranked = _rerank_online_state(
                    state=online_state,
                    question=question,
                    local_guard=local_guard,
                    local_original_weight=online_local_original_weight,
                    local_rerank_weight=online_local_rerank_weight,
                    no_source_protection=online_no_source_protection,
                )
                counts = _source_counts(reranked, max_k)
                print(
                    f"  online top{max_k} æ¥æº: local={counts['local']}, web={counts['web']}, other={counts['other']}"
                )
            elif online_rerank:
                reranked = _run_online_rerank(
                    question=question,
                    retrieved=retrieved,
                    idx=idx,
                    use_web=online_use_web,
                    pre_topk_local_guard=local_guard,
                )
                if False:
                    reranked = _apply_local_guard(reranked)
                counts = _source_counts(reranked, max_k)
                print(
                    f"  online top{max_k} 来源: local={counts['local']}, web={counts['web']}, other={counts['other']}"
                )
            else:
                reranked = rerank_chunks(
                    question,
                    retrieved[:max_k],
                    max_chars=rerank_max_chars,
                    fusion=rerank_fusion,
                    fusion_mode=rerank_fusion_mode,
                    original_weight=original_weight,
                    rerank_weight=rerank_weight,
                )
            hit_after = gold_hit(
                gold_answer,
                reranked,
                gold_paper_title=gold_paper_title,
                gold_chunk_title=chunk_title,
                gold_parent_title=parent_title,
                require_local=require_local_hit,
            )
            _update_counters(g_after, hit_after, topk_list)
            _update_counters(by_paper_after[paper], hit_after, topk_list)
            print(f"  after  rerank 命中位置: {hit_after if hit_after > 0 else '未命中'}")

            # 如果 rerank 导致排名下降，输出详细信息
            if hit_before > 0 and (hit_after < 0 or hit_after > hit_before):
                print(f"  WARNING: Rerank 导致排名下降: {hit_before} -> {hit_after if hit_after > 0 else '未命中'}")
                print(f"     Gold anchor (前30字): {gold_answer[:30]}")

    # ── 汇总输出 ──
    print("\n" + "═" * 60)
    print("评估结果汇总")
    print("═" * 60)

    result: Dict[str, Any] = {
        "total": total,
        "mode": mode,
        "hit_policy": "local source + matching paper_title + answer anchor or source_chunk_title/source_parent_title",
    }

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
    parser.add_argument(
        "--rerank_max_chars",
        type=int,
        default=1500,
        help="Rerank 输入文本最大字符数；<=0 表示不截断",
    )
    parser.add_argument(
        "--rerank_fusion",
        action="store_true",
        help="使用原始召回排名与 rerank 排名融合，而不是纯 rerank 排序",
    )
    parser.add_argument(
        "--rerank_fusion_mode",
        choices=["rrf", "score"],
        default="rrf",
        help="融合方式：rrf 使用排名融合；score 使用 0-1 归一化分数融合",
    )
    parser.add_argument("--original_weight", type=float, default=0.7, help="融合排序中的原始召回排名权重")
    parser.add_argument("--rerank_weight", type=float, default=0.3, help="融合排序中的 rerank 排名权重")
    parser.add_argument(
        "--online_pipeline",
        action="store_true",
        help="reuse online retrieval: embedding top20 + HyDE top20 + RRF top30 + node_rerank",
    )
    parser.add_argument(
        "--online_local_original_weight",
        type=float,
        default=None,
        help="eval-only override for node_rerank.LOCAL_ORIGINAL_WEIGHT",
    )
    parser.add_argument(
        "--online_local_rerank_weight",
        type=float,
        default=None,
        help="eval-only override for node_rerank.LOCAL_RERANK_WEIGHT",
    )
    parser.add_argument(
        "--online_no_source_protection",
        action="store_true",
        help="eval-only baseline: mix local and web by reranker score without source protection",
    )
    parser.add_argument(
        "--online_rerank",
        action="store_true",
        help="严格调用线上 node_web_search_mcp + node_rerank 评估混合重排",
    )
    parser.add_argument(
        "--no_online_web",
        action="store_true",
        help="online_rerank 模式下不调用 web 搜索，只调用线上 node_rerank",
    )
    parser.add_argument(
        "--local_guard",
        action="store_true",
        help="online_rerank 后应用 local 保护策略：内部论文问题 local 优先，web 作为补充",
    )
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
        rerank_max_chars=args.rerank_max_chars if args.rerank_max_chars > 0 else None,
        rerank_fusion=args.rerank_fusion,
        rerank_fusion_mode=args.rerank_fusion_mode,
        original_weight=args.original_weight,
        rerank_weight=args.rerank_weight,
        online_rerank=args.online_rerank,
        online_pipeline=args.online_pipeline,
        online_use_web=not args.no_online_web,
        local_guard=args.local_guard,
        online_local_original_weight=args.online_local_original_weight,
        online_local_rerank_weight=args.online_local_rerank_weight,
        online_no_source_protection=args.online_no_source_protection,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
