import sys
from app.utils.task_utils import *
from dotenv import load_dotenv
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
import time
load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 5
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.5
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 1.2
# 本地文档权重系数（>1表示本地权重更大）
LOCAL_WEIGHT: float = 2.5
# 本地文档内部排序：论文内部问题以 RRF/Milvus 原始排序为准。
LOCAL_ORIGINAL_WEIGHT: float = 1.0
LOCAL_RERANK_WEIGHT: float = 0.0
# 外部信息问题允许 rerank 对 local 也做轻量校正。
EXTERNAL_LOCAL_ORIGINAL_WEIGHT: float = 0.85
EXTERNAL_LOCAL_RERANK_WEIGHT: float = 0.15
# 联网结果默认仅作为补充来源，避免 title/snippet 因关键词密度高而压过本地论文证据。
WEB_WEIGHT: float = 1.0
# 外部信息问题中，web 可正常参与竞争。
EXTERNAL_WEB_WEIGHT: float = 2.5
# Rerank 输入最大字符数。Cross-encoder 耗时较高，保留截断以控制延迟。
RERANK_TEXT_MAX_CHARS: int = 700
# local 检索强度阈值。分数来自 Milvus/RRF 上游，无法取到时默认信任 local 排名。
LOCAL_STRONG_SCORE: float = 0.55
LOCAL_MEDIUM_SCORE: float = 0.35


def _copy_local_metadata(entity, target):
    for field in (
        "paper_title",
        "file_title",
        "parent_title",
        "chunk_type",
        "citations",
        "citation_refs",
        "fig_refs",
        "figures",
        "table_refs",
        "tables",
    ):
        if field in entity:
            target[field] = entity.get(field)
    return target


def _normalize_scores(scores):
    if not scores:
        return []
    values = [float(score) for score in scores]
    min_score = min(values)
    max_score = max(values)
    score_range = max_score - min_score
    if score_range <= 1e-9:
        return [1.0 for _ in values]
    return [(score - min_score) / score_range for score in values]


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _local_retrieval_strength(local_docs):
    """
    根据本地召回结果本身判断 local 是否足够可信。
    strong: local 结果充足且 top 分数较高，web 只补充。
    medium: local 有一定命中但不够确定，允许 rerank/web 参与。
    weak: local 少或分数低，web 可以进入前排。
    """
    if not local_docs:
        return "weak"

    scores = [
        _safe_float(doc.get("retrieval_score"))
        for doc in local_docs
        if _safe_float(doc.get("retrieval_score")) is not None
    ]

    if not scores:
        return "strong" if len(local_docs) >= 3 else "medium"

    top_score = max(scores)
    top3_scores = scores[:3]
    avg_top3 = sum(top3_scores) / len(top3_scores)

    if top_score >= LOCAL_STRONG_SCORE or avg_top3 >= LOCAL_MEDIUM_SCORE:
        return "strong"
    if top_score >= LOCAL_MEDIUM_SCORE or len(local_docs) >= 5:
        return "medium"
    return "weak"


def _choose_rerank_policy(local_docs, web_docs):
    local_strength = _local_retrieval_strength(local_docs)

    if not web_docs:
        policy = "local_only"
    elif not local_docs or local_strength == "weak":
        policy = "web_allowed"
    elif local_strength == "strong":
        policy = "local_protected"
    else:
        policy = "balanced"

    logger.info(
        f"Step 2: rerank策略={policy}, local_strength={local_strength}"
    )
    return policy


def _policy_weights(policy):
    if policy == "local_protected":
        return LOCAL_ORIGINAL_WEIGHT, LOCAL_RERANK_WEIGHT, WEB_WEIGHT
    if policy == "local_only":
        return LOCAL_ORIGINAL_WEIGHT, LOCAL_RERANK_WEIGHT, 0.0
    if policy == "web_allowed":
        return 0.60, 0.40, EXTERNAL_WEB_WEIGHT
    return (
        EXTERNAL_LOCAL_ORIGINAL_WEIGHT,
        EXTERNAL_LOCAL_RERANK_WEIGHT,
        EXTERNAL_WEB_WEIGHT,
    )


# Rerank节点（工作流入口）
def node_rerank(state):
    """
    Rerank节点
    对检索到的文档进行重新排序，提高相关性
    """
    logger.info("---Rerank---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # 阶段一：合并文档
    doc_items = step_1_merge_docs(state)
    # 阶段二：对文档进行重排序
    scored_docs = step_2_rerank_docs(state, doc_items)
    # 阶段三：动态 TopK
    topk_docs = step_3_topk(scored_docs)
    logger.info(
        f"Rerank 输出完成，最终保留 {len(topk_docs)} 条文档，来源分布="
        f"{[doc.get('source') for doc in topk_docs[:10]]}"
    )

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    return {"reranked_docs": topk_docs}


def step_1_merge_docs(state):
    """
    阶段一：文档合并与标准化

    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式。

    输入来源：
    1. rrf_chunks (List[Dict]): 本地知识库检索结果（经 RRF 融合排序）。
       - 结构：包含 Milvus entity 信息的复杂字典或对象。
       - 关键字段：chunk_id, content, title/paper_title。
    2. web_search_docs (List[Dict]): 联网搜索结果（经 MCP 搜索返回）。
       - 结构：包含搜索摘要的扁平字典。
       - 关键字段：snippet, title, url。

    输出结果 (List[Dict]):
    - 标准化文档列表，每项包含：
      - text: 用于重排序的核心文本（content 或 snippet）
      - title: 标题（用于增强语义或展示）
      - doc_id/chunk_id: 唯一标识（本地文档有，联网文档为 None）
      - url: 来源链接（本地为空，联网文档有）
      - source: 来源标记 ("local" 或 "web")
    """

    # 1. 提取输入源
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []

    logger.info(
        f"Step 1: 开始合并文档 - 本地RRF源: {len(rrf_docs)}条, 联网Web源: {len(web_docs)}条"
    )
    doc_items = []
    # ---------------------------------------------------------
    # 2. 处理本地知识库文档 (rrf_chunks)
    # ---------------------------------------------------------
    for i, doc in enumerate(rrf_docs):
        # 简化：直接使用 dict(doc) 转换，如果 doc 本身是 dict 则无损，如果是对象则尝试转换
        # 由于上游 RRF 节点已经做了 _as_entity_list 处理，这里 doc 极大概率已经是纯字典
        # 因此可以移除繁琐的 try-except 和 entity 嵌套判断，直接取值

        # 兼容性处理：优先取 'entity' 字段（防守式编程），若无则视为 doc 本身即 entity
        # 注意：这里的 doc 应当已经是字典（由上游 _as_entity_list 保证）
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc

        # 提取核心文本 (content)，这是重排序的依据
        # 如果不是字典或无 content，则跳过
        if not isinstance(entity, dict):
            logger.warning(f"本地文档格式异常 (index={i}): {type(entity)}")
            continue

        content = entity.get("content")
        if not content:
            # 仅在 debug 模式记录，避免生产环境日志刷屏
            logger.debug(f"跳过无内容文档 (index={i}, keys={list(entity.keys())})")
            continue

        # 提取元数据 (使用 .get 链式回退，简洁明了)
        doc_id = entity.get("chunk_id") or entity.get("id")
        title = (
            entity.get("title")
            or entity.get("paper_title")
            or entity.get("item_name")
            or ""
        )

        # 组装标准化对象，并保留图表/引用等元数据，供答案节点提取图片。
        item = {
            "text": content,
            "content": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,  # 兼容旧逻辑保留字段
            "title": title,
            "url": "",
            "source": "local",
            "original_rank": i + 1,
            "retrieval_score": _safe_float(
                _first_present(entity.get("rrf_score"), entity.get("score"))
            ),
        }
        doc_items.append(_copy_local_metadata(entity, item))

    # ---------------------------------------------------------
    # 3. 处理联网搜索文档 (web_search_docs)
    # ---------------------------------------------------------
    for i, doc in enumerate(web_docs):
        # 兼容不同字段名：优先取 snippet (摘要)，其次 content
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()

        if not text:
            logger.debug(f"跳过无内容联网结果 (index={i})")
            continue

        doc_items.append(
            {
                "text": text,
                "doc_id": None,  # 联网结果无固定 ID
                "chunk_id": None,
                "title": title,
                "url": url,
                "source": "web",
                "original_rank": i + 1,
            }
        )

    logger.info(f"Step 1: 文档合并完成，共输出 {len(doc_items)} 条标准化文档")
    return doc_items


def step_2_rerank_docs(state, doc_items):
    """
    阶段二：对文档进行重排序
    - 输入 doc_items：[{ text,doc_id}, ...]（由第一阶段产出）
    - 输出：在 state 中写入 reranked_docs（结构化列表）
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""

    # 如果没有文档或问题，直接返回
    if not doc_items or not question:
        logger.warning("Step 2: 跳过重排序 (无文档或无问题)")
        return []

    local_docs = [d for d in doc_items if d.get("source") == "local"]
    web_docs = [d for d in doc_items if d.get("source") == "web"]

    logger.info(f"Step 2: 两路rerank - 本地:{len(local_docs)}条, 联网:{len(web_docs)}条")

    try:
        t0 = time.perf_counter()
        reranker = get_reranker_model()
        policy = _choose_rerank_policy(local_docs, web_docs)
        local_original_weight, local_rerank_weight, web_weight = _policy_weights(policy)

        scored_docs = []

        if local_docs:
            local_texts = [
                (f"{d.get('title', '')}: {d['text']}" if d.get('title') else d['text'] or "")[:RERANK_TEXT_MAX_CHARS]
                for d in local_docs
            ]
            if local_rerank_weight > 0:
                local_pairs = [[question, t] for t in local_texts]
                local_scores = reranker.compute_score(local_pairs, batch_size=8)
                local_norm_scores = _normalize_scores(local_scores)
            else:
                local_scores = [0.0 for _ in local_docs]
                local_norm_scores = [0.0 for _ in local_docs]
            local_total = len(local_docs)

            for rank, (item, text, raw_score, norm_score) in enumerate(
                zip(local_docs, local_texts, local_scores, local_norm_scores),
                start=1,
            ):
                original_rank_score = (local_total - rank + 1) / local_total
                final_score = (
                    local_original_weight * original_rank_score
                    + local_rerank_weight * norm_score
                ) * LOCAL_WEIGHT
                scored_item = item.copy()
                scored_item.update(
                    {
                        "text": item.get("text") or "",
                        "content": item.get("content") or item.get("text") or "",
                        "score": final_score,
                        "raw_rerank_score": float(raw_score),
                        "rerank_score": norm_score,
                        "original_rank_score": original_rank_score,
                        "retrieval_score": item.get("retrieval_score"),
                        "rerank_policy": policy,
                        "source": "local",
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "url": "",
                        "title": item.get("title") or "",
                        "rerank_text": text,
                    }
                )
                scored_docs.append(scored_item)

        if web_docs:
            web_texts = [
                (f"{d.get('title', '')}: {d['text']}" if d.get('title') else d['text'] or "")[:RERANK_TEXT_MAX_CHARS]
                for d in web_docs
            ]
            web_pairs = [[question, t] for t in web_texts]
            web_scores = reranker.compute_score(web_pairs, batch_size=8)
            web_norm_scores = _normalize_scores(web_scores)

            for item, text, raw_score, norm_score in zip(web_docs, web_texts, web_scores, web_norm_scores):
                scored_docs.append({
                    "text": item.get("text") or text,
                    "content": item.get("text") or text,
                    "score": norm_score * web_weight,
                    "raw_rerank_score": float(raw_score),
                    "rerank_score": norm_score,
                    "rerank_policy": policy,
                    "source": "web",
                    "chunk_id": None,
                    "doc_id": None,
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                    "rerank_text": text,
                })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        t1 = time.perf_counter()
        logger.info(f"Step 2: 两路rerank完成，耗时 {t1-t0:.3f}s")
        return scored_docs

    except Exception as e:
        logger.error(f"Step 2: 重排序异常: {e}", exc_info=True)
        return [
            {
                "text": x.get("text"),
                "content": x.get("content") or x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
                "figures": x.get("figures"),
                "tables": x.get("tables"),
                "fig_refs": x.get("fig_refs"),
                "table_refs": x.get("table_refs"),
            }
            for x in doc_items
        ]


def step_3_topk(scored_docs):
    """
    阶段三：动态 TopK（最多 10）
    基于 scored_docs（已按 score 降序排序）进行智能截断，
    核心逻辑：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    :param scored_docs: 列表，元素为带score的文档字典，已按score降序排列，格式如[{"doc": 文档对象, "score": 相关性分数}, ...]
    :return: 列表，动态截断后的TopK文档列表，数量≤10
    """
    # 硬上限：最多取前10条，取全局常量与实际文档数的较小值（避免索引越界）
    # 注：max_topk从全局常量读取，不依赖外部状态，保证逻辑一致性
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK  # 硬下限：至少保留的文档数量（全局常量配置）
    gap_ratio = RERANK_GAP_RATIO  # 相对断崖阈值：分数下降的相对比例阈值（全局常量配置）
    gap_abs = RERANK_GAP_ABS  # 绝对断崖阈值：分数下降的绝对差值阈值（全局常量配置）

    # 1) 断崖截断核心逻辑：从min_topk之后开始检测分数断崖，出现则提前截断
    topk = max_topk  # 默认值：无断崖时取满硬上限（最多10条）
    # 仅当实际可取值超过硬下限时，才触发断崖检测（否则直接取满min_topk）
    if topk > min_topk:
        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距）
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")  # 当前位置文档的分数
            s2 = scored_docs[i + 1].get("score")  # 下一个位置文档的分数

            gap = s1 - s2  # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel = gap / (abs(s1) + 1e-6)
            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(
                    f"Step 3: 触发断崖截断 @ index={i} (Score {s1:.4f} -> {s2:.4f}, Gap={gap:.4f})"
                )
                topk = i + 1  # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                break  # 触发截断后立即退出循环，不再检测后续位置

    # 按最终计算的topk值，截取前topk条文档
    topk_docs = scored_docs[:topk]

    logger.info(f"Step 3: 截断完成，保留前 {len(topk_docs)} 条文档 (TopK={topk})")

    if topk_docs:
        preview = ", ".join(
            [
                f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})"
                for d in topk_docs[:3]
            ]
        )
        logger.debug(f"Step 3: Top3 文档预览: {preview}")

    # 返回动态TopK处理后的文档列表
    return topk_docs
