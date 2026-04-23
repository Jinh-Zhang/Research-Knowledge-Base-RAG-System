import sys
import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_paper_titles,
)
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import (
    get_milvus_client,
    create_hybrid_search_requests,
    hybrid_search,
)
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger


load_dotenv(find_dotenv())


HIGH_SCORE_THRESHOLD = 0.8
MID_SCORE_THRESHOLD = 0.5
TITLE_CONTAINS_SCORE_THRESHOLD = 0.45
TOKEN_OVERLAP_THRESHOLD = 0.6



def _normalize_title_text(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_tokens(text: str) -> List[str]:
    normalized = _normalize_title_text(text)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def _calc_title_overlap_ratio(extracted_name: str, candidate_title: str) -> float:
    extracted_tokens = set(_title_tokens(extracted_name))
    candidate_tokens = set(_title_tokens(candidate_title))
    if not extracted_tokens or not candidate_tokens:
        return 0.0
    return len(extracted_tokens & candidate_tokens) / len(extracted_tokens)


def _is_title_text_match(extracted_name: str, candidate_title: str) -> bool:
    extracted_norm = _normalize_title_text(extracted_name)
    candidate_norm = _normalize_title_text(candidate_title)
    if not extracted_norm or not candidate_norm:
        return False

    if extracted_norm == candidate_norm:
        return True

    if extracted_norm in candidate_norm:
        return True

    return (
        _calc_title_overlap_ratio(extracted_norm, candidate_norm)
        >= TOKEN_OVERLAP_THRESHOLD
    )


def _dedupe_title_candidates(candidates: List[str], limit: int = 3) -> List[str]:
    deduped = []
    seen = set()
    for candidate in candidates or []:
        text = (candidate or "").strip()
        norm = _normalize_title_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


def _llm_infer_best_paper(original_query: str, candidates: List[str]) -> Optional[str]:
    """
    当向量检索命中多个中置信度候选时，调用 LLM 从候选列表中推断最可能的论文标题。
    若 LLM 无法确认，返回 None。
    """
    if not candidates:
        return None

    candidates_text = "\n".join(f"- {c}" for c in candidates)
    prompt = (
        f"用户的问题是：\"{original_query}\"\n\n"
        f"从向量数据库中检索到以下候选论文标题：\n{candidates_text}\n\n"
        "请判断用户最可能询问的是哪篇论文。\n"
        "如果可以明确推断出唯一一篇，请仅返回该论文的完整标题（原文，不要修改），不要加任何解释。\n"
        "如果无法确定，请仅返回：UNKNOWN"
    )

    try:
        client = get_llm_client(json_mode=False)
        messages = [
            SystemMessage(content="你是一个专业的科研助手，擅长根据上下文推断用户意图。"),
            HumanMessage(content=prompt),
        ]
        response = client.invoke(messages)
        result = (response.content or "").strip()
        logger.info(f"LLM 二次推断结果: '{result}'")
        if result and result != "UNKNOWN":
            # 验证 LLM 返回的标题确实在候选列表中（防止幻觉）
            for candidate in candidates:
                if candidate.strip() == result or _normalize_title_text(candidate) == _normalize_title_text(result):
                    return candidate
        return None
    except Exception as e:
        logger.error(f"LLM 二次推断失败: {e}")
        return None


def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """
    利用LLM从当前问题以及历史会话中提取出主要询问的论文标题 paper_titles（可多个，JSON列表形式）
    若论文标题不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
    :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
    :param history: 列表[字典] - 近期会话历史
    :return: 字典 - 提取结果，格式：{"paper_titles": [], "rewritten_query": ""}
    """
    logger.info("Step 3: 开始提取信息 (LLM)")

    # 1. 初始化准备
    client = get_llm_client(json_mode=True)

    # 构造历史对话文本
    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"

    logger.info(f"Step 3: 历史上下文构建完成，长度: {len(history_text)} 字符")

    # 2. 加载提示词
    try:
        # 使用关键字参数传递，避免参数位置错误
        prompt = load_prompt(
            "rewritten_query_and_itemnames", history_text=history_text, query=query
        )
        logger.debug(f"Step 3: 提示词加载成功，Prompt长度: {len(prompt)}")
    except Exception as e:
        logger.error(f"Step 3: 加载提示词失败: {e}")
        return {"paper_titles": [], "rewritten_query": query}

    messages = [
        SystemMessage(
            content="你是一个专业的科研助手，擅长理解用户意图和提取关键信息。"
        ),
        HumanMessage(content=prompt),
    ]

    try:
        logger.info("Step 3: 正在调用 LLM 进行提取...")
        response = client.invoke(messages)
        content = response.content
        logger.debug(f"Step 3: LLM 原始响应: {content}")

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)

        # 健壮性检查
        if "paper_titles" not in result:
            result["paper_titles"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
        if "retrieval_titles" not in result:
            result["retrieval_titles"] = []

        result["paper_titles"] = _dedupe_title_candidates(
            result.get("paper_titles", [])
        )
        result["retrieval_titles"] = _dedupe_title_candidates(
            result.get("retrieval_titles", [])
        )

        logger.info(
            f"Step 3: 提取结果解析成功 - 论文标题: {result['paper_titles']}, 检索短语: {result['retrieval_titles']}, 重写问题: {result['rewritten_query']}"
        )
        return result

    except Exception as e:
        logger.error(f"Step 3: LLM 提取或解析失败: {e}")
        return {"paper_titles": [], "rewritten_query": query}


def _generate_general_answer(query: str, history: List[Dict]) -> str:
    """
    当前问题没有明确论文检索目标时，走普通大模型回答，并追加论文 RAG 引导。
    """
    history_text = ""
    for msg in (history or [])[-6:]:
        role = msg.get("role", "unknown")
        text = msg.get("text", "")
        if text:
            history_text += f"{role}: {text}\n"

    prompt = (
        "你是一个友好的科研助手。用户当前问题没有明确指定要检索的论文，"
        "请直接正常回答用户问题，回答要简洁、自然。不要假装已经检索论文数据库。\n\n"
        f"历史对话：\n{history_text or '无'}\n"
        f"用户问题：{query}\n\n"
        "请回答："
    )

    try:
        llm = get_llm_client(temperature=0.5)
        response = llm.invoke(prompt)
        answer = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.error(f"通用回答生成失败: {e}", exc_info=True)
        answer = ""

    return answer


def step_4_vectorize_and_query(paper_titles: List[str]) -> List[Dict]:
    """
    对提取的 paper_titles 进行向量化并在 Milvus 中进行混合搜索
    """
    logger.info(f"Step 4: 开始向量化检索，目标论文: {paper_titles}")
    results = []

    client = get_milvus_client()
    if not client:
        logger.error("Step 4: 无法连接到 Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: 环境变量中未找到 ITEM_NAME_COLLECTION")
        return results

    try:
        logger.info("Step 4: 正在生成 Embedding (Dense + Sparse)...")
        embeddings = generate_embeddings(paper_titles)
        logger.info(
            f"Step 4: 向量生成完成，开始 Milvus 搜索 (Collection: {collection_name})"
        )

        for i, name in enumerate(paper_titles):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                # 构造混合搜索请求
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector, sparse_vector=sparse_vector, limit=5
                )

                # 执行混合搜索
                # 权重调整为 0.8 (Dense) / 0.2 (Sparse) 以优化评分
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["paper_title"],
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        paper_title = entity.get("paper_title") or entity.get(
                            "item_name"
                        )
                        score = hit.get("distance")

                        if paper_title:
                            matches.append({"paper_title": paper_title, "score": score})
                            logger.debug(
                                f"Step 4: '{name}' 匹配项: {paper_title} (Score: {score:.4f})"
                            )

                results.append({"extracted_name": name, "matches": matches})
                logger.info(
                    f"Step 4: 商品 '{name}' 检索完成，找到 {len(matches)} 个匹配项"
                )

            except Exception as inner_e:
                logger.error(f"Step 4: 处理商品 '{name}' 时出错: {inner_e}")
                results.append({"extracted_name": name, "matches": []})

    except Exception as e:
        logger.error(f"Step 4: 向量化或搜索过程发生全局错误: {e}")

    return results


def step_5_align_paper_titles(query_results: List[Dict], original_query: str = "") -> Dict:
    """
    根据 Milvus 搜索评分，对齐论文标题，生成「确认论文标题」和「候选论文标题」
    """
    logger.info("Step 5: 开始对齐论文标题 (Score Analysis)")

    confirmed_paper_titles = []
    options = []

    for res in query_results:
        extracted_name = res.get("extracted_name", "").strip()
        matches = res.get("matches", []) or []

        if not matches:
            logger.info(f"Step 5: '{extracted_name}' 无匹配结果")
            continue

        # 按分数降序
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 打印详细评分日志辅助调试
        top_matches_log = ", ".join(
            [f"{m['paper_title']}({m['score']:.3f})" for m in matches[:3]]
        )
        logger.info(f"Step 5: '{extracted_name}' Top匹配: {top_matches_log}")

        # 优先走标题文本宽松匹配：像 "Test-Time Adaptation" 这类短语，
        # 即使用户没输入完整论文标题，也应该尽量命中包含该短语的真实标题。
        text_matched = []
        for match in matches:
            candidate_title = match.get("paper_title") or ""
            score = match.get("score", 0)
            overlap_ratio = _calc_title_overlap_ratio(extracted_name, candidate_title)
            if _is_title_text_match(extracted_name, candidate_title):
                text_matched.append(match)
                logger.info(
                    f"Step 5: 标题文本匹配命中 -> extracted='{extracted_name}' | "
                    f"candidate='{candidate_title}' | score={score:.3f} | overlap={overlap_ratio:.3f}"
                )

        if text_matched:
            # 文本命中时优先确认更高分结果；如果首个结果分数偏低，则退化为候选而非直接拒绝。
            text_matched.sort(key=lambda x: x.get("score", 0), reverse=True)
            best_match = text_matched[0]
            best_title = best_match.get("paper_title")
            best_score = best_match.get("score", 0)
            if best_score >= TITLE_CONTAINS_SCORE_THRESHOLD:
                confirmed_paper_titles.append(best_title)
                logger.info(
                    f"Step 5: 规则T命中 (Title Text Match) -> 确认: {best_title}"
                )
            else:
                options.extend(
                    [
                        m.get("paper_title")
                        for m in text_matched[:5]
                        if m.get("paper_title")
                    ]
                )
                logger.info(
                    f"Step 5: 规则T命中但分数偏低 -> 添加候选: {[m.get('paper_title') for m in text_matched[:5]]}"
                )
            continue

        # 筛选
        high = [m for m in matches if m.get("score", 0) >= HIGH_SCORE_THRESHOLD]
        mid = [m for m in matches if m.get("score", 0) >= MID_SCORE_THRESHOLD]

        # 规则 A: 单个高置信度
        if len(high) == 1:
            confirmed_name = high[0].get("paper_title")
            confirmed_paper_titles.append(confirmed_name)
            logger.info(f"Step 5: 规则A命中 (Single High) -> 确认: {confirmed_name}")
            continue

        # 规则 B: 多个高置信度
        if len(high) > 1:
            picked = None
            # 优先匹配同名
            if extracted_name:
                for m in high:
                    if m.get("paper_title") == extracted_name:
                        picked = m
                        logger.info(
                            f"Step 5: 规则B命中 (Exact Match in High) -> 确认: {picked.get('paper_title')}"
                        )
                        break

            # 否则取最高分
            if not picked:
                picked = high[0]
                logger.info(
                    f"Step 5: 规则B命中 (Highest Score) -> 确认: {picked.get('paper_title')}"
                )

            confirmed_paper_titles.append(picked.get("paper_title"))
            continue

        # 规则 C: 无高置信度，取中置信度候选 -> 先用 LLM 二次推断
        if len(mid) > 0:
            current_options = [m.get("paper_title") for m in mid[:5] if m.get("paper_title")]
            logger.info(f"Step 5: 规则C命中 (Mid Confidence) -> 候选: {current_options}")

            inferred = _llm_infer_best_paper(original_query or extracted_name, current_options)
            if inferred:
                confirmed_paper_titles.append(inferred)
                logger.info(f"Step 5: 规则C LLM推断命中 -> 确认: {inferred}")
            else:
                options.extend(current_options)
                logger.info(f"Step 5: 规则C LLM无法推断 -> 添加候选: {current_options}")
            continue

        logger.info(f"Step 5: 规则D命中 (Low Confidence) -> 无匹配")

    result = {
        "confirmed_paper_titles": list(set(confirmed_paper_titles)),
        "options": list(set(options)),
    }
    logger.info(f"Step 5: 对齐结果: {result}")
    return result


def step_6_check_confirmation(
    state: Dict,
    align_result: Dict,
    session_id: str,
    history: List[Dict],
    rewritten_query: str,
) -> Dict:
    """
    检查对齐结果，更新 State
    """
    logger.info("Step 6: 检查确认状态并更新 State")

    # 健壮性处理
    if align_result is None:
        align_result = {}

    confirmed = align_result.get("confirmed_paper_titles", [])
    options = align_result.get("options", [])

    # 分支 A: 有确认论文标题
    if confirmed:
        logger.info(f"Step 6: [分支A] 存在确认论文标题: {confirmed}")

        # 更新历史消息中的 paper_titles
        ids_to_update = []
        for msg in history:
            if not msg.get("paper_titles"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))

        if ids_to_update:
            logger.info(f"Step 6: 更新 {len(ids_to_update)} 条历史消息的关联论文标题")
            update_message_paper_titles(ids_to_update, confirmed)

        state["paper_titles"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    # 分支 B: 有候选论文标题
    if options:
        logger.info(f"Step 6: [分支B] 存在候选论文标题: {options}")
        options_str = "、".join(options[:3])
        answer = f"您是想问以下哪篇论文：{options_str}？请进一步明确论文标题。"
        state["answer"] = answer
        state["paper_titles"] = []
        return state

    # 分支 C: 无结果
    logger.info("Step 6: [分支C] 无确认也无候选")
    state["answer"] = _generate_general_answer(
        state.get("original_query", "") or rewritten_query,
        history,
    )
    state["paper_titles"] = []
    state["rewritten_query"] = rewritten_query
    return state


def step_7_write_history(
    state: Dict,
    user_id: str,
    session_id: str,
    history: List[Dict],
    rewritten_query: str,
    message_id: str,
) -> Dict:
    """
    写入最终历史记录
    """
    logger.info("Step 7: 写入会话历史")

    # 如果有助手回答（分支 B/C），写入助手消息
    if state.get("answer"):
        logger.info("Step 7: 保存助手回答")
        save_chat_message(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            paper_titles=[],
        )

    # 更新用户消息（关联 rewrite_query 和 paper_titles）
    logger.info(f"Step 7: 更新用户消息 (ID: {message_id})")
    save_chat_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        paper_titles=state.get("paper_titles", []),
        message_id=message_id,
    )

    return state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    主节点函数：论文标题确认流程
    """
    logger.info(">>> node_item_name_confirm: 开始处理")

    session_id = state["session_id"]
    user_id = state.get("user_id", "anonymous")
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    # 标记任务开始
    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # 1. 获取历史记录
    history = get_recent_messages(session_id, user_id=user_id, limit=10)
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")

    # 2. 保存用户当前消息 (初始保存，后续 step 7 会更新)
    message_id = save_chat_message(
        user_id, session_id, "user", original_query, "", state.get("paper_titles", [])
    )
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")

    # 3. 提取信息
    extract_res = step_3_extract_info(original_query, history)
    paper_titles = _dedupe_title_candidates(extract_res.get("paper_titles", []))
    retrieval_titles = _dedupe_title_candidates(extract_res.get("retrieval_titles", []))
    rewritten_query = extract_res.get("rewritten_query", original_query)

    if len(paper_titles) == 0:
        if retrieval_titles:
            logger.info(
                f"Node: LLM 未提取到标准论文标题，改用同次返回的检索短语 -> {retrieval_titles}"
            )
            paper_titles = retrieval_titles
        else:
            logger.info("Node: LLM 未提取到论文标题，检索短语也为空")

    # 更新 State 中的 rewrite_query
    state["rewritten_query"] = rewritten_query

    align_result = {}

    # 4. & 5. 如果有提取到论文标题，进行搜索和对齐
    if len(paper_titles) > 0:
        query_results = step_4_vectorize_and_query(paper_titles)
        align_result = step_5_align_paper_titles(query_results, original_query)
    else:
        logger.info("Node: 未提取到论文标题，跳过向量检索")

    # 6. 检查确认状态
    state = step_6_check_confirmation(
        state, align_result, session_id, history, rewritten_query
    )

    # 7. 写入最终历史
    final_state = step_7_write_history(
        state, user_id, session_id, history, rewritten_query, message_id
    )

    # 将 history 存入 state，供后续节点（如 node_answer_output）使用
    final_state["history"] = history

    # 标记任务完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)

    logger.info(
        f"Node: 处理结束, Final State Paper Titles: {final_state.get('paper_titles')}"
    )
    return final_state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "is_stream": False,
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("paper_titles"):
            print(f"\n[PASS] 成功提取并确认论文标题: {result_state['paper_titles']}")
        else:
            print(f"\n[WARN] 未确认到论文标题 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
