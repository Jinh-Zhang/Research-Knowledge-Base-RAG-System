import sys
import json
import asyncio
import threading
import re
from typing import Any
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerStreamableHttp
from app.core.logger import logger


def _normalize_title_text(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_tokens(text: str):
    normalized = _normalize_title_text(text)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def _title_overlap_ratio(expected: str, candidate: str) -> float:
    expected_tokens = set(_title_tokens(expected))
    candidate_tokens = set(_title_tokens(candidate))
    if not expected_tokens or not candidate_tokens:
        return 0.0
    return len(expected_tokens & candidate_tokens) / len(expected_tokens)


def _is_related_to_requested_titles(doc: dict, requested_titles) -> bool:
    requested_titles = [title for title in (requested_titles or []) if title]
    if not requested_titles:
        return True

    candidate_title = (doc.get("title") or "").strip()
    candidate_text = f"{candidate_title}\n{(doc.get('snippet') or '').strip()}"
    candidate_norm = _normalize_title_text(candidate_text)
    if not candidate_norm:
        return False

    for requested in requested_titles:
        requested_norm = _normalize_title_text(requested)
        if not requested_norm:
            continue
        if requested_norm in candidate_norm:
            return True
        if _title_overlap_ratio(requested, candidate_title) >= 0.6:
            return True
    return False


def _flatten_exception_messages(exc: BaseException) -> str:
    """将异常及其嵌套子异常展开为便于日志查看的字符串。"""
    if isinstance(exc, BaseExceptionGroup):
        parts = []
        for index, sub_exc in enumerate(exc.exceptions, start=1):
            parts.append(f"sub[{index}]={_flatten_exception_messages(sub_exc)}")
        return f"{type(exc).__name__}({'; '.join(parts)})"
    return f"{type(exc).__name__}: {exc}"


def _run_async(coro: Any):
    """在同步节点中安全执行协程，兼容已有事件循环场景。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder: dict[str, Any] = {"result": None, "error": None}

    def runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_holder["result"] = loop.run_until_complete(coro)
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if result_holder["error"] is not None:
        raise result_holder["error"]
    return result_holder["result"]


async def mcp_call(query):
    if not mcp_config.mcp_base_url:
        logger.error("[MCP] 配置缺失: MCP_DASHSCOPE_BASE_URL 为空")
        return None
    if not mcp_config.api_key:
        logger.error("[MCP] 配置缺失: API Key 为空")
        return None

    try:
        logger.info(f"[MCP] 连接 WebSearch MCP: {mcp_config.mcp_base_url}")

        async with MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {
                    "Authorization": f"Bearer {mcp_config.api_key}"
                },
                "timeout": 300,
            },
            cache_tools_list=True,
        ) as search_mcp:

            logger.info(f"[MCP] 调用 bailian_web_search: {query}")

            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )

            logger.info("[MCP] 调用成功")
            return result

    except Exception as e:
        # 如果是 ExceptionGroup（TaskGroup）
        if hasattr(e, "exceptions"):
            for i, sub in enumerate(e.exceptions, 1):
                logger.error(f"[MCP] 子异常[{i}]: {type(sub).__name__}: {sub}", exc_info=True)
        else:
            logger.error(f"[MCP] 异常: {type(e).__name__}: {e}", exc_info=True)
        return None


def node_web_search_mcp(state):
    """
    LangGraph同步节点函数：处理MCP搜索逻辑，作为整个搜索流程的入口。

    该节点会调用 mcp_call 异步函数获取搜索结果，并将其解析为结构化数据存储到 state 中。

    :param state: LangGraph的全局状态对象，包含 session_id, rewritten_query 等信息
    :return: 字典，包含结构化的搜索结果 web_search_docs，供后续节点使用
    """
    logger.info("---node_web_search_mcp 开始处理---")

    # 1. 标记任务开始
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # 2. 获取查询词
    query = state.get("rewritten_query", "")
    if not query:
        # 尝试回退到原始查询
        query = state.get("original_query", "")

    docs = []
    requested_titles = state.get("requested_titles") or []

    # 3. 执行搜索
    if query:
        try:
            # 同步-异步桥接：通过asyncio.run()执行异步的mcp_call函数
            logger.info(f"启动异步 MCP 调用，Query: {query}")

            # ======================================================================
            # MCP 返回结果格式解析说明
            # ----------------------------------------------------------------------
            # result 是一个 CallToolResult 对象 (定义在 agents.mcp.types 中)
            # result.content 是一个 TextContent 对象的列表，通常只有一项
            # result.content[0].text 是一个 JSON 字符串，包含实际的搜索结果
            #
            # 示例数据结构：
            # result.content[0].text = """
            # {
            #   "pages": [
            #     {
            #       "title": "HAK 180 烫金机使用手册",
            #       "url": "http://example.com/manual",
            #       "snippet": "在出厂默认状态下，若想设置局部转印..."
            #     },
            #     ...
            #   ]
            # }
            # """
            # ======================================================================
            result = _run_async(mcp_call(query))

            # 4. 解析结果
            if result and not result.isError and result.content:
                # 解析MCP原始结果：提取文本内容并转为JSON对象
                # result.content 通常是一个列表，第一项包含文本结果
                raw_text = result.content[0].text
                try:
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []

                    logger.info(f"MCP 返回原始页面数量: {len(pages)}")

                    # 遍历结果，统一封装为结构化格式
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()

                        # 过滤无核心摘要的结果
                        if not snippet:
                            continue

                        docs.append({"title": title, "url": url, "snippet": snippet})

                    if requested_titles:
                        filtered_docs = [
                            doc for doc in docs if _is_related_to_requested_titles(doc, requested_titles)
                        ]
                        logger.info(
                            f"MCP 标题相关性过滤: 原始 {len(docs)} 条 -> 保留 {len(filtered_docs)} 条, requested_titles={requested_titles}"
                        )
                        docs = filtered_docs

                except json.JSONDecodeError:
                    logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")
            else:
                if result and result.isError:
                    logger.error(f"MCP 返回错误: {result}")
                else:
                    logger.warning("MCP 返回结果为空或无效")

            logger.info(f"结构化搜索结果数量: {len(docs)}")

        except Exception as e:
            logger.error(f"MCP 搜索节点执行异常: {e}", exc_info=True)
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")

    # 5. 标记任务结束
    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    logger.info("---node_web_search_mcp 处理结束---")

    # 若有有效搜索结果，返回结果供后续节点使用；无则返回空字典
    if docs:
        return {"web_search_docs": docs}
    return {}


if __name__ == "__main__":
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False,
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get("web_search_docs", [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
