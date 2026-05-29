import sys
import json
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re
from urllib.parse import unquote
from html.parser import HTMLParser

MAX_CONTEXT_CHARS = 12000
IMAGE_BLOCK_PATTERN = re.compile(r"【\s*图片\s*】|\[\s*图片\s*\]")
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
DISPLAY_TITLE_KEYS = ("paper_title", "file_title", "title", "parent_title")

def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    2 根据state中的问题、重新问题、历史对话、提问论文（paper_titles）、 重排内容 组织prompt 并调用llm 生成答案
    3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
    4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
    5 做最后一次push操作（主要是为了触发前端图片渲染)
       {
          "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
          "status": "completed",
          "image_urls": [
              "http://local-server/images/panel_view.jpg",
              "http://local-server/images/button_detail.jpg"
          ]
        }
    """
    logger.info("---node_answer_output (答案生成) 节点开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
    answer_exists = step_1_check_answer(state)

    # 阶段二  如果没有answer则 构建 Prompt
    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt
        state["answer_suffix"] = _build_answer_suffix_from_docs(state.get("reranked_docs") or [])

        # 阶段三：  如果没有answer则 调用大模型输出答案
        step_3_generate_response(state, prompt)

    if state.get("answer"):
        if not state.get("is_stream"):
            set_task_result(state["session_id"], "answer", state["answer"])

    # 提取候选图片信息，再按答案中明确输出的【图片】区块过滤。
    # 这样检索上下文里有其它图时，不会把模型没有选择的图片展示给用户。
    candidate_image_infos = _extract_image_infos_from_docs(state.get("reranked_docs") or [])
    selected_image_urls = _extract_image_urls_from_answer(state.get("answer") or "")
    image_infos = _filter_image_infos_by_urls(candidate_image_infos, selected_image_urls)
    image_urls = [x["image_url"] for x in image_infos if x.get("image_url")]
    state["image_urls"] = image_urls
    state["image_infos"] = image_infos
    if not state.get("is_stream"):
        set_task_result(state["session_id"], "image_urls", image_urls)
        set_task_result(state["session_id"], "image_infos", image_infos)

    # 阶段四：把答案写入到mongodb的history中
    if state.get("answer"):
        logger.info("---写入MongoDB历史记录---")
        step_4_write_history(
            state,
            image_urls=image_urls,
            image_infos=image_infos,
        )

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # 阶段五: 流式输出结束，发送 final 事件 [最后兜底，确保图片都能争取渲染和结束]
    logger.info(f"---发送 final 事件---图片为：{image_urls}")
    if state.get("is_stream"):
        push_to_session(
            state["session_id"],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "status": "completed",
                "image_urls": image_urls,  # 兼容旧前端
                "image_infos": image_infos,  # 带 Figure 编号/来源/caption 的结构化图片信息
            },
        )

    logger.info("---node_answer_output 节点处理结束---")
    return state


def _apply_answer_prefix(state: QueryGraphState) -> QueryGraphState:
    prefix = (state.get("answer_prefix") or "").strip()
    answer = (state.get("answer") or "").strip()
    if not prefix or not answer:
        return state
    if answer.startswith(prefix):
        return state
    state["answer"] = f"{prefix}\n\n{answer}"
    return state


def _apply_answer_suffix(state: QueryGraphState) -> QueryGraphState:
    suffix = (state.get("answer_suffix") or "").strip()
    answer = (state.get("answer") or "").strip()
    if not suffix or not answer:
        return state
    if answer.endswith(suffix):
        return state
    state["answer"] = f"{answer}\n\n{suffix}"
    return state


def step_1_check_answer(state) -> bool:
    """
    阶段一：检查 state 中是否已有 answer。
    - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
    - 若不存在：返回 False
    """
    answer = state.get("answer", None)
    is_stream = state.get("is_stream")
    if answer:
        if is_stream:
            logger.info("---Step 1: 发现已有答案，执行流式推送---")
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    else:
        return False


# 目标结构
# HAK 180 烫金机的操作面板位于机器正前方。开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
# 具体的按键位置请参考下图：
# 【图片】
# http://local-server/images/panel_view.jpg
# http://local-server/images/button_detail.jpg
def step_2_construct_prompt(state: QueryGraphState) -> str:
    """
    第一阶段：构建 Prompt
    根据state中的问题、重新问题、历史对话、提问论文（paper_titles）、 重排内容 组织prompt
    """
    # 1. 获取相关信息
    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "")
    # 优先使用重写后的问题
    question = rewritten_query if rewritten_query else original_query
    history = state.get("history", [])
    paper_titles = state.get("paper_titles", [])
    reranked_docs = state.get("reranked_docs") or []

    # 2 从重排内容中，提取为资料字符串，不可超过限额
    # 优先使用结构化 reranked_docs（包含 source/chunk_id/url/score），便于约束与引用
    # ---------------------------------------------------------
    # 逻辑解释：
    # 1. 遍历重排序后的文档列表 (reranked_docs)，这些文档已经按相关性从高到低排序。
    # 2. 对每个文档提取关键信息 (text, source, chunk_id, url, title, score)。
    # 3. 构造 "元数据头 + 正文" 格式的字符串，例如：
    #    "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
    #     这里是文档的正文内容..."
    # 4. 累加字符长度，如果超过 MAX_CONTEXT_CHARS (如 12000 字符)，则停止添加，
    #    确保 Prompt 长度在 LLM 的处理范围内，避免 Token 溢出。
    # ---------------------------------------------------------
    docs = []
    used = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (
            _get_doc_field(doc, "content", "")
            or _get_doc_field(doc, "text", "")
            ).strip()
        if not text:
            continue
        source = doc.get("source") or ""
        chunk_id = doc.get("chunk_id")
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        score = doc.get("score")

        figure_text = _format_figures_for_prompt(doc)
        table_text = _format_tables_for_prompt(doc)
        doc_text = text + figure_text + table_text

        meta_parts = [f"片段{i}"]
        if source:
            meta_parts.append(f"[{source}]")
        if chunk_id:
            meta_parts.append(f"[chunk_id={chunk_id}]")
        if url:
            meta_parts.append(f"[url={url}]")
        if score is not None:
            # 保留四位小数
            meta_parts.append(f"[score={float(score):.4f}]")
        if title:
            meta_parts.append(f"[title={title}]")
        doc_block = " ".join(meta_parts) + "\n" + doc_text
        if used + len(doc_block) > MAX_CONTEXT_CHARS:
            break
        docs.append(doc_block)
        # 计算使用长度！ + 2 两个\n\n
        used += len(doc_block) + 2
    context_str = "\n\n".join(docs) if docs else "无参考内容"

    # 3. 格式化 History (历史对话)
    # ---------------------------------------------------------
    # 逻辑解释：
    # 1. 遍历历史对话记录 (history)。
    # 2. 将每轮对话格式化为 "用户: ... \n 助手: ..." 的文本块。
    # 3. 同样进行长度累加判断 (used)，确保历史记录+参考文档的总长度不超过 MAX_CONTEXT_CHARS。
    #    注意：这里的 used 变量是接着上面处理文档后的长度继续累加的，
    #    意味着如果文档占用了太多 Token，历史记录可能会被截断或完全丢弃。
    # ---------------------------------------------------------
    history_str = ""
    if history:
        for msg in history:
            # 修正：MongoDB存储格式为 {"role": "user"/"assistant", "text": "..."}
            role = msg.get("role")
            text = msg.get("text")
            if role == "user" and text:
                history_str += f"用户: {text}\n"
            elif role == "assistant" and text:
                history_str += f"助手: {text}\n"

            used += len(history_str) + 2
            if used > MAX_CONTEXT_CHARS:
                break
    else:
        history_str = "无历史对话"

    # 4. 格式化 Paper Titles (提问论文)
    paper_titles_str = ", ".join(paper_titles) if paper_titles else "无指定论文"

    # 5. 组装 Prompt
    prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        paper_titles=paper_titles_str,
        question=question,
    )

    logger.info(f"组装后的提示词为：{prompt}")

    return prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
    """
    第二阶段：生成回答
    调用llm生成答案，支持流式输出
    """
    logger.info("---Step 3: 开始生成回答 (LLM Generation)---")
    logger.debug(f"最终Prompt内容: {prompt}")

    # 获取 LLM 客户端
    # 注意：这里我们使用统一的 get_llm_client 获取实例
    llm = get_llm_client()

    # 判断是否需要流式输出
    # 通常 state 中会注入 stream_queue 用于 SSE 推送
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")

    if is_stream:
        logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
        final_text = ""
        try:
            # 使用 stream 方法进行流式生成
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    final_text += delta
                    # 将增量内容放入队列
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

            logger.info(f"流式输出完成，总长度: {len(final_text)}")

        except Exception as e:
            logger.error(f"流式生成出错: {e}", exc_info=True)
            # 发生错误时，尝试推送到前端
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

        state["answer"] = final_text
        _apply_answer_prefix(state)
        _apply_answer_suffix(state)
    else:
        # 非流式直接调用
        logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
        try:
            response = llm.invoke(prompt)
            content = response.content
            state["answer"] = content
            _apply_answer_prefix(state)
            _apply_answer_suffix(state)
            set_task_result(session_id, "answer", state["answer"])
            logger.info(f"生成回答完成，长度: {len(content)}")
        except Exception as e:
            logger.error(f"生成回答出错: {e}", exc_info=True)
            state["answer"] = "抱歉，生成回答时出现错误。"

    return state

def _get_doc_field(doc, key, default=None):
    """
    兼容普通dict和Milvus hit结构：
    doc[key]
    doc["entity"][key]
    """
    if default is None:
        default = ""
    if not isinstance(doc, dict):
        return default

    if key in doc and doc.get(key) is not None:
        return doc.get(key)

    entity = doc.get("entity") or {}
    if isinstance(entity, dict) and key in entity and entity.get(key) is not None:
        return entity.get(key)

    return default


def _best_doc_title(doc):
    for key in DISPLAY_TITLE_KEYS:
        value = _get_doc_field(doc, key, "")
        if value:
            return value
    return ""


def _load_json_field(value, default):
    """
    兼容：
    - list/dict 原始对象
    - JSON字符串
    - 空值
    """
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _normalize_url(url):
    return (url or "").strip().replace(" ", "%20").rstrip("，。,;；)]】）>\"'")


def _url_match_keys(url):
    normalized = _normalize_url(url)
    if not normalized:
        return set()
    keys = {normalized}
    decoded = unquote(normalized)
    if decoded:
        keys.add(decoded)
        keys.add(decoded.replace(" ", "%20"))
    return keys


def _is_image_url(url):
    base = _normalize_url(url).split("?", 1)[0].split("#", 1)[0].lower()
    return base.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"))


def _extract_image_urls_from_answer(answer):
    """
    只从模型显式输出的【图片】区块里取图片 URL。
    没有该区块时返回空列表，避免把检索候选图误展示成答案图片。
    """
    answer = answer or ""
    matches = list(IMAGE_BLOCK_PATTERN.finditer(answer))
    if not matches:
        return []

    tail = answer[matches[-1].end():]
    urls = []
    seen = set()
    for match in URL_PATTERN.finditer(tail):
        url = _normalize_url(match.group(0))
        if not _is_image_url(url) or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _filter_image_infos_by_urls(image_infos, selected_urls):
    if not selected_urls:
        return []

    info_by_url = {}
    for info in image_infos or []:
        if not isinstance(info, dict) or not info.get("image_url"):
            continue
        for key in _url_match_keys(info.get("image_url")):
            info_by_url.setdefault(key, info)

    filtered = []
    for url in selected_urls:
        normalized = _normalize_url(url)
        info = next((info_by_url.get(key) for key in _url_match_keys(normalized) if info_by_url.get(key)), None)
        if info:
            filtered.append(info)
        else:
            filtered.append({"image_url": normalized})
    return filtered


def _format_figures_for_prompt(doc):
    """
    把 split 阶段生成的 figures 字段转换成可放进prompt的文本。
    """
    figures = _load_json_field(_get_doc_field(doc, "figures", []), [])
    if not figures:
        text = (
            _get_doc_field(doc, "content", "")
            or _get_doc_field(doc, "text", "")
        )
        figures = _extract_figures_from_text(text)
    if not figures:
        return ""

    lines = ["\n【相关图片】"]
    for fig in figures:
        if not isinstance(fig, dict):
            continue
        fig_id = fig.get("figure_id", "")
        caption = fig.get("caption", "")
        image_url = fig.get("image_url", "")
        if image_url:
            lines.append(f"Figure {fig_id}: {caption}")
            lines.append(f"Image URL: {image_url}")

    return "\n".join(lines)


def _format_tables_for_prompt(doc):
    """
    把 split 阶段生成的 tables 字段转换成可放进 prompt 的文本。
    """
    tables = _load_json_field(_get_doc_field(doc, "tables", []), [])
    if not tables:
        return ""

    lines = ["\n\nRelated tables:"]
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_id = table.get("table_id", "")
        caption = table.get("caption", "")
        table_html = table.get("table_html", "")
        if caption:
            lines.append(f"Table {table_id}: {caption}" if table_id else caption)
        if table_html:
            table_markdown = _html_table_to_markdown(table_html)
            if table_markdown:
                lines.append(f"Table Markdown:\n{table_markdown}")
            else:
                lines.append(f"Table HTML: {table_html}")

    return "\n".join(lines)


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = None
        self.current_cell = None
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th") and self.current_row is not None:
            self.current_cell = []
            self.in_cell = True

    def handle_data(self, data):
        if self.in_cell and self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self.in_cell and self.current_row is not None:
            text = re.sub(r"\s+", " ", "".join(self.current_cell or [])).strip()
            self.current_row.append(text)
            self.current_cell = None
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None:
            if any(cell for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None


def _html_table_to_markdown(table_html):
    if not table_html:
        return ""
    parser = _TableHTMLParser()
    try:
        parser.feed(table_html)
    except Exception:
        return ""

    rows = parser.rows
    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]

    def fmt(row):
        return "| " + " | ".join((cell or "").replace("|", "\\|") for cell in row) + " |"

    lines = [fmt(header), "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def _extract_figures_from_text(text):
    """
    从 chunk 正文里解析 Markdown 图片及其紧邻的 Figure/Fig. caption。
    兼容这种常见格式：
      ![alt](url)
      Figure 5: caption...
    """
    if not text:
        return []

    figures = []
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    caption_pattern = re.compile(
        r"\b(?:Figure|Fig\.?)\s*([A-Za-z0-9_.-]+)\s*[:：]\s*([^\n]+)",
        re.IGNORECASE,
    )

    for match in image_pattern.finditer(text):
        alt_text = (match.group(1) or "").strip()
        image_url = (match.group(2) or "").strip()
        if not image_url:
            continue

        # 图片后 600 字符内找最近的 Figure caption，避免跨太远误配。
        tail = text[match.end(): match.end() + 600]
        caption_match = caption_pattern.search(tail)
        figure_id = ""
        caption = ""
        if caption_match:
            figure_id = (caption_match.group(1) or "").strip()
            caption = (caption_match.group(2) or "").strip()

        figures.append(
            {
                "figure_id": figure_id,
                "caption": caption,
                "image_url": image_url,
                "alt_text": alt_text,
            }
        )

    return figures


def _normalize_figure_caption(caption, figure_id=""):
    caption = (caption or "").strip()
    if not caption:
        return ""
    if figure_id:
        caption = re.sub(
            rf"^\s*(?:Figure|Fig\.?)\s*{re.escape(str(figure_id))}\s*[:：]\s*",
            "",
            caption,
            flags=re.IGNORECASE,
        )
    else:
        caption = re.sub(r"^\s*(?:Figure|Fig\.?)\s*[A-Za-z0-9_.-]+\s*[:：]\s*", "", caption, flags=re.IGNORECASE)
    return caption.strip()


def _build_answer_suffix_from_docs(docs):
    if not docs:
        return ""

    local_titles = []
    web_titles = []
    seen_local = set()
    seen_web = set()

    for doc in docs:
        source = (_get_doc_field(doc, "source", "") or "").strip().lower()
        title = (_best_doc_title(doc) or "").strip()
        if not title:
            continue
        if source == "web":
            if title not in seen_web:
                seen_web.add(title)
                web_titles.append(title)
        else:
            if title not in seen_local:
                seen_local.add(title)
                local_titles.append(title)

    lines = []
    if local_titles:
        lines.append(f"主要本地来源：{'；'.join(local_titles[:3])}")
    if web_titles:
        lines.append(f"主要联网来源：{'；'.join(web_titles[:3])}")
    return "\n".join(lines)


def _extract_image_infos_from_docs(docs):
    """
    辅助方法：从文档列表中提取图片URL

    核心逻辑：
    1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
    2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
       - 验证后缀名是否为图片格式 (.jpg, .png 等)。
    3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
       - 匹配 Markdown 图片语法: ![alt text](image_url)。
    4. 对提取到的 URL 进行去重处理，返回唯一图片列表。

    :param docs: 文档列表，每个文档为字典格式
    :return: 图片 URL 字符串列表
    """
    image_infos = []
    seen = set()  # 用于去重，避免同一张图片重复出现
    if not docs:
        return []
    # ---------------------------------------------------------
    # 正则表达式解释：r'!\[.*?\]\((.*?)\)'
    # 1. !\[   -> 匹配 Markdown 图片语法的开头 "![" (注意 [ 需要转义)
    # 2. .*?   -> 非贪婪匹配图片描述文本 (Alt Text)，即 [] 中间的内容
    # 3. \]    -> 匹配描述文本的结束符 "]"
    # 4. \(    -> 匹配 URL 部分的开始符 "("
    # 5. (.*?) -> 捕获组 (Group 1)：非贪婪匹配括号内的实际 URL 内容
    # 6. \)    -> 匹配 URL 部分的结束符 ")"
    # ( ... ) （不带反斜杠）：这就是 捕获组 。
    # 它的作用是告诉程序：“虽然我匹配了整个 ![...](...) 结构，但我 只要 这括号里的内容”。
    # ---------------------------------------------------------
    md_img_pattern = re.compile(r"!\[.*?\]\((.*?)\)")

    def is_image_url(url):
        url = (url or "").strip().split("?", 1)[0].split("#", 1)[0].lower()
        return url.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"))

    def add_img(url, source="", require_image_ext=False, meta=None):
        url = (url or "").strip()
        if not url:
            return
        if require_image_ext and not is_image_url(url):
            return
        if url in seen:
            if meta:
                for info in image_infos:
                    if info.get("image_url") == url:
                        for key, value in meta.items():
                            if value and not info.get(key):
                                info[key] = value
                        break
            return
        else:
            logger.debug(f"发现图片 URL ({source}): {url}")
            seen.add(url)
            info = dict(meta or {})
            info["image_url"] = url
            image_infos.append(info)

    logger.info(f"开始提取图片，待处理文档数: {len(docs)}")

    for i, doc in enumerate(docs):
        doc_meta = {
            "doc_rank": i + 1,
            "source": _get_doc_field(doc, "source", ""),
            "chunk_id": _get_doc_field(doc, "chunk_id", "") or _get_doc_field(doc, "doc_id", ""),
            "title": _best_doc_title(doc),
            "paper_title": _get_doc_field(doc, "paper_title", ""),
            "file_title": _get_doc_field(doc, "file_title", ""),
            "parent_title": _get_doc_field(doc, "parent_title", ""),
        }
        add_img(
            _get_doc_field(doc, "url", ""),
            f"doc[{i}].url",
            require_image_ext=True,
            meta=doc_meta,
        )

        # 2. content/text 里的 Markdown 图片
        text = (
            _get_doc_field(doc, "content", "")
            or _get_doc_field(doc, "text", "")
        ).strip()

        if text:
            parsed_figures = _extract_figures_from_text(text)
            captioned_urls = {
                fig.get("image_url") for fig in parsed_figures if fig.get("image_url")
            }
            for fig in parsed_figures:
                add_img(
                    fig.get("image_url"),
                    f"doc[{i}].content/text",
                    meta={
                        **doc_meta,
                        "figure_id": fig.get("figure_id", ""),
                        "caption": fig.get("caption", ""),
                        "alt_text": fig.get("alt_text", ""),
                    },
                )
            for img_url in md_img_pattern.findall(text):
                if img_url in captioned_urls:
                    continue
                add_img(img_url, f"doc[{i}].content/text", meta=doc_meta)

        # 3. split 阶段新增的 figures 字段
        figures = _load_json_field(_get_doc_field(doc, "figures", []), [])
        if isinstance(figures, list):
            for fig in figures:
                if isinstance(fig, dict):
                    add_img(
                        fig.get("image_url"),
                        f"doc[{i}].figures",
                        meta={
                            **doc_meta,
                            "figure_id": fig.get("figure_id", ""),
                            "caption": _normalize_figure_caption(
                                fig.get("caption", ""),
                                fig.get("figure_id", ""),
                            ),
                            "alt_text": fig.get("alt_text", ""),
                        },
                    )

    logger.info(f"图片提取完成，共找到 {len(image_infos)} 张唯一图片: {image_infos}")
    return image_infos


def _extract_images_from_docs(docs):
    return [x["image_url"] for x in _extract_image_infos_from_docs(docs) if x.get("image_url")]


def step_4_write_history(
    state: QueryGraphState,
    image_urls=None,
    image_infos=None,
) -> QueryGraphState:
    """
    阶段四：把本轮答案写入 MongoDB history。
    利用 utils/mongo_history_utils.py 中的 save_chat_messages 方法。
    """
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "anonymous")
    answer = (state.get("answer") or "").strip()
    paper_titles = state.get("paper_titles") or []

    try:
        if answer:
            save_chat_message(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query="",
                paper_titles=paper_titles,
                image_urls=image_urls,
                image_infos=image_infos,
                message_id=None,
            )
    except Exception as e:
        # 写历史失败不应影响主链路
        logger.error(f"写入Mongo历史记录失败: {e}")

    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            
            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """,
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断...",
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。",
        },
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"},
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "paper_titles": ["Retrieval-Augmented Generation"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None,  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
