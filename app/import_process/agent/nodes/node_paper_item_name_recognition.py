import os
import json
import re
import sys
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client
from app.utils.escape_milvus_string_utils import escape_milvus_string


DEFAULT_PAPER_CHUNK_K = 6
SINGLE_CHUNK_CONTENT_MAX_LEN = 1000
CONTEXT_TOTAL_MAX_CHARS = 4000


VENUE_ALIASES = {
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "conference on neural information processing systems": "NeurIPS",
    "neural information processing systems": "NeurIPS",
    "iclr": "ICLR",
    "international conference on learning representations": "ICLR",
    "icml": "ICML",
    "international conference on machine learning": "ICML",
    "cvpr": "CVPR",
    "computer vision and pattern recognition": "CVPR",
    "conference on computer vision and pattern recognition": "CVPR",
    "iccv": "ICCV",
    "international conference on computer vision": "ICCV",
    "eccv": "ECCV",
    "european conference on computer vision": "ECCV",
    "acl": "ACL",
    "association for computational linguistics": "ACL",
    "emnlp": "EMNLP",
    "empirical methods in natural language processing": "EMNLP",
    "naacl": "NAACL",
    "aaai": "AAAI",
    "ijcai": "IJCAI",
    "kdd": "KDD",
    "www": "WWW",
    "sigir": "SIGIR",
    "wacv": "WACV",
    "colm": "COLM",
    "icassp": "ICASSP",
    "interspeech": "INTERSPEECH",
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _fallback_title(file_title: str) -> str:
    title = _clean_text(file_title)
    title = re.sub(r"\.(pdf|md)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_-]+", " ", title)
    return title.strip() or "Unknown Paper"


def _normalize_venue(value: str) -> str:
    venue = _clean_text(value)
    if not venue:
        return ""

    normalized = re.sub(r"[_\-]+", " ", venue).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for raw, canonical in VENUE_ALIASES.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(raw)}(?![A-Za-z])", normalized, re.IGNORECASE):
            return canonical

    return venue


def _extract_file_metadata(file_title: str) -> Dict[str, str]:
    """
    Extract stable paper metadata commonly encoded in the uploaded filename.

    Many conference PDFs do not contain venue/year in the body, while filenames
    often look like "NeurIPS 2025 - Paper Title.pdf". Keep this deterministic
    and conservative so it can be used for filtering imported papers.
    """
    title = _clean_text(file_title)
    normalized = re.sub(r"[_\-]+", " ", title)

    venue = ""
    for raw, canonical in VENUE_ALIASES.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(raw)}(?![A-Za-z])", normalized, re.IGNORECASE):
            venue = canonical
            break

    year = ""
    year_match = re.search(r"(?<!\d)(20[0-4]\d)(?!\d)", normalized)
    if year_match:
        year = year_match.group(1)

    return {"venue": venue, "year": year}


def _join_values(values: List[str]) -> str:
    cleaned = [
        _clean_text(str(value)) for value in (values or []) if _clean_text(str(value))
    ]
    return "; ".join(cleaned)


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict[str, Any]]]:
    file_title = state.get("file_title", "") or state.get("file_name", "")
    chunks = state.get("chunks") or []

    if not file_title and chunks and isinstance(chunks[0], dict):
        file_title = chunks[0].get("file_title", "")

    if not isinstance(chunks, list) or not chunks:
        logger.warning("state中chunks为空或非列表类型，无法进行论文信息识别")
        return file_title, []

    logger.info(f"论文识别输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks


def step_2_build_context(
    chunks: List[Dict[str, Any]],
    k: int = DEFAULT_PAPER_CHUNK_K,
    max_chars: int = CONTEXT_TOTAL_MAX_CHARS,
) -> str:
    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0
    for idx, chunk in enumerate(chunks[:k], start=1):
        if not isinstance(chunk, dict):
            continue

        chunk_title = _clean_text(chunk.get("title", ""))
        chunk_content = _clean_text(chunk.get("content", ""))
        if not (chunk_title or chunk_content):
            continue

        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]

        piece = f"【片段{idx}】\n标题：{chunk_title}\n内容：{chunk_content}"
        parts.append(piece)
        total_chars += len(piece)
        if total_chars > max_chars:
            break

    return "\n\n".join(parts).strip()[:max_chars]


def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        logger.warning("论文识别节点返回非JSON，使用兜底标题")
        return {}


def step_3_call_llm(file_title: str, context: str) -> Dict[str, Any]:
    fallback_title = _fallback_title(file_title)
    file_meta = _extract_file_metadata(file_title)
    if not context:
        return {
            "paper_title": fallback_title,
            "authors_text": "",
            "year": file_meta.get("year", ""),
            "venue": file_meta.get("venue", ""),
            "keywords_text": "",
            "abstract": "",
        }

    try:
        system_prompt = load_prompt("paper_recognition_system")
        human_prompt = load_prompt(
            "paper_item_name_recognition",
            file_title=file_title,
            context=context,
        )
        llm = get_llm_client(json_mode=True)
        if not llm:
            raise ValueError("LLM client is unavailable")

        resp = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        result = _parse_llm_json(getattr(resp, "content", ""))
    except Exception as exc:
        logger.error(f"论文信息识别失败，使用兜底标题：{exc}", exc_info=True)
        result = {}

    paper_title = _clean_text(result.get("paper_title", "")) or fallback_title
    authors = result.get("authors") or []
    if not isinstance(authors, list):
        authors = [_clean_text(str(authors))] if authors else []
    authors = [
        _clean_text(str(author)) for author in authors if _clean_text(str(author))
    ]

    keywords = result.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = [_clean_text(str(keywords))] if keywords else []
    keywords = [
        _clean_text(str(keyword)) for keyword in keywords if _clean_text(str(keyword))
    ]

    year_text = _clean_text(str(result.get("year", "")))
    year_match = re.search(r"(?<!\d)(20[0-4]\d)(?!\d)", year_text)
    year = (year_match.group(1) if year_match else year_text) or file_meta.get("year", "")
    venue = _normalize_venue(str(result.get("venue", ""))) or file_meta.get("venue", "")
    abstract = _clean_text(result.get("abstract", ""))

    return {
        "paper_title": paper_title,
        "authors_text": _join_values(authors),
        "year": year,
        "venue": venue,
        "keywords_text": _join_values(keywords),
        "abstract": abstract,
    }


def step_4_update_chunks(
    state: ImportGraphState,
    chunks: List[Dict[str, Any]],
    paper_meta: Dict[str, Any],
) -> None:
    paper_title = paper_meta["paper_title"]
    state["item_name"] = paper_title
    state["paper_title"] = paper_title
    state["paper_metadata"] = paper_meta

    authors_text = _clean_text(paper_meta.get("authors_text", ""))
    keywords_text = _clean_text(paper_meta.get("keywords_text", ""))
    year = _clean_text(str(paper_meta.get("year", "")))
    venue = _clean_text(str(paper_meta.get("venue", "")))

    for chunk in chunks:
        chunk["item_name"] = paper_title
        chunk["paper_title"] = paper_title
        # chunk["authors"] = paper_meta.get("authors", [])
        chunk["authors_text"] = authors_text
        chunk["year"] = year
        chunk["venue"] = venue
        # chunk["keywords"] = paper_meta.get("keywords", [])
        chunk["keywords_text"] = keywords_text

    state["chunks"] = chunks

def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]:
    """
    步骤 5: 为商品名称生成BGE-M3稠密+稀疏双向量（Milvus向量检索核心）
    核心说明：
        - 稠密向量（dense_vector）：BGE-M3固定1024维，记录文本深层语义信息
        - 稀疏向量（sparse_vector）：变长键值对，记录文本关键词/特征位置信息
    依赖工具：
        generate_embeddings：封装BGE-M3模型，批量生成双向量，兼容单条/批量输入
    参数：
        item_name: 步骤3识别的商品名称（非空，空值时直接返回空向量）
    返回值：
        Tuple[Any, Any]: (稠密向量列表, 稀疏向量字典)，空值/异常时返回(None, None)
    """
    logger.info(f"开始执行步骤5：为商品名称[{item_name}]生成BGE-M3双向量")

    # 商品名称为空，直接返回空向量，跳过模型调用
    if not item_name:
        logger.warning("商品名称为空，跳过向量生成，返回空向量")
        return None, None

    try:
        # 调用向量生成工具：传入列表支持批量生成，单条数据仍用列表保证格式统一
        vector_result = generate_embeddings([item_name])

        # 向量生成结果非空，才进行后续解析
        if vector_result and "dense" in vector_result and "sparse" in vector_result:
            # 稠密向量解析：取批量结果第一个，为Python列表（Milvus存储要求）
            dense_vector = vector_result["dense"][0]
            # 稀疏向量解析：取批量结果第一个，CSR矩阵解析为字典格式
            sparse_vector = vector_result["sparse"][0]
            logger.info("步骤5：BGE-M3稠密+稀疏向量生成成功")
        else:
            logger.warning("步骤5：向量生成工具返回空结果，无法提取双向量")
            dense_vector, sparse_vector = None, None

    # 捕获所有异常：模型加载失败、向量生成超时、格式错误等
    except Exception as e:
        logger.error(f"步骤5：向量生成失败，原因：{str(e)}", exc_info=True)
        dense_vector, sparse_vector = None, None

    return dense_vector, sparse_vector


def step_6_save_to_milvus(
    state: ImportGraphState,
    file_title: str,
    item_name: str,
    dense_vector,
    sparse_vector,
):
    """
    步骤 6: 将商品名称、文件标题、双向量持久化到Milvus向量数据库
    核心逻辑：
        1. 配置校验：检查Milvus连接地址和集合名配置，缺失则跳过
        2. 客户端获取：获取单例Milvus客户端，连接失败则跳过
        3. 集合初始化：无集合则创建（定义Schema+索引），有集合则直接使用（保留原有配置）
        4. 幂等性处理：删除同名商品数据，避免重复存储
        5. 数据插入：构造符合Schema的数据，非空向量才添加
        6. 集合加载：插入后强制加载集合，确保数据立即可查/Attu可见
    参数：
        state: 流程状态对象，用于最终状态同步
        file_title: 处理后的文件标题
        item_name: 识别后的商品名称（主键去重依据）
        dense_vector: 步骤5生成的稠密向量（1024维列表）
        sparse_vector: 步骤5生成的稀疏向量（字典格式）
    """
    # 从环境变量读取Milvus核心配置，与MilvusConfig配置类保持一致
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")

    # 配置缺失校验：任一配置为空则跳过Milvus存储，记录警告
    if not all([milvus_uri, collection_name]):
        logger.warning(
            "Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存"
        )
        return

    logger.info(
        f"开始执行步骤6：将论文标题[{item_name}]保存到Milvus集合[{collection_name}]"
    )

    try:
        # 获取Milvus单例客户端，连接失败则直接返回
        client = get_milvus_client()
        if not client:
            logger.error("无法获取Milvus客户端（连接失败），跳过数据保存")
            return

        # 集合初始化：不存在则创建（定义Schema+索引），存在则直接使用
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus集合[{collection_name}]不存在，开始创建Schema和索引")
            # 创建集合Schema：自增主键+动态字段，适配灵活的数据存储
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            # 添加自增主键字段：INT64类型，唯一标识每条数据
            schema.add_field(
                field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True
            )
            # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
            schema.add_field(
                field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
            )
            # 添加论文标题字段：VARCHAR类型，最大长度65535，去重依据
            schema.add_field(
                field_name="paper_title", datatype=DataType.VARCHAR, max_length=65535
            )
            schema.add_field(
                field_name="venue", datatype=DataType.VARCHAR, max_length=64
            )
            schema.add_field(
                field_name="year", datatype=DataType.VARCHAR, max_length=16
            )
            # 添加稠密向量字段：FLOAT_VECTOR，1024维（BGE-M3固定维度）
            schema.add_field(
                field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024
            )
            # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR，变长
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

            # 构建索引参数：为向量字段创建索引，提升检索性能
            index_params = client.prepare_index_params()
            # 优化版稠密向量索引：HNSW + COSINE (恢复最佳性能配置)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                # HNSW (Hierarchical Navigable Small World) 是目前性能最好、最常用的基于图的索引，检索速度极快，精度极高。
                index_type="HNSW",
                # 使用 COSINE 作为稠密向量相似度计算方式
                metric_type="COSINE",
                # M: 图中每个节点的最大连接数(常用16-64)
                # efConstruction: 构建索引时的搜索范围(越大建索引越慢，但精度越高，常用100-200)
                # 不同数据体量的推荐建议(万级)：
                # 10000 条数据：M=16, efConstruction=200
                # 50000 条数据：M=32, efConstruction=300
                # 100000 条数据：M=64, efConstruction=400
                params={"M": 16, "efConstruction": 200},
            )

            # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+IP，关闭量化保证精度
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，特点是大部分元素为 0，只有少数维度有值）设计的倒排索引，是稀疏向量检索的标配索引类型。
                index_type="SPARSE_INVERTED_INDEX",
                # IP（内积，Inner Product）如果向量是 “文本语义向量 + 关键词权重”，长度代表文本与主题的关联强度，此时用 IP 能同时体现 “语义匹配度” 和 “关联强度”。
                metric_type="IP",
                # DAAT_MAXSCORE：稀疏向量检索时，只计算可能得高分的维度，跳过大量0值，速度更快。
                # quantization="none"：稀疏向量里的权重是小数，不做压缩，保证精度不丢。
                params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"},
            )

            # 创建集合：Schema + 索引参数
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引")

        # 幂等性处理：删除同名论文标题数据，避免重复存储（核心：先加载集合才能删除）
        clean_paper_title = (item_name or "").strip()
        if clean_paper_title:
            client.load_collection(collection_name=collection_name)
            # 论文标题转义，防止特殊字符导致过滤表达式解析失败
            safe_paper_title = escape_milvus_string(clean_paper_title)
            filter_expr = f'paper_title=="{safe_paper_title}"'
            # 执行删除操作
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(
                f"Milvus幂等性处理完成，已删除集合中[{clean_paper_title}]的历史数据"
            )

        # 构造插入Milvus的数据：基础字段+非空向量字段
        paper_meta = state.get("paper_metadata") or {}
        data = {
            "file_title": file_title,
            "paper_title": item_name,
            "venue": _clean_text(str(paper_meta.get("venue", ""))),
            "year": _clean_text(str(paper_meta.get("year", ""))),
        }
        # 稠密向量非空才添加，避免空值入库报错
        if dense_vector is not None:
            data["dense_vector"] = dense_vector
        # 稀疏向量非空则归一化后添加，保证检索准确性
        if sparse_vector is not None:
            data["sparse_vector"] = sparse_vector

        # 插入数据：列表格式支持批量插入，单条数据保持格式统一
        client.insert(collection_name=collection_name, data=[data])
        # 插入后强制加载集合，确保数据立即可查、Attu可视化界面可见
        client.load_collection(collection_name=collection_name)

        # 最终同步论文标题到全局状态
        state["item_name"] = item_name
        if "paper_title" not in state:
            state["paper_title"] = item_name
        logger.info(
            f"步骤6：论文标题[{item_name}]成功存入Milvus集合[{collection_name}]，数据：{list(data.keys())}"
        )

    # 捕获所有Milvus操作异常：连接中断、入库失败、索引错误等，不中断主流程
    except Exception as e:
        logger.error(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)


def node_paper_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【论文信息识别】{node_name}")
    add_running_task(state.get("task_id", ""), node_name)

    try:
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            return state

        context = step_2_build_context(chunks)
        paper_meta = step_3_call_llm(file_title, context)
        step_4_update_chunks(state, chunks, paper_meta)

        paper_title = paper_meta["paper_title"]
        dense_vector, sparse_vector = step_5_generate_vectors(paper_title)
        step_6_save_to_milvus(
            state, file_title, paper_title, dense_vector, sparse_vector
        )

        logger.info(
            f">>> 核心节点执行完成：【论文信息识别】{node_name}，论文标题：{paper_title}"
        )
    except Exception as exc:
        logger.error(
            f">>> 核心节点执行失败：【论文信息识别】{node_name}，错误信息：{exc}",
            exc_info=True,
        )
        file_title = state.get("file_title", "")
        fallback_title = _fallback_title(file_title)
        file_meta = _extract_file_metadata(file_title)
        state["item_name"] = fallback_title
        state["paper_title"] = fallback_title
        state["paper_metadata"] = {
            "paper_title": fallback_title,
            "authors_text": "",
            "year": file_meta.get("year", ""),
            "venue": file_meta.get("venue", ""),
            "keywords_text": "",
            "abstract": "",
        }
    finally:
        add_done_task(state.get("task_id", ""), node_name)

    return state
