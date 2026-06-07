from typing import List
from typing_extensions import TypedDict


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """

    session_id: str
    user_id: str
    original_query: str

    # 检索阶段中间数据
    embedding_chunks: list
    hyde_embedding_chunks: list
    kg_chunks: list
    web_search_docs: list

    # 排序阶段数据
    rrf_chunks: list
    reranked_docs: list

    # 生成阶段数据
    prompt: str
    answer: str
    answer_prefix: str
    answer_suffix: str
    requested_titles: List[str]
    fallback_to_web_only: bool

    # 辅助信息
    paper_titles: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
