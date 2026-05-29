from typing import TypedDict
import copy

from app.core.logger import logger


class ImportGraphState(TypedDict):
    """
    Shared state for the import graph.
    """

    task_id: str

    # Flow control flags.
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    is_normalize_to_md_enabled: bool
    source_format: str

    # Chunk/split flags.
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # Paths.
    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str
    split_path: str
    embeddings_path: str

    # Content.
    md_content: str
    chunks: list
    item_name: str
    paper_title: str
    paper_metadata: dict

    # Storage.
    embeddings_content: list


graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normalize_to_md_enabled": False,
    "source_format": "",
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "paper_title": "",
    "paper_metadata": {},
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    state = create_default_state(local_file_path="example.pdf")
    logger.info(state)
