import sys
from os.path import basename, splitext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_done_task, add_running_task


PDF_EXTENSIONS = {".pdf"}
MD_EXTENSIONS = {".md", ".markdown"}
NORMALIZE_TO_MD_EXTENSIONS = {".txt", ".docx"}


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Entry node for the import graph.
    """
    func_name = sys._getframe().f_code.co_name
    logger.debug("[%s] start\n%s", func_name, format_state(state))
    add_running_task(state["task_id"], func_name)

    document_path = state.get("local_file_path", "")
    if not document_path:
        logger.error("[%s] local_file_path is empty", func_name)
        add_done_task(state["task_id"], func_name)
        return state

    ext = splitext(document_path)[1].lower()
    state["source_format"] = ext.lstrip(".")
    state["is_pdf_read_enabled"] = False
    state["is_md_read_enabled"] = False
    state["is_normalize_to_md_enabled"] = False

    if ext in PDF_EXTENSIONS:
        logger.info("[%s] detected PDF input: %s", func_name, document_path)
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = document_path
    elif ext in MD_EXTENSIONS:
        logger.info("[%s] detected markdown input: %s", func_name, document_path)
        state["is_md_read_enabled"] = True
        state["md_path"] = document_path
    elif ext in NORMALIZE_TO_MD_EXTENSIONS:
        logger.info("[%s] detected normalize-to-md input: %s", func_name, document_path)
        state["is_normalize_to_md_enabled"] = True
    else:
        logger.warning(
            "[%s] unsupported file type: %s, only supports %s",
            func_name,
            document_path,
            ", ".join(sorted(PDF_EXTENSIONS | MD_EXTENSIONS | NORMALIZE_TO_MD_EXTENSIONS)),
        )

    state["file_title"] = splitext(basename(document_path))[0]
    logger.info("[%s] file_title=%s", func_name, state["file_title"])

    add_done_task(state["task_id"], func_name)
    logger.debug("[%s] end\n%s", func_name, format_state(state))
    return state


if __name__ == "__main__":
    logger.info("===== node_entry tests =====")

    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="notes.txt",
    )
    node_entry(test_state1)

    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="paper.md",
    )
    node_entry(test_state2)

    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="paper.pdf",
    )
    node_entry(test_state3)
