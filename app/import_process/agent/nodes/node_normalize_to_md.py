import os
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


WORD_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

NORMALIZE_EXTENSIONS = {".txt", ".docx"}


def node_normalize_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    Normalize txt/docx source files into markdown so downstream nodes can stay unchanged.
    """
    node_name = sys._getframe().f_code.co_name
    add_running_task(state["task_id"], node_name)

    try:
        source_path = Path(state.get("local_file_path") or state.get("md_path") or "")
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        ext = source_path.suffix.lower()
        if ext not in NORMALIZE_EXTENSIONS:
            raise ValueError(f"Unsupported normalize format: {ext}")

        state["source_format"] = ext.lstrip(".")
        state["is_md_read_enabled"] = True

        if ext == ".txt":
            md_path, md_content = _normalize_txt_to_md(source_path, state)
        else:
            md_path, md_content = _normalize_docx_to_md(source_path, state)

        state["md_path"] = md_path
        state["md_content"] = md_content
        logger.info(f"Normalized source file to markdown: {source_path} -> {md_path}")
        return state
    finally:
        add_done_task(state["task_id"], node_name)


def _read_text_with_fallbacks(path: Path) -> str:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        getattr(last_error, "encoding", "utf-8"),
        getattr(last_error, "object", b""),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 1),
        getattr(last_error, "reason", "unable to decode input file"),
    )


def _normalize_txt_to_md(source_path: Path, state: ImportGraphState) -> Tuple[str, str]:
    raw_text = _read_text_with_fallbacks(source_path)
    body = _normalize_plain_text(raw_text)
    md_content = f"# {state.get('file_title') or source_path.stem}\n\n{body}\n"
    md_path = _build_md_output_path(source_path, state)
    _write_md_output(md_path, md_content)
    return md_path, md_content


def _normalize_docx_to_md(source_path: Path, state: ImportGraphState) -> Tuple[str, str]:
    md_path = _build_md_output_path(source_path, state)
    images_dir = Path(md_path).parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_path, "r") as zf:
        document_xml = zf.read("word/document.xml")
        rel_map = _load_docx_relationships(zf)
        root = ET.fromstring(document_xml)
        body = root.find("w:body", WORD_NS)
        blocks: List[str] = [f"# {state.get('file_title') or source_path.stem}"]

        if body is not None:
            for child in body:
                local_name = _local_name(child.tag)
                if local_name == "p":
                    block = _docx_paragraph_to_md(child, rel_map, zf, images_dir)
                elif local_name == "tbl":
                    block = _docx_table_to_md(child)
                else:
                    block = ""
                if block:
                    blocks.append(block)

    md_content = "\n\n".join(part for part in blocks if part).strip() + "\n"
    _write_md_output(md_path, md_content)
    return md_path, md_content


def _build_md_output_path(source_path: Path, state: ImportGraphState) -> str:
    local_dir = Path(state.get("local_dir") or source_path.parent)
    local_dir.mkdir(parents=True, exist_ok=True)
    return str((local_dir / f"{source_path.stem}.md").resolve())


def _write_md_output(md_path: str, md_content: str) -> None:
    Path(md_path).write_text(md_content, encoding="utf-8")


def _normalize_plain_text(raw_text: str) -> str:
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines: List[str] = []
    previous_blank = False
    for line in lines:
        current = line.rstrip()
        is_blank = not current.strip()
        if is_blank:
            if previous_blank:
                continue
            normalized_lines.append("")
        else:
            normalized_lines.append(current)
        previous_blank = is_blank
    return "\n".join(normalized_lines).strip() or "(empty document)"


def _load_docx_relationships(zf: zipfile.ZipFile) -> Dict[str, str]:
    rel_path = "word/_rels/document.xml.rels"
    if rel_path not in zf.namelist():
        return {}

    rel_root = ET.fromstring(zf.read(rel_path))
    rel_map: Dict[str, str] = {}
    for rel in rel_root.findall("rel:Relationship", WORD_NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rel_map[rel_id] = target
    return rel_map


def _docx_paragraph_to_md(
    paragraph: ET.Element,
    rel_map: Dict[str, str],
    zf: zipfile.ZipFile,
    images_dir: Path,
) -> str:
    text = _extract_paragraph_text(paragraph).strip()
    style = _extract_paragraph_style(paragraph)
    images = _extract_paragraph_images(paragraph, rel_map, zf, images_dir)

    blocks: List[str] = []
    if text:
        heading_level = _style_to_heading_level(style)
        if heading_level:
            blocks.append(f"{'#' * heading_level} {text}")
        else:
            blocks.append(text)
    blocks.extend(images)
    return "\n\n".join(blocks)


def _extract_paragraph_text(paragraph: ET.Element) -> str:
    parts: List[str] = []
    for node in paragraph.iter():
        local_name = _local_name(node.tag)
        if local_name == "t":
            parts.append(node.text or "")
        elif local_name == "tab":
            parts.append("\t")
        elif local_name in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def _extract_paragraph_style(paragraph: ET.Element) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", WORD_NS)
    if style is None:
        return ""
    return style.attrib.get(f"{{{WORD_NS['w']}}}val", "") or style.attrib.get("val", "")


def _style_to_heading_level(style: str) -> int:
    style_lower = (style or "").lower()
    if style_lower == "title":
        return 1
    if style_lower == "subtitle":
        return 2
    if style_lower.startswith("heading"):
        suffix = style_lower.replace("heading", "", 1)
        if suffix.isdigit():
            return max(1, min(int(suffix), 6))
    return 0


def _extract_paragraph_images(
    paragraph: ET.Element,
    rel_map: Dict[str, str],
    zf: zipfile.ZipFile,
    images_dir: Path,
) -> List[str]:
    image_blocks: List[str] = []
    for blip in paragraph.findall(".//a:blip", WORD_NS):
        rel_id = blip.attrib.get(f"{{{WORD_NS['r']}}}embed") or blip.attrib.get("embed")
        if not rel_id:
            continue

        target = rel_map.get(rel_id, "")
        if not target:
            continue

        member_name = _resolve_docx_member_path(target)
        if not member_name or member_name not in zf.namelist():
            continue

        image_name = Path(member_name).name
        output_path = images_dir / image_name
        if not output_path.exists():
            with zf.open(member_name) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        image_blocks.append(f"![{image_name}](images/{image_name})")
    return image_blocks


def _resolve_docx_member_path(target: str) -> str:
    normalized = target.replace("\\", "/").lstrip("/")
    if normalized.startswith("word/"):
        return normalized
    return f"word/{normalized}"


def _docx_table_to_md(table: ET.Element) -> str:
    rows: List[List[str]] = []
    for tr in table.findall("./w:tr", WORD_NS):
        cells: List[str] = []
        for tc in tr.findall("./w:tc", WORD_NS):
            paragraphs = tc.findall("./w:p", WORD_NS)
            texts = [_extract_paragraph_text(p) for p in paragraphs]
            cell_text = " ".join(text for text in texts if text).strip()
            cells.append(_sanitize_table_cell(cell_text))
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    header = normalized_rows[0]
    body = normalized_rows[1:] or [[""] * max_cols]

    lines = [
        _markdown_table_row(header),
        _markdown_table_row(["---"] * max_cols),
    ]
    lines.extend(_markdown_table_row(row) for row in body)
    return "\n".join(lines)


def _sanitize_table_cell(text: str) -> str:
    return (text or "").replace("\n", "<br>").replace("|", "\\|")


def _markdown_table_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _local_name(tag: str) -> str:
    if "}" not in tag:
        return tag
    return tag.split("}", 1)[1]
