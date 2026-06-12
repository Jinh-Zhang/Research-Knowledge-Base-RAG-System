import os
import sys
import time
import shutil
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_done_task, add_running_task


MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_token
MINERU_POLL_TIMEOUT_SECONDS = max(60, int(mineru_config.poll_timeout_seconds or 1800))
MINERU_POLL_INTERVAL_SECONDS = max(1, int(mineru_config.poll_interval_seconds or 5))


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """Convert a PDF into markdown through the MinerU pipeline."""
    func_name = sys._getframe().f_code.co_name
    logger.debug(f"[{func_name}] node start\nstate={format_state(state)}")
    add_running_task(state["task_id"], func_name)

    try:
        pdf_path_obj, output_dir_obj = step_1_validate_paths(state)
        zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_obj)
        md_path = step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        state["md_path"] = md_path
        logger.info(f"[{func_name}] markdown generated: {md_path}")

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
            logger.debug(
                f"[{func_name}] markdown loaded, chars={len(state['md_content'])}"
            )
        except Exception as e:
            logger.error(f"[{func_name}] failed to read markdown content: {e}")

        logger.info(f"[{func_name}] node finished, state_keys={list(state.keys())}")
    except Exception as e:
        logger.error(f"[{func_name}] PDF to markdown failed: {e}", exc_info=True)
        raise
    finally:
        add_done_task(state["task_id"], func_name)
        logger.debug(f"[{func_name}] node end\nstate={format_state(state)}")

    return state


def step_1_validate_paths(state: ImportGraphState):
    """Validate pdf path and output directory."""
    log_prefix = "[step_1_validate_paths]"
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()

    if not pdf_path:
        raise ValueError(f"{log_prefix} missing pdf_path")
    if not local_dir:
        raise ValueError(f"{log_prefix} missing local_dir")

    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"{log_prefix} pdf file does not exist: {pdf_path_obj}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"{log_prefix} path is not a file: {pdf_path_obj}")

    if not output_dir_obj.exists():
        logger.info(f"{log_prefix} create output dir: {output_dir_obj}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, output_dir_obj


def step_2_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path):
    """Upload the PDF to MinerU and poll until the extraction result is ready."""
    del output_dir_obj

    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError(
            "MinerU config is incomplete. Please set MINERU_BASE_URL and MINERU_API_TOKEN."
        )

    logger.info(f"[mineru] start processing pdf: {pdf_path_obj.name}")
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}",
    }

    upload_url = f"{MINERU_BASE_URL}/file-urls/batch"
    upload_payload = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm",
    }
    logger.debug(f"[mineru] request upload url: url={upload_url} payload={upload_payload}")
    resp = requests.post(
        url=upload_url,
        headers=request_headers,
        json=upload_payload,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"[mineru] failed to get upload url: status={resp.status_code} body={resp.text}"
        )

    resp_data = resp.json()
    if resp_data.get("code") != 0:
        raise RuntimeError(f"[mineru] upload url api error: {resp_data}")

    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]
    logger.info(f"[mineru] upload url ready: batch_id={batch_id}")

    logger.info(f"[mineru] reading pdf bytes: {pdf_path_obj.name}")
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()

    upload_session = requests.Session()
    upload_session.trust_env = False
    try:
        put_resp = upload_session.put(url=signed_url, data=file_data, timeout=60)
        if put_resp.status_code != 200:
            logger.warning(
                "[mineru] upload failed on first attempt, retry with application/pdf: "
                f"status={put_resp.status_code}"
            )
            put_resp = upload_session.put(
                url=signed_url,
                data=file_data,
                headers={"Content-Type": "application/pdf"},
                timeout=60,
            )
            if put_resp.status_code != 200:
                raise RuntimeError(
                    "[mineru] upload failed after retry: "
                    f"status={put_resp.status_code} body={put_resp.text}"
                )
        logger.info(f"[mineru] upload finished: {pdf_path_obj.name}")
    except Exception as e:
        raise RuntimeError(f"[mineru] upload request failed: {e}") from e
    finally:
        upload_session.close()

    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    timeout_seconds = MINERU_POLL_TIMEOUT_SECONDS
    poll_interval = MINERU_POLL_INTERVAL_SECONDS
    progress_log_interval = max(30, poll_interval * 6)
    start_time = time.time()
    next_progress_log_at = 0

    logger.info(
        f"[mineru] polling started: batch_id={batch_id} timeout={timeout_seconds}s "
        f"interval={poll_interval}s"
    )

    while True:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time > timeout_seconds:
            raise TimeoutError(
                f"[任务轮询] 超时！任务处理超{int(timeout_seconds)}秒，batch_id：{batch_id}"
            )

        try:
            poll_resp = requests.get(url=poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(
                f"[mineru] polling request failed, retry after {poll_interval}s: {e}"
            )
            time.sleep(poll_interval)
            continue

        if poll_resp.status_code != 200:
            if 500 <= poll_resp.status_code < 600:
                logger.warning(
                    "[mineru] polling got server error, retry after "
                    f"{poll_interval}s: status={poll_resp.status_code}"
                )
                time.sleep(poll_interval)
                continue
            raise RuntimeError(
                f"[mineru] polling failed: status={poll_resp.status_code} body={poll_resp.text}"
            )

        poll_data = poll_resp.json()
        if poll_data.get("code") != 0:
            raise RuntimeError(f"[mineru] polling api error: {poll_data}")

        extract_results = (poll_data.get("data") or {}).get("extract_result") or []
        if not extract_results:
            if elapsed_time >= next_progress_log_at:
                logger.info(
                    f"[mineru] waiting for result: batch_id={batch_id} elapsed={elapsed_time}s"
                )
                next_progress_log_at = elapsed_time + progress_log_interval
            time.sleep(poll_interval)
            continue

        result_item = extract_results[0]
        state_status = result_item.get("state", "unknown")

        if state_status == "done":
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError(
                    f"[mineru] task finished without full_zip_url: batch_id={batch_id}"
                )
            logger.info(
                f"[mineru] polling finished: batch_id={batch_id} elapsed={elapsed_time}s"
            )
            logger.info(f"[mineru] result zip url: {full_zip_url}")
            return full_zip_url

        if state_status == "failed":
            err_msg = result_item.get("err_msg", "unknown error")
            raise RuntimeError(
                f"[mineru] task failed: batch_id={batch_id} error={err_msg}"
            )

        if elapsed_time >= next_progress_log_at:
            logger.info(
                f"[mineru] task still running: batch_id={batch_id} "
                f"state={state_status} elapsed={elapsed_time}s"
            )
            next_progress_log_at = elapsed_time + progress_log_interval

        time.sleep(poll_interval)


def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
    """Download the MinerU zip result, extract it, and locate the markdown file."""
    logger.info(f"===== start processing MinerU result for [{pdf_stem}] =====")

    logger.info(f"[step_3:1/4] download zip: {zip_url}")
    download_session = requests.Session()
    download_session.trust_env = False
    try:
        resp = download_session.get(zip_url, timeout=120)
    finally:
        download_session.close()

    if resp.status_code != 200:
        raise RuntimeError(
            f"[step_3:1/4] zip download failed: status={resp.status_code}"
        )

    zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"[step_3:1/4] zip saved: {zip_save_path}")

    logger.info("[step_3:2/4] extract zip")
    extract_target_dir = output_dir_obj / pdf_stem
    if extract_target_dir.exists():
        try:
            shutil.rmtree(extract_target_dir)
            logger.info(f"[step_3:2/4] removed old extract dir: {extract_target_dir}")
        except Exception as e:
            logger.warning(f"[step_3:2/4] failed to remove old extract dir: {e}")

    extract_target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
        zip_file_obj.extractall(extract_target_dir)
    logger.info(f"[step_3:2/4] zip extracted: {extract_target_dir}")

    logger.info("[step_3:3/4] search markdown files")
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(
            f"[step_3:3/4] no markdown file found under: {extract_target_dir}"
        )
    logger.info(f"[step_3:3/4] found markdown files: {len(md_file_list)}")

    target_md_file = None
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"[step_3:4/4] matched same-name markdown: {target_md_file.name}")
            break

    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"[step_3:4/4] matched full.md: {target_md_file.name}")
                break

    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"[step_3:4/4] fallback markdown selected: {target_md_file.name}")

    if target_md_file.stem != pdf_stem:
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            target_md_file.rename(new_md_path)
            target_md_file = new_md_path
            logger.info(f"[step_3:4/4] markdown renamed to: {target_md_file.name}")
        except OSError as e:
            logger.warning(f"[step_3:4/4] failed to rename markdown: {e}")

    final_md_path = str(target_md_file.absolute())
    logger.info(f"===== MinerU result processed for [{pdf_stem}]: {final_md_path} =====")
    return final_md_path


if __name__ == "__main__":
    logger.info("===== start node_pdf_to_md unit test =====")

    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"project root: {PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hl3040缃戠粶璇存槑涔?pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    node_pdf_to_md(test_state)
    logger.info("===== finish node_pdf_to_md unit test =====")
