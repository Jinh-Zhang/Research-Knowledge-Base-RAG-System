from pathlib import Path
import asyncio
import logging
import sys
import uuid
import uvicorn
from typing import Dict, Optional
from fastapi import (
    FastAPI,
    BackgroundTasks,
    HTTPException,
    Request,
    Response,
    Cookie,
    Header,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app


class _IgnoreWindowsAsyncioDisconnectFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if "_ProactorBasePipeTransport._call_connection_lost" in message:
            return False

        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
            return False

        return True


def _configure_windows_asyncio():
    if not sys.platform.startswith("win"):
        return

    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio_logger = logging.getLogger("asyncio")
    if not any(isinstance(f, _IgnoreWindowsAsyncioDisconnectFilter) for f in asyncio_logger.filters):
        asyncio_logger.addFilter(_IgnoreWindowsAsyncioDisconnectFilter())


_configure_windows_asyncio()

# 后续导入启动图对象
# from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# 返回chat.html页面
@app.get("/chat.html")  # 对外访问地址
async def chat():
    # 从 api -> query_process
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    # 定义chat.html位置
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    # 如果不存在，抛出404异常
    if not chat_html_path.exists():
        raise HTTPException(
            status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！"
        )
    return FileResponse(chat_html_path)


AUTH_COOKIE_NAME = "kb_auth_token"


def _extract_bearer_token(authorization: Optional[str]) -> str:
    """从 Authorization 头中提取 Bearer token。"""
    if not authorization:
        return ""
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix) :].strip()
    return ""


def require_current_user(
    session_token: str = None, authorization: Optional[str] = None
) -> Dict[str, str]:
    """从 Bearer token 或 cookie 中解析当前登录用户，没有登录则抛出 401。"""
    bearer_token = _extract_bearer_token(authorization)
    user = get_user_by_session_token(bearer_token) if bearer_token else None
    if not user and session_token:
        user = get_user_by_session_token(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或登录已失效")
    return user


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class QueryRequest(BaseModel):
    """查询请求数据结构"""

    query: str = Field(..., description="查询内容")  # ...必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


class SessionCreateRequest(BaseModel):
    session_name: str = Field("", description="会话名称")


class SessionRenameRequest(BaseModel):
    session_name: str = Field(..., description="会话名称")


@app.post("/auth/register")
async def register(payload: LoginRequest, response: Response):
    try:
        user = create_user(payload.username, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = create_login_session(user["user_id"], user["username"])
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )
    return {"user": user, "session_token": token}


@app.post("/auth/login")
async def login(payload: LoginRequest, response: Response):
    user = verify_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_login_session(user["user_id"], user["username"])
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )
    return {"user": user, "session_token": token}


@app.post("/auth/logout")
async def logout(response: Response, kb_auth_token: str = Cookie(None)):
    delete_login_session(kb_auth_token or "")
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return {"message": "已退出登录"}


@app.get("/auth/me")
async def auth_me(
    kb_auth_token: str = Cookie(None), authorization: Optional[str] = Header(None)
):
    user = require_current_user(kb_auth_token, authorization)
    return {"user": user}


@app.post("/query")
async def query(
    background_tasks: BackgroundTasks,
    request: QueryRequest,
    kb_auth_token: str = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    current_user = require_current_user(kb_auth_token, authorization)
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    print(
        "开始处理流程... 是否流式:",
        is_stream,
        f"其他参数:{user_query}, session_id:{session_id}",
    )

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(
            run_query_graph, current_user["user_id"], session_id, user_query, is_stream
        )
        # 返回结果
        print("开始处理结果....")
        return {"message": "结果正在处理中...", "session_id": session_id}
    else:
        # 同步运行
        run_query_graph(current_user["user_id"], session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        image_urls = get_task_result(session_id, "image_urls", [])
        image_infos = get_task_result(session_id, "image_infos", [])
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "user_id": current_user["user_id"],
            "answer": answer,
            "image_urls": image_urls,
            "image_infos": image_infos,
            "done_list": [],
        }


# 定义查询接口
def run_query_graph(
    user_id: str, session_id: str, user_query: str, is_stream: bool = True
):
    print(
        f"开始流程图处理... user={user_id} session={session_id} {user_query} {is_stream}"
    )

    default_state = {
        "original_query": user_query,
        "session_id": session_id,
        "user_id": user_id,
        "is_stream": is_stream,
    }
    try:
        # 后期运行
        query_app.invoke(default_state)
        # 整体任务就更新完了！ 接下来就是数据的更新了！
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print("调用流式/stream...")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


@app.get("/history/{session_id}")
async def history(
    session_id: str,
    limit: int = 50,
    kb_auth_token: str = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """
    查询当前会话历史记录
    """
    current_user = require_current_user(kb_auth_token, authorization)
    try:
        records = get_recent_messages(
            session_id, user_id=current_user["user_id"], limit=limit
        )
        items = []
        for r in records:
            items.append(
                {
                    "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                    "user_id": r.get("user_id", ""),
                    "session_id": r.get("session_id", ""),
                    "role": r.get("role", ""),
                    "text": r.get("text", ""),
                    "rewritten_query": r.get("rewritten_query", ""),
                    "paper_titles": r.get("paper_titles", []),
                    "image_urls": r.get("image_urls", []),
                    "image_infos": r.get("image_infos", []),
                    "ts": r.get("ts"),
                }
            )
        return {
            "session_id": session_id,
            "user_id": current_user["user_id"],
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.get("/sessions")
async def sessions(
    kb_auth_token: str = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    current_user = require_current_user(kb_auth_token, authorization)
    records = list_chat_sessions(current_user["user_id"], limit=50)
    items = []
    for r in records:
        items.append(
            {
                "session_id": r.get("session_id", ""),
                "session_name": r.get("session_name", "") or "未命名会话",
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
                "message_count": r.get("message_count", 0),
            }
        )
    return {"items": items}


@app.post("/sessions")
async def create_session(
    payload: SessionCreateRequest,
    kb_auth_token: str = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    current_user = require_current_user(kb_auth_token, authorization)
    session_id = str(uuid.uuid4())
    session_name = (payload.session_name or "").strip()
    if not session_name:
        session_name = f"新会话{len(list_chat_sessions(current_user['user_id'], limit=200)) + 1}"
    doc = create_or_update_chat_session(
        current_user["user_id"],
        session_id,
        session_name=session_name,
    )
    return {
        "session_id": session_id,
        "session_name": doc.get("session_name", "") or "未命名会话",
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


@app.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    payload: SessionRenameRequest,
    kb_auth_token: str = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    current_user = require_current_user(kb_auth_token, authorization)
    try:
        rename_chat_session(current_user["user_id"], session_id, payload.session_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, "session_name": payload.session_name.strip()}


@app.delete("/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    kb_auth_token: str = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    current_user = require_current_user(kb_auth_token, authorization)
    count = clear_history(session_id, current_user["user_id"])
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
