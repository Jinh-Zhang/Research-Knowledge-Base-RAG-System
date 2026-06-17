# 导入系统模块：用于读取环境变量
import os
import secrets
import hashlib

# 导入日志模块：用于记录程序运行日志（成功/失败/错误信息）
import logging

# 导入类型注解模块：用于函数参数/返回值的类型提示，提升代码可读性和规范性
from typing import List, Dict, Any, Optional

# 导入时间模块：用于生成时间戳，记录对话的创建时间
from datetime import datetime

# 导入pymongo核心模块：MongoDB原生Python驱动，实现数据库连接和操作
# ASCENDING：表示升序排序，用于MongoDB索引和查询排序
from pymongo import MongoClient, ASCENDING

# 导入bson的ObjectId：MongoDB默认的主键类型，用于唯一标识文档
from bson import ObjectId

# 导入dotenv模块：用于从.env文件加载环境变量，避免硬编码敏感配置（如MongoDB连接地址）
from dotenv import load_dotenv

# 加载.env文件中的环境变量，使os.getenv能读取到配置
load_dotenv()


class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具类 (基于原生 PyMongo 实现)
    核心功能：封装MongoDB的连接、集合初始化、索引创建，为上层提供统一的数据库操作入口
    扩展功能：支持与LangChain消息对象的格式转换（原代码预留能力）
    """

    def __init__(self):
        """
        类初始化方法：完成MongoDB的连接、数据库/集合获取、索引创建
        初始化失败会抛出异常并记录错误日志，确保程序感知连接问题
        """
        try:
            # 从环境变量读取MongoDB连接地址（敏感配置，不硬编码）
            self.mongo_url = os.getenv("MONGO_URL")
            # 从环境变量读取要使用的数据库名称
            self.db_name = os.getenv("MONGO_DB_NAME")

            # 创建MongoDB客户端实例，建立与数据库的连接
            self.client = MongoClient(self.mongo_url)
            # 获取指定名称的数据库对象
            self.db = self.client[self.db_name]
            # 获取对话记录的集合（相当于关系型数据库的表），集合名：chat_message
            self.chat_message = self.db["chat_message"]
            self.chat_session = self.db["chat_session"]
            # 获取用户账户集合
            self.user_account = self.db["user_account"]
            # 获取登录会话集合
            self.user_session = self.db["user_session"]

            # 为chat_message集合创建复合索引，提升查询性能
            # 索引规则：user_id升序 + session_id升序 + ts降序，适配"按用户会话查最新记录"的核心查询场景
            # create_index自带幂等性：索引已存在时不会重复创建，无需额外判断
            self.chat_message.create_index(
                [("user_id", 1), ("session_id", 1), ("ts", -1)]
            )
            self.chat_session.create_index(
                [("user_id", 1), ("updated_at", -1)]
            )
            self.chat_session.create_index(
                [("user_id", 1), ("session_id", 1)], unique=True
            )
            self.user_account.create_index([("username", 1)], unique=True)
            self.user_session.create_index([("session_token", 1)], unique=True)
            self.user_session.create_index([("user_id", 1), ("created_at", -1)])

            # 记录成功日志，确认数据库连接和初始化完成
            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            # 捕获所有初始化异常，记录详细错误日志
            logging.error(f"Failed to connect to MongoDB: {e}")
            # 重新抛出异常，让调用方感知初始化失败，避免使用未初始化的实例
            raise


# 定义全局变量：存储HistoryMongoTool的单例实例
# 作用：避免多次创建HistoryMongoTool实例，从而避免重复建立MongoDB连接
_history_mongo_tool = None
# 模块加载时尝试初始化单例实例，实现预加载
# 目的：将数据库连接的初始化提前到模块加载阶段，避免第一次调用接口时才建立连接（提升首次响应速度）
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    # 初始化失败时仅记录警告日志，不抛出异常
    # 原因：模块加载阶段的异常可能导致整个程序启动失败，此处保留懒加载兜底（get_history_mongo_tool会再次尝试创建）
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取HistoryMongoTool的单例实例（懒加载模式）
    核心逻辑：全局实例为空时创建，不为空时直接返回，保证整个程序只有一个数据库连接实例
    :return: HistoryMongoTool的单例实例
    """
    # 声明使用全局变量，避免函数内视为局部变量
    global _history_mongo_tool
    # 懒加载：仅当全局实例为空时，才创建新的实例
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    # 返回单例实例
    return _history_mongo_tool


def _normalize_username(username: str) -> str:
    """统一用户名格式，避免大小写和首尾空白导致重复账户。"""
    return (username or "").strip().lower()


def _hash_password(password: str, salt: str) -> str:
    """使用标准库生成带盐密码摘要，避免明文存储密码。"""
    raw = f"{salt}:{password}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def create_user(username: str, password: str) -> Dict[str, str]:
    """创建本地账号，用户名唯一。"""
    normalized_username = _normalize_username(username)
    if not normalized_username:
        raise ValueError("用户名不能为空")
    if not password or len(password) < 6:
        raise ValueError("密码长度不能少于6位")

    mongo_tool = get_history_mongo_tool()
    if mongo_tool.user_account.find_one({"username": normalized_username}):
        raise ValueError("用户名已存在")

    salt = secrets.token_hex(16)
    document = {
        "username": normalized_username,
        "password_salt": salt,
        "password_hash": _hash_password(password, salt),
        "created_at": datetime.now().timestamp(),
    }
    result = mongo_tool.user_account.insert_one(document)
    return {"user_id": str(result.inserted_id), "username": normalized_username}


def verify_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """校验用户名密码，成功返回用户信息。"""
    normalized_username = _normalize_username(username)
    mongo_tool = get_history_mongo_tool()
    user = mongo_tool.user_account.find_one({"username": normalized_username})
    if not user:
        return None

    expected_hash = _hash_password(password, user.get("password_salt", ""))
    if expected_hash != user.get("password_hash"):
        return None

    return {"user_id": str(user["_id"]), "username": user["username"]}


def create_login_session(user_id: str, username: str) -> str:
    """创建登录会话 token，供 cookie 鉴权使用。"""
    token = secrets.token_urlsafe(32)
    mongo_tool = get_history_mongo_tool()
    mongo_tool.user_session.insert_one(
        {
            "user_id": user_id,
            "username": username,
            "session_token": token,
            "created_at": datetime.now().timestamp(),
        }
    )
    return token


def get_user_by_session_token(session_token: str) -> Optional[Dict[str, str]]:
    """根据登录 token 获取当前用户。"""
    if not session_token:
        return None

    mongo_tool = get_history_mongo_tool()
    session = mongo_tool.user_session.find_one({"session_token": session_token})
    if not session:
        return None

    return {
        "user_id": session.get("user_id", ""),
        "username": session.get("username", ""),
    }


def delete_login_session(session_token: str) -> int:
    """删除登录会话 token。"""
    if not session_token:
        return 0
    mongo_tool = get_history_mongo_tool()
    result = mongo_tool.user_session.delete_many({"session_token": session_token})
    return result.deleted_count


def clear_history(session_id: str, user_id: str) -> int:
    """
    清空指定会话的所有历史对话记录
    :param session_id: 会话唯一标识，用于筛选要删除的记录
    :param user_id: 当前登录用户ID，用于隔离不同账户数据
    :return: 实际删除的文档数量，删除失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式避免重复创建数据库连接
    mongo_tool = get_history_mongo_tool()
    try:
        # 执行批量删除操作：删除所有session_id匹配的文档
        result = mongo_tool.chat_message.delete_many(
            {"session_id": session_id, "user_id": user_id}
        )
        mongo_tool.chat_session.delete_one(
            {"session_id": session_id, "user_id": user_id}
        )
        # 记录删除成功日志，包含删除数量和会话ID，便于问题排查
        logging.info(
            f"Deleted {result.deleted_count} messages for user {user_id}, session {session_id}"
        )
        # 返回实际删除的数量（delete_many的返回对象包含deleted_count属性）
        return result.deleted_count
    except Exception as e:
        # 捕获删除异常，记录错误日志，包含会话ID
        logging.error(
            f"Error clearing history for user {user_id}, session {session_id}: {e}"
        )
        # 异常时返回0，标识删除失败
        return 0


def create_or_update_chat_session(
    user_id: str,
    session_id: str,
    session_name: str = "",
    first_message: str = "",
) -> Dict[str, Any]:
    """创建或更新会话元数据。"""
    mongo_tool = get_history_mongo_tool()
    now = datetime.now().timestamp()
    existing = mongo_tool.chat_session.find_one(
        {"user_id": user_id, "session_id": session_id}
    )

    fallback_name = (first_message or "").strip()
    if fallback_name:
        fallback_name = fallback_name[:30]
    name = (session_name or "").strip() or (existing or {}).get("session_name") or fallback_name or "未命名会话"

    update = {
        "$set": {
            "session_name": name,
            "updated_at": now,
        },
        "$setOnInsert": {
            "user_id": user_id,
            "session_id": session_id,
            "created_at": now,
        },
    }
    mongo_tool.chat_session.update_one(
        {"user_id": user_id, "session_id": session_id},
        update,
        upsert=True,
    )
    return mongo_tool.chat_session.find_one(
        {"user_id": user_id, "session_id": session_id}
    ) or {}


def rename_chat_session(user_id: str, session_id: str, session_name: str) -> bool:
    """重命名会话。"""
    name = (session_name or "").strip()
    if not name:
        raise ValueError("会话名称不能为空")
    mongo_tool = get_history_mongo_tool()
    result = mongo_tool.chat_session.update_one(
        {"user_id": user_id, "session_id": session_id},
        {
            "$set": {
                "session_name": name[:80],
                "updated_at": datetime.now().timestamp(),
            },
            "$setOnInsert": {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now().timestamp(),
            },
        },
        upsert=True,
    )
    return result.modified_count > 0 or result.upserted_id is not None


def list_chat_sessions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """返回当前用户的会话列表，兼容旧数据。"""
    mongo_tool = get_history_mongo_tool()
    try:
        sessions = list(
            mongo_tool.chat_session.find({"user_id": user_id})
            .sort("updated_at", -1)
            .limit(limit)
        )
        seen = {s.get("session_id") for s in sessions}

        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": "$session_id",
                    "updated_at": {"$max": "$ts"},
                    "created_at": {"$min": "$ts"},
                    "message_count": {"$sum": 1},
                    "first_text": {"$first": "$text"},
                }
            },
            {"$sort": {"updated_at": -1}},
            {"$limit": limit},
        ]
        for item in mongo_tool.chat_message.aggregate(pipeline):
            sid = item.get("_id")
            if not sid or sid in seen:
                continue
            name = (item.get("first_text") or "").strip()[:30] or "历史会话"
            sessions.append(
                {
                    "user_id": user_id,
                    "session_id": sid,
                    "session_name": name,
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "message_count": item.get("message_count", 0),
                }
            )

        sessions.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
        return sessions[:limit]
    except Exception as e:
        logging.error(f"Error listing chat sessions for user {user_id}: {e}")
        return []


def save_chat_message(
    user_id: str,
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    paper_titles: List[str] = None,
    image_urls: List[str] = None,
    image_infos: List[dict] = None,
    message_id: str = None,
) -> str:
    """
    写入/更新单条会话记录到MongoDB
    支持两种模式：无message_id时新增记录，有message_id时更新已有记录
    :param user_id: 当前登录用户ID，用于账户级隔离
    :param session_id: 会话唯一标识，关联对话所属的会话
    :param role: 消息角色，固定值：user（用户）/assistant（助手）
    :param text: 对话核心内容，用户的提问或助手的回答
    :param rewritten_query: 重写后的查询语句（可选，用于检索增强等场景，默认空字符串）
    :param paper_titles: 关联的论文标题列表（可选，支持多篇论文，默认None）
    :param image_urls: 关联的图片URL列表（可选，默认None）
    :param image_infos: 关联的图片结构化信息（可选，默认None）
    :param message_id: 记录主键ID（可选，有值则更新，无值则新增）
    :return: 插入/更新的记录唯一标识（新增返回ObjectId字符串，更新返回传入的message_id）
    """
    # 生成当前时间的时间戳（秒级），用于记录消息的创建时间，后续用于排序和查询
    ts = datetime.now().timestamp()

    # 构造要插入/更新的文档数据（MongoDB的基本数据单元是文档，类似Python字典）
    document = {
        "user_id": user_id,
        "session_id": session_id,  # 会话ID，关联维度
        "role": role,  # 消息角色
        "text": text,  # 消息内容
        "rewritten_query": rewritten_query or "",  # 重写查询，空值处理为空字符串
        "paper_titles": paper_titles,  # 关联论文标题列表
        "image_urls": image_urls,  # 关联图片URL列表
        "image_infos": image_infos,  # 关联图片结构化信息
        "ts": ts,  # 时间戳，排序和时间筛选维度
    }

    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    create_or_update_chat_session(
        user_id=user_id,
        session_id=session_id,
        first_message=text if role == "user" else "",
    )
    # 判断是否传入主键ID，区分更新/新增逻辑
    if message_id:
        # 有message_id：执行更新操作（根据主键更新）
        result = mongo_tool.chat_message.update_one(
            {
                "_id": ObjectId(message_id)
            },  # 更新条件：主键匹配（需将字符串转为ObjectId类型）
            {"$set": document},  # 更新操作：$set表示只更新指定字段，保留其他字段
        )
        # 更新操作返回传入的message_id作为标识
        return message_id
    else:
        # 无message_id：执行新增操作
        result = mongo_tool.chat_message.insert_one(document)
        # 新增操作返回插入的ObjectId并转为字符串，便于上层使用（避免直接返回ObjectId对象）
        return str(result.inserted_id)


def update_message_paper_titles(ids: List[str], paper_titles: List[str]) -> int:
    """
    批量更新历史会话记录的关联论文标题
    :param ids: 要更新的记录主键ID列表（字符串类型）
    :param paper_titles: 要设置的新论文标题列表
    :return: 实际更新的文档数量，更新失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    try:
        # 将字符串类型的主键列表转为MongoDB的ObjectId类型（数据库中主键是ObjectId类型）
        object_ids = [ObjectId(i) for i in ids]
        # 执行批量更新操作
        result = mongo_tool.chat_message.update_many(
            # 更新条件：复合条件，同时满足
            {
                "_id": {"$in": object_ids}  # 主键在指定的ID列表中（批量筛选）
            },
            {"$set": {"paper_titles": paper_titles}},  # 更新操作：设置新的论文标题列表
        )
        # 记录更新成功日志，包含更新数量和新的论文名称
        logging.info(
            f"Updated {result.modified_count} records to paper_titles: {paper_titles}"
        )
        # 返回实际更新的数量（modified_count：真正被修改的文档数，区别于matched_count）
        return result.modified_count
    except Exception as e:
        # 捕获批量更新异常，记录错误日志
        logging.error(f"Error updating history paper_titles: {e}")
        # 异常时返回0，标识更新失败
        return 0


def get_recent_messages(
    session_id: str, user_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近N条对话记录，返回原始字典格式
    结果按时间正序排列，可直接喂给LLM作为上下文
    :param session_id: 会话唯一标识，用于筛选指定会话的记录
    :param user_id: 当前登录用户ID，用于账户级隔离
    :param limit: 条数限制，默认返回最近10条
    :return: 对话记录列表（字典格式），查询失败返回空列表
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    try:
        # 构造查询条件：仅查询指定session_id的记录
        query = {"session_id": session_id, "user_id": user_id}

        # 执行查询：按时间戳升序排序，限制返回条数
        # find(query)：获取符合条件的游标（惰性加载，不立即查询）
        # sort("ts", ASCENDING)：按ts字段升序（从旧到新），适配LLM上下文顺序
        # limit(limit)：限制返回的最大条数
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        # 将游标转为列表，触发实际数据库查询，获取所有符合条件的文档
        messages = list(cursor)

        # 返回查询结果列表
        return messages
    except Exception as e:
        # 捕获查询异常，记录错误日志
        logging.error(f"Error getting recent messages: {e}")
        # 异常时返回空列表，避免上层处理None报错
        return []
