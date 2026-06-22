# 科研知识库 RAG 系统

一个基于检索增强生成（RAG）技术的科研论文知识库系统，支持论文批量导入、智能检索和对话式查询。

## 功能特性

### 📚 论文导入与处理
- **多源导入**：支持 arXiv、CVF、OpenReview、PapersWithCode 等多个学术数据源
- **批量处理**：支持批量下载和导入科研论文
- **智能解析**：使用 MinerU 进行 PDF 智能解析，转换为结构化 Markdown
- **实体识别**：自动识别论文中的关键实体和概念
- **文档切片**：智能文档分割，保持语义完整性

### 🔍 智能检索
- **向量检索**：基于 BGE-M3 模型的高质量向量化检索
- **重排序**：使用 BGE-Reranker-Large 对检索结果进行重排序
- **混合检索**：结合向量检索和关键词匹配
- **上下文增强**：智能提取相关上下文信息

### 💬 对话式查询
- **问答系统**：基于 LangGraph 的智能问答流程
- **流式输出**：支持实时流式返回查询结果
- **会话管理**：支持多会话管理和历史记录
- **用户认证**：内置用户注册、登录和权限管理

### 📊 评估与优化
- **RAG 评估**：使用 RAGAS 框架进行系统性能评估
- **多维度指标**：支持答案相关性、忠实度、上下文精确度等多维度评估

## 技术架构

### 核心技术栈
- **Web 框架**：FastAPI + Uvicorn
- **向量数据库**：Milvus
- **文档数据库**：MongoDB
- **对象存储**：MinIO
- **PDF 解析**：MinerU (Magic-PDF)
- **语言模型**：支持 OpenAI API 兼容接口（DeepSeek、Qwen 等）
- **Embedding 模型**：BGE-M3
- **重排序模型**：BGE-Reranker-Large
- **AI 框架**：LangChain + LangGraph

### 项目结构

```
knowledge_base/
├── app/                          # 应用核心代码
│   ├── clients/                  # 数据库客户端工具
│   │   ├── milvus_utils.py      # Milvus 向量数据库工具
│   │   ├── mongo_history_utils.py # MongoDB 历史记录工具
│   │   └── minio_utils.py       # MinIO 对象存储工具
│   ├── conf/                     # 配置模块
│   │   ├── embedding_config.py  # Embedding 模型配置
│   │   ├── lm_config.py         # 语言模型配置
│   │   ├── milvus_config.py     # Milvus 配置
│   │   └── reranker_config.py   # 重排序模型配置
│   ├── core/                     # 核心功能
│   │   ├── logger.py            # 日志配置
│   │   └── load_prompt.py       # Prompt 模板加载
│   ├── import_process/           # 论文导入处理流程
│   │   ├── agent/               # 导入流程图
│   │   │   ├── main_graph.py    # 主流程图
│   │   │   ├── nodes/           # 流程节点
│   │   │   └── state.py         # 状态定义
│   │   └── api/                 # 导入 API
│   ├── query_process/            # 查询处理流程
│   │   ├── agent/               # 查询流程图
│   │   │   └── main_graph.py    # 查询主流程
│   │   ├── api/                 # 查询 API
│   │   │   └── query_service.py # FastAPI 查询服务
│   │   └── nodes/               # 查询流程节点
│   ├── lm/                       # 语言模型封装
│   ├── tool/                     # 工具函数
│   └── utils/                    # 通用工具
├── scripts/                      # 脚本工具
│   └── batch_import_papers.py   # 批量导入论文脚本
├── prompts/                      # Prompt 模板
├── result/                       # 评估结果
├── logs/                         # 日志文件
├── .env                          # 环境变量配置
└── pyproject.toml                # 项目依赖配置
```

## 快速开始

### 环境要求

- Python >= 3.11
- CUDA 12.4（如需 GPU 加速）
- Docker（用于运行 Milvus、MongoDB、MinIO）

### 安装依赖

推荐使用 `uv` 进行依赖管理：

```bash
# 安装 uv（如果还没有安装）
pip install uv

# 安装项目依赖
uv sync
```

或使用传统方式：

```bash
pip install -e .
```

### 配置环境变量

复制 `.env` 文件并根据实际情况修改配置：

```bash
cp .env .env.local
```

关键配置项：

```env
# 语言模型配置
LLM_DEFAULT_MODEL=deepseek-v4-flash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1

# Embedding 模型路径
BGE_M3_PATH=/path/to/bge-m3
BGE_DEVICE=cuda:0  # 或 cpu

# Milvus 配置
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=paper_chunks_v2
ITEM_NAME_COLLECTION=paper_item_names_v2

# MongoDB 配置
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=paper_test

# MinIO 配置
MINIO_ENDPOINT=127.0.0.1:9002
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=paper-list

# MinerU PDF 解析服务
MINERU_API_TOKEN=your_token
MINERU_BASE_URL=https://mineru.net/api/v4

# 重排序模型
BGE_RERANKER_LARGE=/path/to/bge-reranker-large
BGE_RERANKER_DEVICE=cuda:0
```

### 启动基础服务

使用 Docker Compose 启动 Milvus、MongoDB 和 MinIO：

```bash
cd volumes
docker-compose up -d
```

或分别启动各服务（请参考各服务官方文档）。

### 下载模型

需要下载以下模型到本地：

1. **BGE-M3** (Embedding 模型)
   ```bash
   # 使用 modelscope 下载
   from modelscope import snapshot_download
   snapshot_download('BAAI/bge-m3', cache_dir='./models')
   ```

2. **BGE-Reranker-Large** (重排序模型)
   ```bash
   snapshot_download('BAAI/bge-reranker-large', cache_dir='./models')
   ```

## 使用指南

### 1. 批量导入论文

使用 `batch_import_papers.py` 脚本从各种来源批量导入论文：

#### 从 arXiv 导入

```bash
python scripts/batch_import_papers.py \
  --source arxiv \
  --query "transformer attention" \
  --max-results 10 \
  --output-dir ./output/arxiv_import
```

#### 从会议论文集导入

```bash
python scripts/batch_import_papers.py \
  --conference cvpr \
  --year 2024 \
  --query "object detection" \
  --max-results 20
```

#### 从 URL 文件批量导入

```bash
# 创建 urls.txt 文件，每行一个论文 PDF URL
python scripts/batch_import_papers.py \
  --source url_file \
  --url-file urls.txt
```

#### 高级选项

```bash
python scripts/batch_import_papers.py \
  --source arxiv \
  --query "deep learning" \
  --max-results 50 \
  --import-batch-size 5     # 每批导入 5 篇
  --sleep-seconds 10        # 批次间隔 10 秒
  --download-only           # 仅下载，不导入
  --list-only              # 仅列出论文，不下载
```

支持的数据源：
- `arxiv`: arXiv 预印本
- `cvpr`, `iccv`, `eccv`: 计算机视觉顶会
- `neurips`, `icml`, `iclr`: 机器学习顶会
- `openreview`: OpenReview 平台
- `paperswithcode`: Papers with Code
- `url_file`: 自定义 URL 列表

### 2. 启动服务

项目提供了三个独立的 FastAPI 服务，均使用 **Uvicorn** ASGI 服务器运行：

#### 查询服务 (端口 8001)

提供论文查询、对话、会话管理等功能：

```bash
python app/query_process/api/query_service.py
```

服务地址：`http://127.0.0.1:8001`

主要功能：
- 用户认证（注册/登录/登出）
- 智能问答（支持流式/非流式）
- 会话管理（创建/重命名/删除）
- 历史记录查询
- Web 聊天界面

#### 导入服务 (端口 8000)

提供论文上传和导入功能：

```bash
python app/import_process/api/file_import_service.py
```

服务地址：`http://127.0.0.1:8000`

主要功能：
- 单文件上传导入
- 批量文件导入
- 导入任务状态查询
- 导入历史管理

#### 知识库管理服务 (端口 8002)

提供知识库删除和预览功能：

```bash
python app/import_process/api/kb_delete_preview_service.py
```

服务地址：`http://127.0.0.1:8002`

主要功能：
- 知识项删除（soft/full 模式）
- 删除预览（dry-run）
- 清理 Milvus、MinIO 和本地文件

#### 生产环境部署

使用 uvicorn 的多进程模式提高并发性能：

```bash
# 启动查询服务（4个worker进程）
uvicorn app.query_process.api.query_service:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 4

# 启动导入服务
uvicorn app.import_process.api.file_import_service:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2
```

### 3. 使用 API

#### 用户注册

```bash
curl -X POST "http://127.0.0.1:8001/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "password123"}'
```

#### 用户登录

```bash
curl -X POST "http://127.0.0.1:8001/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "password123"}'
```

#### 查询论文（非流式）

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "什么是 Transformer 的注意力机制？",
    "is_stream": false
  }'
```

#### 流式查询

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "解释一下 ResNet 的残差连接原理",
    "session_id": "your-session-id",
    "is_stream": true
  }'

# 然后连接 SSE 流获取结果
curl "http://127.0.0.1:8001/stream/your-session-id"
```

#### 查看会话历史

```bash
curl "http://127.0.0.1:8001/history/your-session-id" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### 管理会话

```bash
# 创建新会话
curl -X POST "http://127.0.0.1:8001/sessions" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_name": "我的研究会话"}'

# 列出所有会话
curl "http://127.0.0.1:8001/sessions" \
  -H "Authorization: Bearer YOUR_TOKEN"

# 重命名会话
curl -X PATCH "http://127.0.0.1:8001/sessions/session-id" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_name": "新名称"}'
```

### 4. Web 界面

访问 `http://127.0.0.1:8001/chat.html` 使用 Web 聊天界面进行对话式查询。

### 5. RAG 系统评估

使用 RAGAS 框架评估 RAG 系统性能：

```bash
# 生成评估数据集
python generate_ragas_eval_data.py

# 运行评估
python run_ragas_eval.py

# 完整评估流程
python eval_rag.py
```

评估结果将保存在 `result/` 目录下。

## 导入流程说明

论文导入采用基于 LangGraph 的流程编排，主要步骤：

1. **PDF 解析** (`node_pdf_to_md`)：使用 MinerU 将 PDF 转换为 Markdown
2. **图片处理** (`node_md_img`)：提取和上传图片到 MinIO
3. **实体识别** (`node_item_name_recognition`)：识别论文中的关键实体
4. **文档分割** (`node_document_split`)：智能切分文档为语义块
5. **向量化** (`node_bge_embedding`)：使用 BGE-M3 生成向量表示
6. **存储** (`node_import_milvus`)：将向量和元数据存入 Milvus

## 查询流程说明

查询处理也采用 LangGraph 编排，主要节点：

1. **查询理解**：分析用户意图，改写查询
2. **向量检索**：从 Milvus 检索相关文档块
3. **重排序**：使用 Reranker 对检索结果重排序
4. **上下文构建**：组织检索到的上下文
5. **答案生成**：使用 LLM 生成最终答案
6. **流式输出**：实时推送结果到前端

## 性能优化建议

### GPU 加速

- 确保 `BGE_DEVICE=cuda:0` 和 `BGE_FP16=1`
- 确保 `BGE_RERANKER_DEVICE=cuda:0` 和 `BGE_RERANKER_FP16=1`

### 批量导入优化

- 使用 `--import-batch-size` 控制并发数，避免内存溢出
- 使用 `--sleep-seconds` 设置批次间隔，避免 API 限流
- 大规模导入时可使用 `--download-only` 先下载，再分批导入

### 检索性能优化

- 调整 Milvus 索引参数（HNSW、IVF_FLAT 等）
- 根据实际数据量调整 `top_k` 和重排序数量
- 使用连接池管理数据库连接

## 常见问题

### 1. PDF 解析失败

- 检查 MinerU API token 是否有效
- 确认 PDF URL 可访问
- 查看日志获取详细错误信息

### 2. 向量化速度慢

- 确认是否启用 GPU（`BGE_DEVICE=cuda:0`）
- 检查 CUDA 环境是否正确安装
- 考虑使用批量处理

### 3. Milvus 连接失败

- 确认 Milvus 服务已启动
- 检查 `MILVUS_URL` 配置是否正确
- 查看 Milvus 日志

### 4. MongoDB 存储问题

- 确认 MongoDB 服务已启动
- 检查连接字符串和数据库名称
- 确保有足够的磁盘空间

## 开发说明

### 日志配置

日志配置在 `.env` 中：

```env
LOG_CONSOLE_ENABLE=True       # 控制台日志
LOG_CONSOLE_LEVEL=INFO        # 控制台日志级别
LOG_FILE_ENABLE=True          # 文件日志
LOG_FILE_LEVEL=INFO           # 文件日志级别
LOG_FILE_RETENTION=7 days     # 日志保留天数
```

日志文件位于 `logs/` 目录。

### 添加新的数据源

在 `paper_import/crawlers/` 目录下添加新的爬虫模块，使用装饰器注册：

```python
from paper_import.search import register_source

@register_source("my_source")
def search_my_source(query: str, max_results: int) -> list:
    # 实现搜索逻辑
    return papers
```

### 自定义 Prompt

Prompt 模板存储在 `prompts/` 目录，可根据需要修改。

## 许可证


## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式


---

**注意**：本系统仅供学术研究使用，请遵守相关论文平台的使用条款和版权规定。
