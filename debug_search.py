"""
检索诊断脚本：逐步排查 retrieved 为空的原因
运行：python debug_search.py
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.lm.embedding_utils import generate_embeddings

COLLECTION = os.environ.get("CHUNKS_COLLECTION", "paper_chunks")
QUERY = "TACT 的全称是什么？"
FILTER_TITLE = "开-Test-Time Adaptation by Causal Trimming"

client = get_milvus_client()

# ── 步骤1：集合是否存在，有多少条数据 ──
print("=" * 60)
print(f"集合名: {COLLECTION}")
try:
    count = client.query(collection_name=COLLECTION, filter="", output_fields=["count(*)"])
    print(f"总条数: {count}")
except Exception as e:
    print(f"query count 失败: {e}")

# ── 步骤2：查看实际存储的 paper_title 有哪些 ──
print("\n── 实际入库的 paper_title（前20条）──")
try:
    rows = client.query(
        collection_name=COLLECTION,
        filter="",
        output_fields=["paper_title"],
        limit=20,
    )
    titles = sorted(set(r.get("paper_title", "") for r in rows))
    for t in titles:
        print(f"  {repr(t)}")
except Exception as e:
    print(f"查询失败: {e}")

# ── 步骤3：不加过滤的混合检索 ──
print(f"\n── 不加过滤的检索，query={QUERY!r} ──")
try:
    emb = generate_embeddings([QUERY])
    reqs = create_hybrid_search_requests(
        dense_vector=emb["dense"][0],
        sparse_vector=emb["sparse"][0],
        expr=None,
        limit=5,
    )
    res = hybrid_search(client, COLLECTION, reqs, ranker_weights=(0.8, 0.2), norm_score=True, limit=5,
                        output_fields=["chunk_id", "content", "paper_title", "title"])
    hits = res[0] if res else []
    print(f"命中 {len(hits)} 条")
    for h in hits:
        entity = h.get("entity", h)
        print(f"  paper_title={entity.get('paper_title')!r}  content[:50]={entity.get('content','')[:50]!r}")
except Exception as e:
    print(f"检索失败: {e}")

# ── 步骤4：加 paper_title 过滤的检索 ──
print(f"\n── 加过滤 paper_title={FILTER_TITLE!r} ──")
try:
    escaped = FILTER_TITLE.replace('"', '\\"')
    expr = f'paper_title == "{escaped}"'
    print(f"  expr: {expr}")
    reqs = create_hybrid_search_requests(
        dense_vector=emb["dense"][0],
        sparse_vector=emb["sparse"][0],
        expr=expr,
        limit=5,
    )
    res = hybrid_search(client, COLLECTION, reqs, ranker_weights=(0.8, 0.2), norm_score=True, limit=5,
                        output_fields=["chunk_id", "content", "paper_title", "title"])
    hits = res[0] if res else []
    print(f"命中 {len(hits)} 条")
    for h in hits:
        entity = h.get("entity", h)
        print(f"  paper_title={entity.get('paper_title')!r}  content[:50]={entity.get('content','')[:50]!r}")
except Exception as e:
    print(f"检索失败: {e}")
