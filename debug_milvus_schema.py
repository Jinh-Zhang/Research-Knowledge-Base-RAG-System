from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from pymilvus import DataType

client = get_milvus_client()
collection_name = milvus_config.chunks_collection

# 删除旧测试集合
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
    print(f"已删除旧测试集合: {collection_name}")

# 创建新集合测试动态字段
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)

client.create_collection(collection_name=collection_name, schema=schema)
print(f"✅ 已创建测试集合: {collection_name}")

# 检查配置
desc = client.describe_collection(collection_name)
print(f"\nenable_dynamic_field: {desc.get('enable_dynamic_field')}")

if desc.get('enable_dynamic_field'):
    print("✅ 动态字段已启用")
else:
    print("❌ 动态字段未启用")

# 清理测试集合
client.drop_collection(collection_name)
print(f"\n已清理测试集合: {collection_name}")
