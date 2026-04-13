from agent.state import ImportGraphState
import sys

def node_import_kg(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入知识图谱 (node_import_kg)
    为什么叫这个名字: 构建 Knowledge Graph (KG) 并存入 Neo4j。
    未来要实现:
    1. 调用 LLM 从文本中抽取实体 (Entity) 和关系 (Relation)。
    2. 连接 Neo4j 数据库。
    3. 执行 Cypher 语句将图谱数据写入数据库。
    """
    print(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    return state