from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from threading import Lock
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None
_bge_m3_lock = Lock()

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    with _bge_m3_lock:
        if _bge_m3_ef is not None:
            logger.debug("BGE-M3模型由另一个任务初始化，复用实例")
            return _bge_m3_ef

        model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
        device = embedding_config.bge_device or "cpu"
        use_fp16 = embedding_config.bge_fp16 or False

        logger.info(
            "开始初始化BGE-M3模型",
            extra={
                "model_name": model_name,
                "device": device,
                "use_fp16": use_fp16,
                "normalize_embeddings": True,
            },
        )

        try:
            _bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                normalize_embeddings=True,
            )
            logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
            return _bge_m3_ef
        except Exception as e:
            _bge_m3_ef = None
            logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
            raise


def generate_embeddings(texts):
    """Generate BGE-M3 dense and sparse embeddings for a non-empty text list."""
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Invalid embedding input: texts must be a non-empty list")
        raise ValueError("texts must be a non-empty list")

    logger.info(f"Generating hybrid embeddings for {len(texts)} texts")
    try:
        model = get_bge_m3_ef()
        embeddings = model.encode_documents(texts)

        processed_sparse = []
        sparse = embeddings["sparse"]
        for i in range(len(texts)):
            start = sparse.indptr[i]
            end = sparse.indptr[i + 1]
            sparse_indices = sparse.indices[start:end].tolist()
            sparse_data = sparse.data[start:end].tolist()
            processed_sparse.append({k: v for k, v in zip(sparse_indices, sparse_data)})

        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],
            "sparse": processed_sparse,
        }
        logger.success(f"Generated embeddings for {len(texts)} texts")
        return result

    except Exception as e:
        logger.error(f"Text embedding generation failed: {str(e)}", exc_info=True)
        raise


