import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import torch


# ===== 自动添加 src 目录到 Python 路径 =====
# 这样无论从哪个目录运行脚本，都能正确导入 src 包
def _ensure_src_in_path():
    """确保 src 目录在 Python 路径中"""
    src_path = Path(__file__).resolve().parent
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)


_ensure_src_in_path()

# 统一加载环境变量，确保配置来源一致
load_dotenv()


def _get_env_int(key: str, default: Optional[int]) -> Optional[int]:
    """获取整型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要整数值，但当前为 {raw}") from exc


def _get_env_float(key: str, default: Optional[float]) -> Optional[float]:
    """获取浮点型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要浮点值，但当前为 {raw}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    """获取布尔型环境变量，支持 true/false/1/0 表达式"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_choice(key: str, choices: set[str], default: str) -> str:
    """获取有限集合中的字符串配置，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(
            f"环境变量 {key} 必须为 {', '.join(sorted(choices))} 之一，但当前为 {raw}"
        )
    return value


# ===== 基础路径设置 =====
CONFIG_DIR = Path(__file__).resolve().parent  # config目录
SRC_DIR = CONFIG_DIR.parent  # 项目根目录 src目录
PROJECT_ROOT = SRC_DIR.parent  # 项目根目录

# ===== 设备设置 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===== MinerU 配置 =====
MINERU_API_KEY = os.getenv("MinerU_API_KEY", "")
MINERU_API_URL = os.getenv("MinerU_API_URL", "")


# ===== OSS 配置 =====
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "")
OSS_BUCKET = os.getenv("OSS_BUCKET", "")


# ===== LLM 模型配置 =====
LLM_MODEL_API_KEY = os.getenv("OPENAI_API_KEY", "")

LLM_MAIN_MODEL = os.getenv("LLM_MODEL_CHAT_MAIN") or None
LLM_FAST_MODEL = os.getenv("LLM_MODEL_CHAT_FAST") or None
LLM_VISION_MODEL = os.getenv("LLM_MODEL_VISION") or None
LLM_HIGH_PRECISION_MODEL = os.getenv("LLM_MODEL_HIGH_PRECISION") or None

LLM_MODEL_TEMPERATURE = _get_env_float("LLM_MODEL_TEMPERATURE", None)
LLM_MODEL_MAX_TOKENS = _get_env_int("LLM_MODEL_MAX_TOKENS", None)
LLM_MODEL_TIMEOUT = _get_env_int("LLM_MODEL_TIMEOUT", None)
LLM_MODEL_MAX_RETRIES = _get_env_int("LLM_MODEL_MAX_RETRIES", None)


LLM_MODEL_MAIN_CONFIG = {
    "model": LLM_MAIN_MODEL,
    "api_key": LLM_MODEL_API_KEY,
    "temperature": LLM_MODEL_TEMPERATURE,
    "max_tokens": LLM_MODEL_MAX_TOKENS,
    "timeout": LLM_MODEL_TIMEOUT,
    "max_retries": LLM_MODEL_MAX_RETRIES,
}

LLM_MODEL_FAST_CONFIG = {
    "model": LLM_FAST_MODEL,
    "api_key": LLM_MODEL_API_KEY,
    "temperature": LLM_MODEL_TEMPERATURE,
    "max_tokens": LLM_MODEL_MAX_TOKENS,
    "timeout": LLM_MODEL_TIMEOUT,
    "max_retries": LLM_MODEL_MAX_RETRIES,
}

LLM_MODEL_VISION_CONFIG = {
    "model": LLM_VISION_MODEL,
    "api_key": LLM_MODEL_API_KEY,
    "temperature": LLM_MODEL_TEMPERATURE,
    "max_tokens": LLM_MODEL_MAX_TOKENS,
    "timeout": LLM_MODEL_TIMEOUT,
    "max_retries": LLM_MODEL_MAX_RETRIES,
}

LLM_MODEL_HIGH_PRECISION_CONFIG = {
    "model": LLM_HIGH_PRECISION_MODEL,
    "api_key": LLM_MODEL_API_KEY,
    "temperature": LLM_MODEL_TEMPERATURE,
    "max_tokens": LLM_MODEL_MAX_TOKENS,
    "timeout": LLM_MODEL_TIMEOUT,
    "max_retries": LLM_MODEL_MAX_RETRIES,
}


# ===== 嵌入模型配置 =====
EMBEDDING_PATH = str(
    PROJECT_ROOT
    / "model"
    / "embeddings"
    / "sentence-transformers"
    / "all-mpnet-base-v2"
)
EMBEDDING_MODELN_CONFIG = {
    "model_name": EMBEDDING_PATH,
    "model_kwargs": {"device": DEVICE},
}


# ===== 向量数据库配置 =====
VECTOR_DB_PATH = str(PROJECT_ROOT / "files" / "vector_store" / "rag")
VECTOR_DB_COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME") or None
VECTOR_DB_CONFIG = {
    "collection_name": VECTOR_DB_COLLECTION_NAME,
    "persist_directory": VECTOR_DB_PATH,
}


# ===== 关系型数据库配置 =====
RELATION_DB_PATH = str(PROJECT_ROOT / "files" / "relation_store" / "rag.db")


# ===== 文本切割配置 =====
TEXT_CHUNK_SON_SIZE = _get_env_int("TEXT_CHUNK_SON_SIZE", None)
TEXT_CHUNK_SON_OVERLAP = _get_env_int("TEXT_CHUNK_SON_OVERLAP", None)
IMG_CHUNK_SON_SIZE = _get_env_int("IMG_CHUNK_SON_SIZE", None)
IMG_CHUNK_SON_OVERLAP = _get_env_int("IMG_CHUNK_SON_OVERLAP", None)


# ===== 图片识别配置 =====
PICTURE_MAX_CONCURRENT_TASKS = _get_env_int("PICTURE_MAX_CONCURRENT_TASKS", None)

# ===== PDF 转换配置 =====
PDF_TO_MD_MAX_CONCURRENT_TASKS = _get_env_int("PDF_TO_MD_MAX_CONCURRENT_TASKS", None)
PDF_TO_MD_TASK_INTERVAL = _get_env_int("PDF_TO_MD_TASK_INTERVAL", None)
PDF_TO_MD_TASK_TIMEOUT = _get_env_int("PDF_TO_MD_TASK_TIMEOUT", None)

# ===== JSON 图片描述配置 =====
JSON_IMAGE_DESCRIPTION_MAX_CONCURRENT = _get_env_int(
    "JSON_IMAGE_DESCRIPTION_MAX_CONCURRENT", 5
)
JSON_IMAGE_DESCRIPTION_TIMEOUT = _get_env_int("JSON_IMAGE_DESCRIPTION_TIMEOUT", 120)
JSON_IMAGE_DESCRIPTION_SUMMARY_LENGTH = _get_env_int(
    "JSON_IMAGE_DESCRIPTION_SUMMARY_LENGTH", 500000
)


# ===== 元数据抽取配置 =====
METADATA_EXTRACTION_MAX_CONCURRENT = _get_env_int("METADATA_EXTRACTION_MAX_CONCURRENT", 5)
METADATA_EXTRACTION_TIMEOUT = _get_env_int("METADATA_EXTRACTION_TIMEOUT", 120)
METADATA_EXTRACTION_MAX_RETRIES = _get_env_int("METADATA_EXTRACTION_MAX_RETRIES", 3)


# =============================================================================
# 处理阶段常量（与管线步骤对应，写入 JSON metadata.parse_stage）
# =============================================================================
STAGE_LAYOUT_JSON_PARSED = os.getenv("STAGE_LAYOUT_JSON_PARSED") or "layout_json_parsed"
STAGE_FRAGMENT_MERGED = os.getenv("STAGE_FRAGMENT_MERGED") or "fragment_merged"
STAGE_REGION_DIVIDED = os.getenv("STAGE_REGION_DIVIDED") or "region_divided"
STAGE_METADATA_EXTRACTED = os.getenv("STAGE_METADATA_EXTRACTED") or "metadata_extracted"
STAGE_IMAGE_DESCRIPTION = os.getenv("STAGE_IMAGE_DESCRIPTION") or "image_description"
STAGE_RAG_EMBEDDING = os.getenv("STAGE_RAG_EMBEDDING") or "rag_embedding"

# 处理阶段列表（按管线顺序）
PROCESS_STAGES = [
    STAGE_LAYOUT_JSON_PARSED,
    STAGE_FRAGMENT_MERGED,
    STAGE_REGION_DIVIDED,
    STAGE_METADATA_EXTRACTED,
    STAGE_IMAGE_DESCRIPTION,
    STAGE_RAG_EMBEDDING,
]
