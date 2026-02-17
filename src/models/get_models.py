import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import sqlite3
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from src.config.settings import (
    LLM_MODEL_FAST_CONFIG,
    LLM_MODEL_MAIN_CONFIG,
    LLM_MODEL_VISION_CONFIG,
    LLM_MODEL_HIGH_PRECISION_CONFIG,
    EMBEDDING_MODELN_CONFIG,
    VECTOR_DB_CONFIG,
    RELATION_DB_PATH,
)




# ===================================
# 获取 LLM 模型
# ===================================
def get_fast_llm_model():
    config = {
        k: v for k, v in LLM_MODEL_FAST_CONFIG.items() if v is not None and v != ""
    }
    return ChatOpenAI(**config)


def get_main_llm_model():
    config = {
        k: v for k, v in LLM_MODEL_MAIN_CONFIG.items() if v is not None and v != ""
    }
    return ChatOpenAI(**config)


def get_vision_llm_model():
    config = {
        k: v for k, v in LLM_MODEL_VISION_CONFIG.items() if v is not None and v != ""
    }
    return ChatOpenAI(**config)


def get_high_precision_llm_model():
    config = {
        k: v
        for k, v in LLM_MODEL_HIGH_PRECISION_CONFIG.items()
        if v is not None and v != ""
    }
    return ChatOpenAI(**config)


# ===================================
# 获取 嵌入模型
# ===================================
def get_embedding_model():
    config = {
        k: v for k, v in EMBEDDING_MODELN_CONFIG.items() if v is not None and v != ""
    }
    return HuggingFaceEmbeddings(**config)


# ===================================
# 获取 向量数据库
# ===================================
def get_vector_db():
    config = {k: v for k, v in VECTOR_DB_CONFIG.items() if v is not None and v != ""}
    client_settings = Settings(
        persist_directory=config["persist_directory"],
        anonymized_telemetry=False,
    )
    return Chroma(**config, client_settings=client_settings)


# ===================================
# 获取 关系型数据库
# ===================================
def get_relation_db():
    return sqlite3.connect(RELATION_DB_PATH)


if __name__ == "__main__":
    print(get_embedding_model())
    print(get_vector_db())
    print(get_fast_llm_model().invoke("Hello, how are you?"))
    print(get_relation_db())
