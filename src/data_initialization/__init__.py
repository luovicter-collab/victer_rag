"""
数据初始化模块

基于MinerU生成的layout.json直接构建JSON数据，无需经过MD处理流程。
"""

# 注意：不要在这里导入 pipeline，避免循环依赖
# 如需使用 pipeline，请直接导入：
# from src.data_initialization.pipeline import async_run_data_initialization_pipeline

from src.data_initialization.processors.layout_json_parser import (
    ElementExtractor,
    DocumentMetadata,
    DocumentElement,
    ElementSource,
    ElementMetadata,
    ExtractionResult,
)

__all__ = [
    "ElementExtractor",
    "DocumentMetadata",
    "DocumentElement",
    "ElementSource",
    "ElementMetadata",
    "ExtractionResult",
]
