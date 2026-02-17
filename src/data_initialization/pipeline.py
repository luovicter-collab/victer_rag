"""
数据初始化管线

从 MinerU 生成的 JSON 文件中提取文档元素，转换为 RAG 嵌入数据格式。
各步骤会按顺序更新 JSON 的 metadata.parse_stage。

流程：
1. PDF -> MD 转换（复用 MinerU，保留 work_dir 中的 layout.json）
2. 元素提取（layout_json_parsed）
3. JSON 片段合并（fragment_merged）
4. 区域划分 head/body/tail（region_divided）
5. 可选：图片描述生成（image_description）
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中（这样才能导入 src 模块）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import logging
import time

from src.config.settings import (
    OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET,
    OSS_ENDPOINT,
    OSS_BUCKET,
    MINERU_API_URL,
    MINERU_API_KEY,
    PROJECT_ROOT,
)
from src.data_initialization.converters.pdf_to_md import PdfToMdConverter
from src.data_initialization.processors.layout_json_parser import ElementExtractor
from src.data_initialization.processors.json_fragment_merger import JsonFragmentMerger
from src.data_initialization.processors.region_extractor import JsonTitleExtractor
from src.data_initialization.processors.imagedescription_from_json import (
    JsonImageDescriptionProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def async_run_data_initialization_pipeline():
    """执行完整的数据初始化管线"""
    start_time = time.perf_counter()
    timings = {}

    # 目录配置
    pdf_dir = PROJECT_ROOT / "files" / "file_store" / "pdf_store"
    md_dir = PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_md"
    work_dir = PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_work"
    zip_dir = PROJECT_ROOT / "files" / "file_store" / "zip_store" / "minerU_zip"
    json_store_dir = PROJECT_ROOT / "files" / "file_store" / "json_store"

    # 1. PDF -> MD 转换
    t0 = time.perf_counter()
    converter = PdfToMdConverter(
        access_key_id=OSS_ACCESS_KEY_ID,
        access_key_secret=OSS_ACCESS_KEY_SECRET,
        endpoint=OSS_ENDPOINT,
        bucket_name=OSS_BUCKET,
        mineru_api_url=MINERU_API_URL,
        mineru_api_key=MINERU_API_KEY,
    )
    await converter.async_batch_convert_pdfs_with_layout(
        pdf_dir=pdf_dir,
        md_output_dir=md_dir,
        work_dir=work_dir,
        zip_output_dir=zip_dir,
    )
    timings["PDF -> Markdown"] = time.perf_counter() - t0

    # 2. 元素提取
    t0 = time.perf_counter()
    extractor = ElementExtractor()
    await extractor.batch_extract(work_dir=work_dir, output_dir=json_store_dir)
    timings["元素提取"] = time.perf_counter() - t0

    # 3. JSON 片段合并（句子/断词修复）
    t0 = time.perf_counter()
    merger = JsonFragmentMerger(json_store_dir=json_store_dir)
    await merger.async_batch_process(input_dir=json_store_dir, output_dir=json_store_dir)
    timings["JSON 片段合并"] = time.perf_counter() - t0

    # 4. 区域划分（head/body/tail，写入 metadata.region_division）
    t0 = time.perf_counter()
    region_extractor = JsonTitleExtractor(json_store_dir=json_store_dir)
    await asyncio.to_thread(region_extractor.extract_from_dir, json_store_dir)
    timings["区域划分"] = time.perf_counter() - t0

    # # 5. 图片描述生成（可选）
    # t0 = time.perf_counter()
    # processor = JsonImageDescriptionProcessor()
    # await processor.batch_process(
    #     input_dir=json_store_dir, output_dir=json_store_dir
    # )
    # timings["图片描述生成"] = time.perf_counter() - t0

    # 输出耗时统计
    total_time = time.perf_counter() - start_time
    timings["总耗时"] = total_time

    logger.info("=" * 50)
    logger.info("  数据初始化管线执行完成")
    logger.info("=" * 50)
    for step, cost in timings.items():
        logger.info(f"  {step:<20s}: {cost:>8.2f} 秒")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(async_run_data_initialization_pipeline())
