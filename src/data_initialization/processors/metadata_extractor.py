"""
文献元数据提取器

从已划分区域的 JSON 文件中提取文献元数据（head 和 tail 区域）。

功能：
1. 从 JSON 文件读取 head 和 tail 区域的文本内容
2. 调用 LLM 提取结构化元数据：
   - Head 区域：标题、作者、机构、摘要、关键词、分类、期刊信息
   - Tail 区域：参考文献列表
3. 将提取的元数据写入 metadata

执行顺序：在 region_extractor 之后、imagedescription_from_json 之前。
"""

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config.prompts.preprocessing_prompts import (
    DOCUMENT_METADATA_EXTRACTION_PROMPT_EN,
    DOCUMENT_METADATA_EXTRACTION_PROMPT_ZH,
    REFERENCES_EXTRACTION_PROMPT_EN,
    REFERENCES_EXTRACTION_PROMPT_ZH,
)
from src.config.settings import (
    PROJECT_ROOT,
    STAGE_METADATA_EXTRACTED,
)
from src.models.get_models import get_fast_llm_model
from src.data_initialization.processors import (
    should_skip_stage,
    update_parse_stage,
)

# ---------------------- 类型定义 ----------------------

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


# ---------------------- 自定义异常 ----------------------


class MetadataExtractorError(Exception):
    """元数据提取过程中的统一异常基类。"""

    pass


# ---------------------- 数据结构 ----------------------


@dataclass
class MetadataExtractionResult:
    """元数据提取结果"""

    doc_id: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    references: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None


@dataclass
class BatchProcessResult:
    """批量处理结果"""

    total_files: int
    success_count: int
    skip_count: int
    fail_count: int


# ---------------------- 辅助函数 ----------------------


def _get_text_from_element(element: Dict[str, Any]) -> str:
    """从元素 content 中提取文本。"""
    content = element.get("content", {})
    if not isinstance(content, dict):
        return ""
    
    text = content.get("text", "")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list) and text:
        return " ".join([str(t).strip() for t in text if t]).strip()
    return ""


def _extract_texts_by_region(
    elements: List[Dict[str, Any]], region_division: Dict[str, Dict[str, int]]
) -> Dict[str, str]:
    """
    根据 region_division 提取各区域的文本。

    Args:
        elements: 元素列表
        region_division: 区域划分信息 {head: {start_seq, end_seq}, body: {...}, tail: {...}}

    Returns:
        {"head": "文本内容", "tail": "文本内容"}
    """
    result = {"head": "", "tail": ""}
    
    if not elements or not region_division:
        return result
    
    # 构建元素索引（1-based）
    element_map = {i + 1: el for i, el in enumerate(elements)}
    
    # 提取 head 区域文本
    head_region = region_division.get("head", {})
    head_start = head_region.get("start_seq", 1)
    head_end = head_region.get("end_seq", 0)
    
    head_texts = []
    for seq in range(head_start, head_end + 1):
        el = element_map.get(seq)
        if el:
            text = _get_text_from_element(el)
            if text:
                head_texts.append(text)
    result["head"] = "\n\n".join(head_texts)
    
    # 提取 tail 区域文本
    tail_region = region_division.get("tail", {})
    tail_start = tail_region.get("start_seq", 1)
    tail_end = tail_region.get("end_seq", 0)
    
    tail_texts = []
    for seq in range(tail_start, tail_end + 1):
        el = element_map.get(seq)
        if el:
            text = _get_text_from_element(el)
            if text:
                tail_texts.append(text)
    result["tail"] = "\n\n".join(tail_texts)
    
    return result


def _parse_json_response(response: str) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON 响应。"""
    # 尝试提取 JSON 代码块
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接解析
        json_str = response
    
    # 清理 JSON 字符串
    json_str = json_str.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试修复常见的 JSON 错误
        # 移除单引号
        json_str = json_str.replace("'", '"')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("JSON 解析失败: %s", response[:200])
            return {}


# ---------------------- 主工具类 ----------------------


class MetadataExtractor:
    """
    文献元数据提取器。

    从 JSON 文件的 head 和 tail 区域提取结构化元数据。

    功能：
    1. 读取 JSON 文件
    2. 提取 head/tail 区域的文本
    3. 调用 LLM 提取元数据（标题、作者、摘要、关键词等）
    4. 调用 LLM 提取参考文献
    5. 将结果写入 metadata

    用法：

        extractor = MetadataExtractor()
        result = await extractor.extract_from_json(
            json_path=Path("json_store/doc.json"),
        )
        print(f"提取结果: {result.success}")
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_concurrent_tasks: int = 3,
        json_store_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
    ) -> None:
        """
        初始化 MetadataExtractor。

        Args:
            llm: ChatOpenAI 实例，用于元数据提取。若为 None，则使用默认配置创建。
            max_concurrent_tasks: 最大并发任务数。
            json_store_dir: JSON 文件存储目录。
            output_dir: 输出目录。
        """
        self.llm = llm or get_fast_llm_model()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # 设置目录路径
        if json_store_dir is None:
            self.json_store_dir = PROJECT_ROOT / "files" / "file_store" / "json_store"
        else:
            self.json_store_dir = Path(json_store_dir).resolve()

        if output_dir is None:
            self.output_dir = self.json_store_dir
        else:
            self.output_dir = Path(output_dir).resolve()

    # ---------------------- 文本提取 ----------------------

    def _extract_region_texts(self, json_data: Dict[str, Any]) -> Dict[str, str]:
        """提取 head 和 tail 区域的文本。"""
        elements = json_data.get("elements", [])
        region_division = json_data.get("metadata", {}).get("region_division", {})
        
        return _extract_texts_by_region(elements, region_division)

    # ---------------------- LLM 调用 ----------------------

    async def _extract_head_metadata(
        self,
        doc_id: str,
        language: str,
        head_text: str,
    ) -> Optional[Dict[str, Any]]:
        """
        从 head 区域提取元数据。

        Args:
            doc_id: 文档 ID
            language: 语言 (zh/en)
            head_text: head 区域文本

        Returns:
            提取的元数据字典
        """
        if not head_text or len(head_text.strip()) < 10:
            logger.warning("Head 区域文本过短，跳过元数据提取")
            return None

        async with self.semaphore:
            try:
                # 选择提示词
                prompt = (
                    DOCUMENT_METADATA_EXTRACTION_PROMPT_ZH
                    if language == "zh"
                    else DOCUMENT_METADATA_EXTRACTION_PROMPT_EN
                )

                # 填充提示词模板
                prompt_text = prompt.format(
                    doc_id=doc_id,
                    language=language,
                    head_text=head_text[:8000],  # 限制文本长度
                    tail_text="",  # head 区域提取时不使用 tail
                )

                # 调用 LLM
                response = await self.llm.ainvoke(
                    [HumanMessage(content=prompt_text)]
                )

                # 解析 JSON 响应
                metadata = _parse_json_response(response.content)
                
                if metadata:
                    logger.info("Head 元数据提取成功: %s", doc_id)
                    return metadata
                else:
                    logger.warning("Head 元数据解析失败: %s", doc_id)
                    return None

            except Exception as e:
                logger.exception("Head 元数据提取失败 (%s): %s", doc_id, str(e))
                return None

    async def _extract_tail_references(
        self,
        doc_id: str,
        language: str,
        tail_text: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        从 tail 区域提取参考文献。

        Args:
            doc_id: 文档 ID
            language: 语言 (zh/en)
            tail_text: tail 区域文本

        Returns:
            参考文献列表
        """
        if not tail_text or len(tail_text.strip()) < 50:
            logger.warning("Tail 区域文本过短，跳过参考文献提取")
            return None

        async with self.semaphore:
            try:
                # 选择提示词
                prompt = (
                    REFERENCES_EXTRACTION_PROMPT_ZH
                    if language == "zh"
                    else REFERENCES_EXTRACTION_PROMPT_EN
                )

                # 填充提示词模板
                prompt_text = prompt.format(
                    doc_id=doc_id,
                    language=language,
                    references_text=tail_text[:15000],  # 限制文本长度
                )

                # 调用 LLM
                response = await self.llm.ainvoke(
                    [HumanMessage(content=prompt_text)]
                )

                # 解析 JSON 响应
                # 响应应该是一个数组
                try:
                    json_match = re.search(r"\[[\s\S]*\]", response.content)
                    if json_match:
                        references = json.loads(json_match.group())
                        logger.info("参考文献提取成功: %s, 共 %d 条", doc_id, len(references))
                        return references
                except (json.JSONDecodeError, AttributeError):
                    logger.warning("参考文献解析失败: %s", doc_id)
                    return None

            except Exception as e:
                logger.exception("参考文献提取失败 (%s): %s", doc_id, str(e))
                return None

    # ---------------------- 主处理方法 ----------------------

    async def extract_from_json(
        self,
        json_path: PathLike,
    ) -> MetadataExtractionResult:
        """
        从 JSON 文件提取元数据。

        Args:
            json_path: JSON 文件路径

        Returns:
            MetadataExtractionResult: 提取结果
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            return MetadataExtractionResult(
                doc_id=json_path.stem,
                success=False,
                error_message="文件不存在",
            )

        try:
            # 读取 JSON
            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            return MetadataExtractionResult(
                doc_id=json_path.stem,
                success=False,
                error_message=f"JSON 读取失败: {str(e)}",
            )

        # 获取基本信息
        doc_id = json_data.get("metadata", {}).get("doc_id", json_path.stem)
        language = json_data.get("metadata", {}).get("language", "en")
        
        # 检查是否已处理
        if should_skip_stage(str(json_path), STAGE_METADATA_EXTRACTED):
            return MetadataExtractionResult(
                doc_id=doc_id,
                success=True,
                metadata=json_data.get("metadata", {}).get("extracted_metadata"),
                references=json_data.get("metadata", {}).get("references"),
            )

        # 提取 head/tail 文本
        region_texts = self._extract_region_texts(json_data)
        
        # 提取 head 元数据
        head_metadata = await self._extract_head_metadata(
            doc_id=doc_id,
            language=language,
            head_text=region_texts.get("head", ""),
        )
        
        # 提取 tail 参考文献
        references = await self._extract_tail_references(
            doc_id=doc_id,
            language=language,
            tail_text=region_texts.get("tail", ""),
        )

        # 更新 JSON 文件
        try:
            if head_metadata or references:
                json_data.setdefault("metadata", {})
                
                if head_metadata:
                    json_data["metadata"]["extracted_metadata"] = head_metadata
                
                if references:
                    json_data["metadata"]["references"] = references
                
                json_data["metadata"]["parse_stage"] = STAGE_METADATA_EXTRACTED
                
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                logger.info("元数据已写入: %s", doc_id)
            
            return MetadataExtractionResult(
                doc_id=doc_id,
                success=True,
                metadata=head_metadata,
                references=references,
            )
            
        except Exception as e:
            return MetadataExtractionResult(
                doc_id=doc_id,
                success=False,
                error_message=f"JSON 写入失败: {str(e)}",
            )

    async def batch_process(
        self,
        input_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        skip_existing: bool = True,
    ) -> BatchProcessResult:
        """
        批量处理目录下所有 JSON 文件。

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            skip_existing: 是否跳过已处理的文件

        Returns:
            BatchProcessResult: 处理结果统计
        """
        input_path = Path(input_dir) if input_dir else self.json_store_dir
        output_path = Path(output_dir) if output_dir else self.output_dir

        if not input_path.exists():
            raise MetadataExtractorError(f"输入目录不存在: {input_path}")

        results = BatchProcessResult(
            total_files=0,
            success_count=0,
            skip_count=0,
            fail_count=0,
        )

        # 遍历所有 JSON 文件
        tasks = []
        for json_file in sorted(input_path.glob("*.json")):
            # 检查是否跳过
            if skip_existing and should_skip_stage(
                str(json_file), STAGE_METADATA_EXTRACTED
            ):
                logger.info("跳过（已处理）: %s", json_file.name)
                results.skip_count += 1
                results.total_files += 1
                continue

            tasks.append(self.extract_from_json(json_file))

        # 并发处理
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in task_results:
                results.total_files += 1
                if isinstance(result, Exception):
                    results.fail_count += 1
                    logger.error("处理异常: %s", str(result))
                elif result.success:
                    results.success_count += 1
                else:
                    results.fail_count += 1

        logger.info(
            "批量处理完成: 总数=%d, 成功=%d, 跳过=%d, 失败=%d",
            results.total_files,
            results.success_count,
            results.skip_count,
            results.fail_count,
        )

        return results


# ---------------------- 独立运行 ----------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    extractor = MetadataExtractor()

    async def main():
        # 单文件处理
        # result = await extractor.extract_from_json(
        #     json_path=Path("files/file_store/json_store/test.json"),
        # )
        # print(f"处理结果: {result.success}")

        # 批量处理
        result = await extractor.batch_process(
            input_dir=PROJECT_ROOT / "files" / "file_store" / "json_store",
        )
        print(f"批量处理完成: {result}")

    asyncio.run(main())
