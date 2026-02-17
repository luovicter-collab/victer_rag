# src/preprocessing/enhancers/_05_imagedescription_from_json.py

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

"""
JSON 图片描述生成器

为 layout_json_parser.py 处理后的 JSON 文件中有 image_path 的元素块生成图片描述。

功能：
1. 从 JSON 文件读取包含 image_path 的元素
2. 优先从 metadata.abstract 提取摘要，没有则调用 LLM
3. 根据语言选择提示词（中/英文）
4. 调用 LLM Vision 模型生成图片描述
5. 更新 JSON 文件的 content.description 字段

与 _02_pictureprecesser.py 的区别：
- 输入格式：JSON 文件（而非 Markdown）
- 摘要来源：从 metadata.abstract 提取（而非生成全文摘要）
- 图片路径：直接从 source.image_path 获取（而非解析 Markdown 链接）
"""

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config.prompts.preprocessing_prompts import (
    DOCUMENT_SUMMARY_PROMPT_EN,
    DOCUMENT_SUMMARY_PROMPT_ZH,
    FIGURE_DESCRIPTION_PROMPT_EN,
    FIGURE_DESCRIPTION_PROMPT_ZH,
    EQUATION_DESCRIPTION_PROMPT_EN,
    EQUATION_DESCRIPTION_PROMPT_ZH,
    TABLE_DESCRIPTION_PROMPT_EN,
    TABLE_DESCRIPTION_PROMPT_ZH,
)
from src.config.settings import (
    PROJECT_ROOT,
    JSON_IMAGE_DESCRIPTION_MAX_CONCURRENT,
    JSON_IMAGE_DESCRIPTION_SUMMARY_LENGTH,
)
from src.models.get_models import get_vision_llm_model, get_fast_llm_model
from src.config.settings import (
    STAGE_IMAGE_DESCRIPTION,
)
from src.data_initialization.processors import (
    should_skip_stage,
    update_parse_stage,
)

# ---------------------- 辅助函数 ----------------------


def _detect_image_type(image_path: str, element_type: str = "") -> str:
    """检测图片类型。

    Args:
        image_path: 图片路径
        element_type: 元素类型（如果有）

    Returns:
        图片类型: "figure", "equation", "table"
    """
    # 如果元素类型已知，直接返回
    if element_type:
        et = element_type.lower()
        if "equation" in et or "formula" in et or "code" in et:
            return "equation"
        if "table" in et:
            return "table"

    # 根据路径特征判断
    path_lower = image_path.lower()
    if (
        "equation" in path_lower
        or "formula" in path_lower
        or "eq" in path_lower.replace("-", "")
    ):
        return "equation"
    if "table" in path_lower or "tab" in path_lower:
        return "table"

    # 默认返回普通图片
    return "figure"


def _get_text_value(text_field: Any) -> str:
    """安全获取文本值，处理字符串和列表情况。

    Args:
        text_field: 文本字段（可能是 str 或 list）

    Returns:
        文本字符串
    """
    if isinstance(text_field, str):
        return text_field.strip()
    elif isinstance(text_field, list):
        # 如果是列表，取第一个元素并转为字符串
        if text_field:
            return str(text_field[0]).strip()
        return ""
    elif text_field is None:
        return ""
    return str(text_field).strip()


# ---------------------- 主工具类 ----------------------

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

# ---------------------- 自定义异常 ----------------------


class ImageDescriptionError(Exception):
    """图片描述生成过程中的统一异常基类。"""

    pass


# ---------------------- 数据结构 ----------------------


@dataclass
class ImageDescriptionResult:
    """单张图片描述结果"""

    element_id: str
    description: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchProcessResult:
    """批量处理结果"""

    total_files: int
    success_count: int
    skip_count: int
    fail_count: int
    total_images: int
    success_images: int
    fail_images: int


@dataclass
class ElementInfo:
    """元素信息"""

    id: str
    image_path: str
    section_title: Optional[str]
    language: str
    type: str = ""  # 元素类型（figure/equation/table）


# ---------------------- 主工具类 ----------------------


class JsonImageDescriptionProcessor:
    """
    JSON 图片描述处理器。

    为 JSON 文件中有 image_path 的元素块生成图片描述。

    功能：
    1. 从 JSON 文件读取包含 image_path 的元素
    2. 优先从 metadata.abstract 提取摘要，没有则调用 LLM
    3. 根据语言选择提示词
    4. 调用 LLM 生成图片描述
    5. 更新 JSON 文件

    用法：

        processor = JsonImageDescriptionProcessor()
        result = await processor.process_single_json(
            json_path=Path("json_store/doc.json"),
            output_dir=Path("json_store"),
        )
        print(f"处理完成：{result.success}")

        # 批量处理
        await processor.batch_process(
            input_dir=Path("json_store"),
            output_dir=Path("json_store"),
        )
    """

    def __init__(
        self,
        llm_vision: Optional[ChatOpenAI] = None,
        llm_summary: Optional[ChatOpenAI] = None,
        max_concurrent_tasks: Optional[int] = None,
        json_store_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
    ) -> None:
        """
        初始化 JsonImageDescriptionProcessor。

        Args:
            llm_vision: ChatOpenAI 实例，用于图片描述生成。若为 None，则使用默认配置创建。
            llm_summary: ChatOpenAI 实例，用于摘要生成。若为 None，则使用默认配置创建。
            max_concurrent_tasks: 最大并发任务数。若为 None，则使用配置中的默认值。
            json_store_dir: JSON 文件存储目录。
                           若为 None，则默认使用 PROJECT_ROOT/files/file_store/json_store。
            output_dir: 输出目录（处理后的 JSON 保存目录）。
                       若为 None，则默认与 json_store_dir 相同。
        """
        self.max_concurrent_tasks = max_concurrent_tasks or (
            JSON_IMAGE_DESCRIPTION_MAX_CONCURRENT or 5
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.llm_vision = llm_vision or get_vision_llm_model()
        self.llm_summary = llm_summary or get_fast_llm_model()

        # 设置目录路径
        if json_store_dir is None:
            self.json_store_dir = PROJECT_ROOT / "files" / "file_store" / "json_store"
        else:
            self.json_store_dir = Path(json_store_dir).resolve()

        if output_dir is None:
            self.output_dir = self.json_store_dir
        else:
            self.output_dir = Path(output_dir).resolve()

        # 当前处理的提示词和语言（在处理文件时动态设置）
        self.prompt: str = FIGURE_DESCRIPTION_PROMPT_EN
        self.current_language: Optional[str] = None

    # ---------------------- JSON 解析相关 ----------------------

    def _find_elements_with_image_path(self, json_data: Dict) -> List[ElementInfo]:
        """查找 JSON 中包含 source.image_path 的元素。

        Args:
            json_data: JSON 数据

        Returns:
            元素信息列表
        """
        elements: List[ElementInfo] = []

        if "elements" not in json_data or not isinstance(json_data["elements"], list):
            return elements

        language = json_data.get("metadata", {}).get("language", "en")

        for element in json_data["elements"]:
            if not isinstance(element, dict):
                continue

            # 检查是否有 image_path
            source = element.get("source", {})
            image_path = source.get("image_path", "")

            if not image_path:
                continue

            element_info = ElementInfo(
                id=element.get("id", ""),
                image_path=image_path,
                section_title=source.get("section_title"),
                language=language,
                type=element.get("type", ""),
            )
            elements.append(element_info)

        return elements

    # ---------------------- 摘要提取相关 ----------------------

    def _extract_abstract_from_json(
        self, json_data: Dict, target_language: str
    ) -> Optional[str]:
        """从 JSON 文件的 metadata 中提取摘要。

        优先提取与目标语言匹配的摘要，如果没有则返回第一个可用的摘要。

        Args:
            json_data: JSON 数据
            target_language: 目标语言代码

        Returns:
            摘要文本，如果没有则返回 None
        """
        metadata = json_data.get("metadata", {})
        abstract = metadata.get("abstract", [])

        if not abstract:
            return None

        # 如果是字符串，直接返回
        if isinstance(abstract, str):
            return abstract if abstract.strip() else None

        # 如果是数组，优先提取与语言匹配的摘要
        if isinstance(abstract, list):
            for item in abstract:
                if isinstance(item, dict):
                    item_lang = item.get("language", "")
                    item_text = item.get("text", "")
                    if item_lang == target_language and item_text.strip():
                        return item_text
                    # 缓存第一个非空摘要作为备选
                    elif not locals().get("_first_abstract") and item_text.strip():
                        locals()["_first_abstract"] = item_text
                elif isinstance(item, str) and item.strip():
                    if not locals().get("_first_abstract"):
                        locals()["_first_abstract"] = item
                    if target_language == "en":
                        return item

            # 返回第一个非空摘要
            return locals().get("_first_abstract")

        return None

    # ---------------------- 语言与提示词相关 ----------------------

    def _get_prompt_by_language_and_type(
        self, language: str, image_type: str = "figure"
    ) -> str:
        """根据语言和图片类型获取图片描述提示词。

        Args:
            language: 语言代码 (zh/en)
            image_type: 图片类型 (figure/equation/table)

        Returns:
            提示词模板
        """
        is_zh = language == "zh"

        if image_type == "equation":
            return (
                EQUATION_DESCRIPTION_PROMPT_ZH
                if is_zh
                else EQUATION_DESCRIPTION_PROMPT_EN
            )
        elif image_type == "table":
            return TABLE_DESCRIPTION_PROMPT_ZH if is_zh else TABLE_DESCRIPTION_PROMPT_EN
        else:
            return (
                FIGURE_DESCRIPTION_PROMPT_ZH
                if is_zh
                else FIGURE_DESCRIPTION_PROMPT_EN
            )

    def _get_summary_prompt_by_language(self, language: str) -> str:
        """根据语言获取文档概要生成提示词。

        Args:
            language: 语言代码

        Returns:
            提示词模板
        """
        return (
            DOCUMENT_SUMMARY_PROMPT_ZH
            if language == "zh"
            else DOCUMENT_SUMMARY_PROMPT_EN
        )

    # ---------------------- LLM 调用相关 ----------------------

    async def _read_image(self, image_path: Path) -> Tuple[bytes, Optional[str]]:
        """同步读取本地图片文件。

        Args:
            image_path: 图片文件路径

        Returns:
            Tuple[bytes, Optional[str]]: 图片文件字节和 MIME 类型
        """
        if not image_path.exists() or not image_path.is_file():
            return b"", None

        try:
            with image_path.open("rb") as f:
                image_bytes = f.read()
            # 简单判断图片类型
            if image_bytes[:2] == b"\xff\xd8":
                mime_type = "image/jpeg"
            elif image_bytes[:4] == b"\x89PNG":
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"  # 默认
            return (image_bytes, mime_type) if image_bytes else (b"", None)
        except Exception:
            return b"", None

    async def _image_to_base64(self, image_path: str) -> Tuple[str, Optional[str]]:
        """将图片转换为 base64 编码字符串。

        Args:
            image_path: 图片路径

        Returns:
            Tuple[str, Optional[str]]: base64 编码字符串和 MIME 类型
        """
        try:
            path = Path(image_path)
            image_bytes, mime_type = await self._read_image(path)
            if not image_bytes:
                return "", None
            base64_str = base64.b64encode(image_bytes).decode()
            return base64_str, mime_type
        except Exception as e:
            logger.exception("图片转换为 base64 失败 (%s): %s", image_path, str(e))
            return "", None

    async def _call_llm_for_description(
        self,
        image_path: str,
        document_summary: str,
        prompt: str,
    ) -> str:
        """调用 LLM 生成图片描述。

        Args:
            image_path: 图片路径
            document_summary: 文档摘要
            prompt: 提示词

        Returns:
            图片描述文本
        """
        async with self.semaphore:
            try:
                base64_image, mime_type = await self._image_to_base64(image_path)
                if not base64_image:
                    return ""

                data_url = f"data:{mime_type or 'image/jpeg'};base64,{base64_image}"

                prompt_text = prompt.format(document_summary=document_summary)

                prompt_msg = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                ]

                response = await self.llm_vision.ainvoke(prompt_msg)
                return response.content.strip()

            except Exception as e:
                logger.exception(
                    "调用 LLM 生成图片描述失败 (%s): %s", image_path, str(e)
                )
                return ""

    async def _generate_document_summary(
        self,
        json_data: Dict,
        language: str,
        file_name: str,
    ) -> str:
        """使用 LLM 生成文档概要。

        Args:
            json_data: JSON 数据
            language: 语言代码
            file_name: 文件名

        Returns:
            文档概要
        """
        # 提取所有文本内容
        text_lines = []
        for element in json_data.get("elements", []):
            if not isinstance(element, dict):
                continue

            element_type = element.get("type", "")
            if element_type in ["page_header", "page_footer", "page_number"]:
                continue

            content = element.get("content", {})
            text = _get_text_value(content.get("text", ""))

            if text:
                text_lines.append(text)

        full_text = "\n".join(text_lines)
        if not full_text.strip():
            logger.warning("无法提取有效文本生成文档概要：%s", file_name)
            return ""

        try:
            summary_prompt_template = self._get_summary_prompt_by_language(language)
            prompt_text = summary_prompt_template.format(
                text=full_text[:JSON_IMAGE_DESCRIPTION_SUMMARY_LENGTH]
            )  # 限制文本长度

            response = await self.llm_summary.ainvoke(
                [HumanMessage(content=prompt_text)]
            )
            summary = response.content.strip()

            logger.info(
                "%s 文档概要生成完成，长度: %d 字符",
                file_name,
                len(summary),
            )
            return summary

        except Exception as e:
            logger.exception("文档概要生成失败：%s", str(e))
            return ""

    # ---------------------- 核心处理方法 ----------------------

    async def process_single_json(
        self,
        json_path: Path,
        output_dir: Optional[PathLike] = None,
    ) -> ImageDescriptionResult:
        """处理单个 JSON 文件，为所有包含 image_path 的元素生成描述。

        Args:
            json_path: JSON 文件路径
            output_dir: 输出目录

        Returns:
            ImageDescriptionResult 处理结果
        """
        output_path = Path(output_dir) / json_path.name if output_dir else json_path

        try:
            # 读取 JSON 文件
            if not json_path.exists():
                raise ImageDescriptionError(f"JSON 文件不存在：{json_path}")

            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)

            # 获取文档信息
            doc_id = json_data.get("metadata", {}).get("doc_id", json_path.stem)
            language = json_data.get("metadata", {}).get("language", "en")

            logger.info("开始处理 %s (语言: %s)", doc_id, language)

            # 查找包含 image_path 的元素
            all_elements = self._find_elements_with_image_path(json_data)
            if not all_elements:
                logger.info("%s 没有需要处理的图片元素", doc_id)
                return ImageDescriptionResult(
                    element_id="",
                    description="",
                    success=True,
                    error_message="无图片元素",
                )

            # 过滤出需要处理的图片（没有 description 或 description 为空）
            # 注意：elem 是 ElementInfo 对象，需要通过 id 从原始 JSON 中查找 description
            elements_to_process = []
            for elem in all_elements:
                # 在原始 JSON 中查找对应元素以获取 description
                existing_desc = ""
                for original_elem in json_data.get("elements", []):
                    if isinstance(original_elem, dict) and original_elem.get("id") == elem.id:
                        existing_desc = original_elem.get("content", {}).get("description", "")
                        break

                if not existing_desc or not existing_desc.strip():
                    elements_to_process.append(elem)

            if not elements_to_process:
                logger.info("%s 所有 %d 个图片已有描述，跳过处理", doc_id, len(all_elements))
                return ImageDescriptionResult(
                    element_id=doc_id,
                    description="",
                    success=True,
                    error_message="所有图片已有描述",
                )

            logger.info("%s 找到 %d 个图片元素，其中 %d 个需要处理", doc_id, len(all_elements), len(elements_to_process))

            # 提取或生成文档摘要
            abstract = self._extract_abstract_from_json(json_data, language)
            if abstract:
                document_summary = abstract
                logger.debug("使用 JSON 中的摘要")
            else:
                document_summary = await self._generate_document_summary(
                    json_data, language, doc_id
                )
                if not document_summary:
                    document_summary = "无法获取文档摘要"

            # 设置提示词（针对第一个元素设置提示词，后续按类型选择）
            original_prompt = self.prompt
            original_language = self.current_language
            self.prompt = self._get_prompt_by_language_and_type(language, "figure")
            self.current_language = language

            try:
                # 并发处理需要处理的图片（根据每张图片的类型选择对应提示词）
                tasks = []
                for element in elements_to_process:
                    img_type = _detect_image_type(element.image_path, element.type)
                    prompt = self._get_prompt_by_language_and_type(language, img_type)
                    tasks.append(
                        self._call_llm_for_description(
                            element.image_path,
                            document_summary,
                            prompt,
                        )
                    )

                descriptions = await asyncio.gather(*tasks, return_exceptions=True)

                # 只更新需要处理的元素的 description 字段
                for element, desc in zip(elements_to_process, descriptions):
                    if isinstance(desc, Exception):
                        logger.warning(
                            "处理元素 %s 失败: %s",
                            element.id,
                            str(desc),
                        )
                        desc_result = ""
                    else:
                        desc_result = desc

                    # 更新 JSON 中对应元素的 description
                    for elem in json_data["elements"]:
                        if not isinstance(elem, dict):
                            continue
                        elem_id = elem.get("id", "")
                        if elem_id == element.id:
                            if "content" not in elem:
                                elem["content"] = {}
                            elem["content"]["description"] = desc_result
                            break

            finally:
                self.prompt = original_prompt
                self.current_language = original_language

            # 保存处理后的 JSON
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            logger.info(
                "%s 处理完成：共 %d 个图片，%d 个已有描述，%d 个新处理",
                doc_id,
                len(all_elements),
                len(all_elements) - len(elements_to_process),
                len(elements_to_process),
            )

            return ImageDescriptionResult(
                element_id=doc_id,
                description="",
                success=True,
            )

        except Exception as e:
            logger.exception("处理 JSON 文件失败：%s", json_path)
            return ImageDescriptionResult(
                element_id=json_path.stem,
                description="",
                success=False,
                error_message=str(e),
            )

    # ---------------------- 批量处理方法 ----------------------

    async def batch_process(
        self,
        input_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        skip_existing: bool = True,
    ) -> BatchProcessResult:
        """批量处理目录下所有 JSON 文件。

        Args:
            input_dir: 输入目录（JSON 文件所在目录）
            output_dir: 输出目录
            skip_existing: 是否跳过已处理的文件（根据 parse_stage 判断）

        Returns:
            BatchProcessResult 批量处理结果统计
        """
        input_path = Path(input_dir) if input_dir else self.json_store_dir
        output_path = Path(output_dir) if output_dir else self.output_dir

        if not input_path.exists():
            raise ImageDescriptionError(f"输入目录不存在：{input_path}")

        # 查找所有 JSON 文件
        json_files = [
            p.resolve()
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".json"
        ]

        if not json_files:
            logger.warning("在 %s 中未找到 JSON 文件", input_path)
            return BatchProcessResult(
                total_files=0,
                success_count=0,
                skip_count=0,
                fail_count=0,
                total_images=0,
                success_images=0,
                fail_images=0,
            )

        logger.info("开始批量处理，共找到 %d 个 JSON 文件", len(json_files))

        # 顺序处理每个文件
        success_count = 0
        skip_count = 0
        fail_count = 0
        total_images = 0
        success_images = 0
        fail_images = 0

        for i, json_file in enumerate(sorted(json_files), start=1):
            logger.info("[%d/%d] 处理：%s", i, len(json_files), json_file.name)

            # 根据 parse_stage 判断是否跳过
            if should_skip_stage(json_file, STAGE_IMAGE_DESCRIPTION, skip_existing):
                logger.info(
                    "跳过（parse_stage 已完成）：%s (当前阶段: %s)",
                    json_file.name,
                    (
                        json_file.parent / json_file.name
                        if hasattr(json_file, "parent")
                        else json_file
                    ),
                )
                skip_count += 1

                # 统计已处理文件中的图片数量
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    images = self._find_elements_with_image_path(existing_data)
                    total_images += len(images)
                    success_images += len(images)
                except Exception:
                    pass
                continue

            try:
                result = await self.process_single_json(
                    json_path=json_file,
                    output_dir=output_path,
                )

                # 处理成功后更新 parse_stage
                if result.success and result.error_message != "无图片元素":
                    update_parse_stage(str(json_file), STAGE_IMAGE_DESCRIPTION)

                if result.success:
                    if result.error_message in ("无图片元素", "所有图片已有描述"):
                        skip_count += 1
                        # 统计已跳过文件中的图片数量
                        try:
                            output_file = output_path / json_file.name
                            with output_file.open("r", encoding="utf-8") as f:
                                processed_data = json.load(f)
                            images = self._find_elements_with_image_path(processed_data)
                            total_images += len(images)
                            success_images += len(images)
                        except Exception:
                            pass
                    else:
                        success_count += 1
                        # 统计图片数量
                        output_file = output_path / json_file.name
                        if output_file.exists():
                            try:
                                with output_file.open("r", encoding="utf-8") as f:
                                    processed_data = json.load(f)
                                images = self._find_elements_with_image_path(
                                    processed_data
                                )
                                total_images += len(images)
                                # 正确统计：遍历元素，检查每个元素的 description
                                for element in processed_data.get("elements", []):
                                    if not isinstance(element, dict):
                                        continue
                                    source = element.get("source", {})
                                    if source.get("image_path"):
                                        desc = element.get("content", {}).get(
                                            "description", ""
                                        )
                                        if desc:
                                            success_images += 1
                                        else:
                                            fail_images += 1
                            except Exception:
                                pass
                else:
                    fail_count += 1

            except Exception as e:
                logger.error("处理失败：%s, 错误信息：%r", json_file.name, e)
                fail_count += 1

        logger.info(
            "批量处理完成：文件总数=%d, 成功=%d, 跳过=%d, 失败=%d, 图片总数=%d, 成功=%d, 失败=%d",
            len(json_files),
            success_count,
            skip_count,
            fail_count,
            total_images,
            success_images,
            fail_images,
        )

        return BatchProcessResult(
            total_files=len(json_files),
            success_count=success_count,
            skip_count=skip_count,
            fail_count=fail_count,
            total_images=total_images,
            success_images=success_images,
            fail_images=fail_images,
        )

    # ---------------------- 同步接口（兼容性） ----------------------

    def run(self) -> None:
        """执行批量处理的主流程（同步版本）。"""
        asyncio.run(self.batch_process(skip_existing=True))


# ---------------------- 独立运行调试示例 ----------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    processor = JsonImageDescriptionProcessor()

    # 单个文件处理
    # await processor.process_single_json(
    #     json_path=Path("json_store/基于遗传算法的校园路径规划研究.json"),
    #     output_dir=Path("json_store"),
    # )

    # 批量处理
    processor.run()
