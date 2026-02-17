# src/data_initialization/processors/layout_json_parser.py

"""
文档元素提取器

从 MinerU 生成的多个 JSON 文件中融合数据，提取文档元素并转换为 RAG 嵌入数据格式。

功能：
1. 从 4 个 JSON 文件读取数据（content_list_v2.json, content_list.json, model.json, layout.json）
2. 融合多源数据，按字段优先级提取
3. 生成符合 RAG 嵌入数据格式的 JSON

输出格式：
{
  "metadata": {
    "doc_id": "文档ID",
    "doc_title": "文档标题",
    "parse_stage": "layout_json_parsed",
    "language": "zh/en",
    "source_file": "原始PDF文件名",
    "pdf_path": "PDF文件绝对路径",
    "total_pages": 页数,
    "total_elements": 元素总数
  },
  "elements": [
    {
      "id": "doc_id_elem_000001",
      "type": "paragraph/title/table/image/code/equation",
      "content": {...},
      "source": {
        "file": "PDF文件名",
        "page": 页码,
        "bbox": {x1, y1, x2, y2},
        "section_title": "所属标题（可选）",
        "image_path": "绝对路径（可选）"
      },
      "metadata": {...}
    }
  ]
}
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.config.settings import PROJECT_ROOT, STAGE_LAYOUT_JSON_PARSED

# ---------------------- 类型与日志配置 ----------------------

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)


# ---------------------- 自定义异常 ----------------------


class ElementExtractorError(Exception):
    """元素提取过程中的统一异常基类。"""

    pass


# ---------------------- 数据结构 ----------------------


@dataclass
class DocumentMetadata:
    """
    文档元数据信息。
    """

    doc_id: str
    doc_title: str
    parse_stage: str
    language: str
    source_file: str
    pdf_path: str  # PDF 文件绝对路径
    total_pages: int
    total_elements: int


@dataclass
class ElementSource:
    """
    元素溯源信息。
    """

    file: str
    page: int
    bbox: Dict[str, int]
    image_path: Optional[str] = None
    # 所属标题（用于非标题元素）
    section_title: Optional[str] = None


@dataclass
class ElementMetadata:
    """
    元素元数据。
    """

    confidence: Optional[float] = None
    char_count: Optional[int] = None
    # 表格相关
    table_type: Optional[str] = None
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    # 代码相关
    line_count: Optional[int] = None
    language: Optional[str] = None
    # 公式相关
    format: Optional[str] = None


@dataclass
class DocumentElement:
    """
    文档元素。
    """

    id: str
    type: str
    content: Dict[str, Any]
    source: ElementSource
    metadata: Optional[ElementMetadata] = None


@dataclass
class ExtractionResult:
    """
    单个文档元素提取结果。
    """

    elements: List[DocumentElement]
    metadata: DocumentMetadata


# ---------------------- 工具函数 ----------------------


def to_absolute_path(work_dir: Path, relative_path: str) -> str:
    """将相对路径转换为绝对路径。"""
    if not relative_path:
        return ""
    if Path(relative_path).is_absolute():
        return relative_path.replace("\\", "/")
    return str((work_dir / relative_path).resolve()).replace("\\", "/")


def detect_language(text: str) -> str:
    """
    简单检测文本语言。

    通过统计中文字符比例判断：
    - 中文字符比例 > 0.3 → 中文 (zh)
    - 否则 → 英文 (en)
    """
    if not text:
        return "en"

    # 计算中文字符数量
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total_chars = len(text)

    if total_chars == 0:
        return "en"

    ratio = chinese_chars / total_chars
    return "zh" if ratio > 0.3 else "en"


def generate_element_id(doc_id: str, index: int) -> str:
    """生成有序的元素 ID。

    ID 格式：{doc_id}_elem_{序号}
    示例：基于遗传算法的校园路径规划研究_elem_000001

    优点：
    1. 不同文献的元素 ID 不会冲突
    2. 通过 ID 前缀可以快速筛选同一文献的元素
    3. 保留顺序信息，便于获取附近块
    """
    return f"{doc_id}_elem_{index:06d}"


def _content_text_is_empty(content: Dict[str, Any]) -> bool:
    """判断 content 的 text 是否为空。仅当存在 text 键且其值为空或仅空白时返回 True（该块会被过滤）。无 text 键的块（如 table）保留。"""
    if not isinstance(content, dict):
        return False
    if "text" not in content:
        return False
    return not str(content["text"]).strip()


def filter_empty_text_elements_and_renumber(
    elements: List[DocumentElement], doc_id: str
) -> List[DocumentElement]:
    """
    过滤掉 content.text 为空的块，并重新编号使 id 连续。

    - 保留：content 无 "text" 键的块（如 table 仅有 html）、或 content.text 非空。
    - 移除：content.text 存在且为空或仅空白。
    - 对保留的块按顺序重新生成 id：doc_id_elem_000001, doc_id_elem_000002, ...
    """
    filtered = [
        e
        for e in elements
        if not _content_text_is_empty(e.content)
    ]
    for i, e in enumerate(filtered):
        e.id = generate_element_id(doc_id, i + 1)
    return filtered


def parse_bbox(
    bbox_list: List[float], page_size: Optional[List[int]] = None
) -> Dict[str, int]:
    """
    解析 bbox 坐标。

    Args:
        bbox_list: [x1, y1, x2, y2] 或归一化坐标 [0-1]
        page_size: 页面尺寸 [宽, 高]，用于归一化坐标转换

    Returns:
        {x1, y1, x2, y2}
    """
    if not bbox_list or len(bbox_list) < 4:
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    # 检查是否为归一化坐标（值在 0-1 之间）
    is_normalized = all(0 <= v <= 1 for v in bbox_list[:4])

    if is_normalized and page_size:
        width, height = page_size
        return {
            "x1": int(bbox_list[0] * width),
            "y1": int(bbox_list[1] * height),
            "x2": int(bbox_list[2] * width),
            "y2": int(bbox_list[3] * height),
        }

    return {
        "x1": int(bbox_list[0]),
        "y1": int(bbox_list[1]),
        "x2": int(bbox_list[2]),
        "y2": int(bbox_list[3]),
    }


def bboxs_overlap(bbox1: Dict[str, int], bbox2: List, tolerance: int = 50) -> bool:
    """
    判断两个 bbox 是否重叠（用于匹配同一资源的多个块）。

    Args:
        bbox1: 第一个 bbox {x1, y1, x2, y2}
        bbox2: 第二个 bbox [x1, y1, x2, y2]
        tolerance: 容差（像素）

    Returns:
        是否重叠
    """
    if not bbox2 or len(bbox2) < 4:
        return False

    # 计算中心点距离
    cx1 = (bbox1["x1"] + bbox1["x2"]) / 2
    cy1 = (bbox1["y1"] + bbox1["y2"]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2

    # 计算距离
    distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    # 如果距离小于容差，认为是同一资源
    return distance < tolerance


def _is_abstract_section(section_title: Optional[str]) -> Optional[str]:
    """判断 section_title 是否为摘要小节，并返回对应语言。

    通过 source.section_title 识别摘要部分，支持中英文及常见变体。

    Args:
        section_title: 元素所属小节标题

    Returns:
        "zh" 表示中文摘要，"en" 表示英文摘要，非摘要小节返回 None
    """
    if not section_title:
        return None
    s = section_title.strip()
    if not s:
        return None
    if "摘要" in s:
        return "zh"
    if s.lower().startswith("abstract"):
        return "en"
    return None


def extract_abstract_from_elements(
    elements: List[DocumentElement],
    doc_language: str,
) -> Optional[Union[str, List[Dict[str, str]]]]:
    """从已提取的元素中根据 section_title 收集摘要内容。

    仅收集 type 为 paragraph、且 source.section_title 为摘要/Abstract 的块。
    若存在多语言摘要（如既有「摘要」又有「Abstract」），返回带 language 的列表，
    供下游按文献语言选用；仅有一种摘要时返回单个字符串。

    Args:
        elements: 文档元素列表（DocumentElement）
        doc_language: 文档语言 "zh" / "en"，仅用于注释，不改变返回值结构

    Returns:
        - 无摘要: None
        - 仅一种摘要: 该摘要文本（str）
        - 多语言摘要: [{"language": "zh", "text": "..."}, {"language": "en", "text": "..."}]
    """
    lang_texts: Dict[str, List[str]] = {}

    for element in elements:
        if element.type != "paragraph":
            continue
        lang = _is_abstract_section(element.source.section_title)
        if lang is None:
            continue
        raw = element.content.get("text")
        if raw is None:
            continue
        if isinstance(raw, list):
            text = " ".join(str(t).strip() for t in raw if t).strip()
        else:
            text = str(raw).strip()
        if text:
            lang_texts.setdefault(lang, []).append(text)

    if not lang_texts:
        return None

    # 每种语言下多段合并为一段
    joined = {
        lang: "\n".join(parts).strip()
        for lang, parts in lang_texts.items()
    }

    if len(joined) == 1:
        return next(iter(joined.values()))
    return [{"language": lang, "text": text} for lang, text in sorted(joined.items())]


# ---------------------- 主工具类 ----------------------


class ElementExtractor:
    """
    文档元素提取器。

    从 MinerU 生成的多个 JSON 文件中融合数据，提取文档元素。

    功能：
    1. 并行加载 4 个 JSON 文件
    2. 融合多源数据（content_list_v2.json 为主，其他为辅）
    3. 生成符合 RAG 嵌入数据格式的 JSON

    用法：

        extractor = ElementExtractor()
        result = await extractor.extract_from_doc(
            doc_name="基于遗传算法的校园路径规划研究",
            work_dir=PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_work",
        )
        print(f"提取了 {result.metadata.total_elements} 个元素")
    """

    # JSON 文件名常量
    CONTENT_LIST_V2 = "content_list_v2.json"
    CONTENT_LIST_JSON = "{uuid}_content_list.json"
    MODEL_JSON = "{uuid}_model.json"
    LAYOUT_JSON = "layout.json"

    # 类型映射：将不同 JSON 中的类型统一
    TYPE_MAPPING = {
        "text": "paragraph",
        "paragraph": "paragraph",
        "title": "title",
        "table": "table",
        "image": "image",
        "figure": "image",
        "code": "code",
        "equation": "equation",
        "header": "page_header",
        "page_header": "page_header",
        "page_footer": "page_footer",
        "page_number": "page_number",
        "reference": "reference",
    }

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        work_dir: Optional[PathLike] = None,
    ) -> None:
        """
        初始化 ElementExtractor。

        Args:
            output_dir: 输出目录（默认：PROJECT_ROOT/files/file_store/json_store）
            work_dir: MinerU work 目录（用于查找 JSON 文件）
        """
        self._project_root = PROJECT_ROOT
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else self._project_root / "files" / "file_store" / "json_store"
        )
        self.work_dir = (
            Path(work_dir)
            if work_dir
            else self._project_root
            / "files"
            / "file_store"
            / "md_store"
            / "minerU_work"
        )

    # ---------------------- 文件加载 ----------------------

    async def _load_json(self, file_path: Path) -> Optional[Dict]:
        """异步加载 JSON 文件。"""
        try:
            if not file_path.exists():
                return None
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            return json.loads(content)
        except Exception as e:
            logger.warning("加载 JSON 文件失败：%s, 错误：%s", file_path, e)
            return None

    async def _load_content_list_v2(self, doc_dir: Path) -> List:
        """加载 content_list_v2.json。"""
        file_path = doc_dir / self.CONTENT_LIST_V2
        data = await self._load_json(file_path)
        if data is None:
            logger.warning("未找到 content_list_v2.json：%s", file_path)
            return []
        return data if isinstance(data, list) else []

    async def _load_content_list_json(self, doc_dir: Path, uuid: str) -> List:
        """加载 {uuid}_content_list.json。"""
        filename = self.CONTENT_LIST_JSON.format(uuid=uuid)
        file_path = doc_dir / filename
        data = await self._load_json(file_path)
        if data is None:
            return []
        return data if isinstance(data, list) else []

    async def _load_model_json(self, doc_dir: Path, uuid: str) -> List:
        """加载 {uuid}_model.json。"""
        filename = self.MODEL_JSON.format(uuid=uuid)
        file_path = doc_dir / filename
        data = await self._load_json(file_path)
        if data is None:
            return []
        return data if isinstance(data, list) else []

    async def _load_layout_json(self, doc_dir: Path) -> Dict:
        """加载 layout.json。"""
        file_path = doc_dir / self.LAYOUT_JSON
        data = await self._load_json(file_path)
        if data is None:
            logger.warning("未找到 layout.json：%s", file_path)
            return {}
        return data if isinstance(data, dict) else {}

    async def _load_all_json_files(self, doc_dir: Path, uuid: str) -> Dict[str, Any]:
        """
        并行加载所有 JSON 文件。

        Returns:
            {
                "content_list_v2": [...],
                "content_list_json": [...],
                "model_json": [...],
                "layout_json": {...}
            }
        """
        tasks = {
            "content_list_v2": self._load_content_list_v2(doc_dir),
            "content_list_json": self._load_content_list_json(doc_dir, uuid),
            "model_json": self._load_model_json(doc_dir, uuid),
            "layout_json": self._load_layout_json(doc_dir),
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        return {
            key: result if not isinstance(result, Exception) else []
            for key, result in zip(tasks.keys(), results)
        }

    # ---------------------- 元素提取核心逻辑 ----------------------

    def _extract_table_info(self, element: Dict, doc_dir: Path) -> Dict[str, Any]:
        """
        提取表格信息。

        优先从 content_list_v2 获取 html 和 table_type，
        从 content_list.json 获取 image_path。
        提取多个 caption（中文和英文），caption 只存 content 字符串。
        """
        result = {
            "html": "",
            "captions": [],  # 多个标题，格式：["表 4-1 白天判断矩阵", "Table 4-1 Daytime judgment matrix"]
            "description": "",
        }

        # 从 content_list_v2 获取
        if "content" in element:
            content = element["content"]
            if isinstance(content, dict):
                result["html"] = content.get("html", "") or content.get(
                    "table_body", ""
                )
                # 提取多个 caption
                captions = content.get("table_caption", [])
                if captions and isinstance(captions, list):
                    for cap in captions:
                        if isinstance(cap, dict):
                            cap_content = cap.get("content", "")
                            if cap_content:
                                result["captions"].append(cap_content)
                        elif isinstance(cap, str) and cap:
                            result["captions"].append(cap)

        # 从 content_list.json 获取 image_path（由调用方根据 bbox 匹配）

        return result

    def _extract_image_info(self, element: Dict) -> Dict[str, Any]:
        """
        提取图片信息。

        提取多个 caption（中文和英文），caption 只存 content 字符串。
        """
        result = {
            "captions": [],  # 多个标题，格式：["图 2-1 遗传算法流程图", "Figure 2-1 Flow chart of genetic algorithm"]
            "description": "",
        }

        # 从 content_list_v2 获取
        if "content" in element:
            content = element["content"]
            if isinstance(content, dict):
                # 提取多个 caption
                captions = content.get("image_caption", [])
                if captions and isinstance(captions, list):
                    for cap in captions:
                        if isinstance(cap, dict):
                            cap_content = cap.get("content", "")
                            if cap_content:
                                result["captions"].append(cap_content)
                        elif isinstance(cap, str) and cap:
                            result["captions"].append(cap)

        return result

    def _extract_code_info(self, element: Dict) -> Dict[str, Any]:
        """提取代码信息。"""
        result = {
            "text": "",
            "language": "",
            "description": "",
        }

        if "content" in element:
            content = element["content"]
            if isinstance(content, dict):
                # 从 content_list_v2 获取
                code_content = content.get("code_content", "")
                if code_content:
                    result["text"] = code_content
                    result["language"] = content.get("code_language", "")

                # 从 content_list.json 直接获取
                if not result["text"]:
                    result["text"] = element.get("code", "")
                    result["language"] = element.get("code_language", "")

        # sub_type
        sub_type = element.get("sub_type", "")

        return result, sub_type

    def _extract_equation_info(self, element: Dict) -> Dict[str, Any]:
        """提取公式信息。

        优先从 content.math_content 获取（content_list_v2.json 的 equation_interline），
        如果没有则从 element.text 获取（content_list.json 的 equation）。
        """
        result = {
            "text": "",
            "format": "latex",
            "description": "",
        }

        # 优先从 content_list_v2.json 的 content.math_content 获取
        if "content" in element and isinstance(element["content"], dict):
            math_content = element["content"].get("math_content", "")
            if math_content:
                # 添加 $$ 包裹，与 content_list.json 格式一致
                result["text"] = f"$$ {math_content} $$"
            result["format"] = element["content"].get("math_type", "latex")

        # 如果没有 content.math_content，从 element.text 获取
        if not result["text"]:
            result["text"] = element.get("text", "")

        result["format"] = element.get("text_format", result["format"])

        return result

    def _build_element_content(
        self, element: Dict, element_type: str
    ) -> Dict[str, Any]:
        """根据元素类型构建 content 对象。"""
        content = {}

        if element_type == "paragraph":
            # 优先从 content_list_v2 获取
            if "content" in element and isinstance(element["content"], dict):
                para_content = element["content"].get("paragraph_content", [])
                if para_content:
                    # 只提取纯文本，不保留 elements 数组（RAG 嵌入只需要可读文本）
                    text_parts = []
                    for item in para_content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("content", ""))
                            elif item.get("type") == "equation_inline":
                                text_parts.append(f"${item.get('content', '')}$")
                    content["text"] = "".join(text_parts)
                else:
                    content["text"] = element.get("text", "")
            else:
                content["text"] = element.get("text", "")

        elif element_type == "list":
            # 提取列表项文本
            if "content" in element and isinstance(element["content"], dict):
                list_items = element["content"].get("list_items", [])
                if list_items:
                    text_parts = []
                    for item in list_items:
                        if isinstance(item, dict):
                            item_content = item.get("item_content", [])
                            if isinstance(item_content, list):
                                for sub_item in item_content:
                                    if isinstance(sub_item, dict):
                                        text_parts.append(sub_item.get("content", ""))
                    content["text"] = "".join(text_parts)
            # 如果没有提取到，尝试直接从 element.text 获取
            if not content.get("text"):
                content["text"] = element.get("text", "")

        elif element_type == "title":
            if "content" in element and isinstance(element["content"], dict):
                title_content = element["content"].get("title_content", [])
                if title_content and isinstance(title_content, list):
                    text_parts = [
                        item.get("content", "")
                        for item in title_content
                        if isinstance(item, dict)
                    ]
                    content["text"] = "".join(text_parts)
                # 从 content_list_v2 获取 level 字段
                content["level"] = element["content"].get("level", 1)
            else:
                content["text"] = element.get("text", "")

        elif element_type == "table":
            # image_path 由调用方根据 bbox 匹配后添加
            table_info = self._extract_table_info(element, None)
            content.update(table_info)

        elif element_type == "image":
            # image_path 由调用方根据 bbox 匹配后添加
            image_info = self._extract_image_info(element)
            content.update(image_info)

        elif element_type == "code":
            code_info, sub_type = self._extract_code_info(element)
            content.update(code_info)

        elif element_type == "equation":
            equation_info = self._extract_equation_info(element)
            content.update(equation_info)

        elif element_type in ["page_header", "page_footer", "page_number"]:
            if "content" in element and isinstance(element["content"], dict):
                # 提取页眉/页脚/页码内容
                for key in [
                    "page_header_content",
                    "page_footer_content",
                    "page_number_content",
                ]:
                    if key in element["content"]:
                        items = element["content"][key]
                        if isinstance(items, list):
                            text_parts = [
                                item.get("content", "")
                                for item in items
                                if isinstance(item, dict)
                            ]
                            content["text"] = "".join(text_parts)
                        break
            else:
                content["text"] = element.get("text", "")

        else:
            # 默认提取 text 字段
            content["text"] = element.get("text", "")

        return content

    def _build_element_metadata(
        self, element: Dict, element_type: str
    ) -> ElementMetadata:
        """构建元素元数据。"""
        metadata = ElementMetadata()

        # 表格相关
        if element_type == "table":
            if "content" in element and isinstance(element["content"], dict):
                metadata.table_type = element["content"].get(
                    "table_type", "simple_table"
                )

            # 估算行列数
            html = element.get("content", {}).get("html", "") or element.get(
                "table_body", ""
            )
            if html:
                row_count = html.count("<tr>") - 1  # 减去表头
                col_count = html.count("</td>")
                if row_count > 0:
                    col_count = col_count // (row_count + 1)
                metadata.row_count = max(1, row_count)
                metadata.col_count = max(1, col_count)

        # 代码相关
        elif element_type == "code":
            text = ""
            if "content" in element and isinstance(element["content"], dict):
                text = element["content"].get("code_content", "")
            if not text:
                text = element.get("code", "")
            if text:
                metadata.line_count = text.count("\n") + 1
            metadata.language = element.get("code_language", "") or (
                element.get("content", {}).get("code_language", "")
                if element.get("content")
                else ""
            )

        # 公式相关
        elif element_type == "equation":
            metadata.format = element.get("text_format", "latex")

        # 文本相关
        elif element_type in ["paragraph", "title", "list"]:
            text = ""
            if "content" in element and isinstance(element["content"], dict):
                if element_type == "paragraph":
                    para_content = element["content"].get("paragraph_content", [])
                    text = "".join(
                        [
                            item.get("content", "")
                            for item in para_content
                            if isinstance(item, dict)
                        ]
                    )
                elif element_type == "title":
                    title_content = element["content"].get("title_content", [])
                    text = "".join(
                        [
                            item.get("content", "")
                            for item in title_content
                            if isinstance(item, dict)
                        ]
                    )
                elif element_type == "list":
                    list_items = element["content"].get("list_items", [])
                    for item in list_items:
                        if isinstance(item, dict):
                            item_content = item.get("item_content", [])
                            if isinstance(item_content, list):
                                text += "".join(
                                    [
                                        sub_item.get("content", "")
                                        for sub_item in item_content
                                        if isinstance(sub_item, dict)
                                    ]
                                )
            if not text:
                text = element.get("text", "")
            if text:
                metadata.char_count = len(text)

        return metadata

    def _normalize_element_type(self, element_type: str) -> str:
        """标准化元素类型。"""
        return self.TYPE_MAPPING.get(element_type, element_type)

    def _is_empty_element(self, element_type: str, content: Dict[str, Any]) -> bool:
        """
        判断元素是否为空（没有有效内容）。

        对于 paragraph, list, title：检查 text 是否为空
        对于 table, image, code, equation：检查是否有其他有效字段
        """

        def _get_text_value(value: Any) -> str:
            """安全获取文本值，处理列表和非字符串情况。"""
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, list):
                return " ".join([str(v) for v in value]).strip()
            elif value is None:
                return ""
            return str(value).strip()

        if element_type in ["paragraph", "list", "title"]:
            text = _get_text_value(content.get("text"))
            return not text

        elif element_type in ["table"]:
            # 表格：检查是否有 html
            html = _get_text_value(content.get("html"))
            return not html

        elif element_type in ["image"]:
            # 图片：检查是否有 captions 或 image_path
            has_captions = content.get("captions")
            has_image_path = content.get("image_path")
            return not (has_captions or has_image_path)

        elif element_type in ["code"]:
            # 代码：检查是否有 text
            text = _get_text_value(content.get("text"))
            return not text

        elif element_type in ["equation"]:
            # 公式：检查是否有 text
            text = _get_text_value(content.get("text"))
            return not text

        return False

    async def _extract_elements_from_content_list_v2(
        self,
        content_list_v2: List,
        content_list_json: List,
        layout_json: Dict,
        doc_dir: Path,
        doc_id: str,
    ) -> List[DocumentElement]:
        """
        从 content_list_v2 提取元素（主数据源）。

        使用 content_list_v2 作为主数据源，
        从 content_list.json 补充 image_path（根据 bbox 匹配），
        从 layout.json 补充置信度。

        特性：
        1. 为非 title 元素添加所属标题字段
        2. 根据 bbox 匹配正确的 image_path
        3. 过滤空元素
        """
        elements: List[DocumentElement] = []
        element_index = 1

        # 构建 content_list_json 索引（按页码分组，用于 bbox 匹配）
        # 结构：{type: {page_idx: [item1, item2, ...]}}
        content_list_index: Dict[str, Dict[int, List[Dict]]] = {}
        for item in content_list_json:
            item_type = item.get("type", "")
            page_idx = item.get("page_idx")
            if item_type in ["table", "image"] and page_idx is not None:
                if item_type not in content_list_index:
                    content_list_index[item_type] = {}
                if page_idx not in content_list_index[item_type]:
                    content_list_index[item_type][page_idx] = []
                content_list_index[item_type][page_idx].append(item)

        # 获取页面尺寸（用于坐标转换）
        page_size = None
        if layout_json.get("pdf_info") and len(layout_json["pdf_info"]) > 0:
            first_page = layout_json["pdf_info"][0]
            page_size = first_page.get("page_size", [595, 841])

        # 当前标题（用于非标题元素）
        current_section_title: Optional[str] = None

        for page_idx, page_elements in enumerate(content_list_v2):
            if not isinstance(page_elements, list):
                continue

            for element in page_elements:
                if not isinstance(element, dict):
                    continue

                # 获取原始类型并标准化
                original_type = element.get("type", "unknown")
                element_type = self._normalize_element_type(original_type)

                # 跳过页眉、页码（在 RAG 中通常不需要）
                if element_type in ["page_header", "page_footer", "page_number"]:
                    continue

                # 更新当前标题
                if element_type == "title":
                    title_text = ""
                    if "content" in element and isinstance(element["content"], dict):
                        title_content = element["content"].get("title_content", [])
                        if title_content and isinstance(title_content, list):
                            title_text = "".join(
                                [
                                    item.get("content", "")
                                    for item in title_content
                                    if isinstance(item, dict)
                                ]
                            )
                    if not title_text:
                        title_text = element.get("text", "")
                    current_section_title = title_text

                # 构建 source
                bbox = element.get("bbox", [])
                bbox_dict = parse_bbox(bbox, page_size)

                # 补充 image_path（从 content_list.json，根据 bbox 匹配）
                image_path = ""
                if element_type in ["table", "image"]:
                    # 获取该页该类型的所有项
                    type_page_items = content_list_index.get(element_type, {}).get(
                        page_idx, []
                    )
                    for item in type_page_items:
                        item_bbox = item.get("bbox", [])
                        if bboxs_overlap(bbox_dict, item_bbox):
                            img_path = item.get("img_path", "")
                            if img_path:
                                image_path = to_absolute_path(doc_dir, img_path)
                            break

                source = ElementSource(
                    file=f"{doc_id}.pdf",
                    page=page_idx,
                    bbox=bbox_dict,
                    image_path=image_path if image_path else None,
                    section_title=(
                        current_section_title if element_type != "title" else None
                    ),
                )

                # 构建 content
                content = self._build_element_content(element, element_type)

                # 构建 metadata
                metadata = self._build_element_metadata(element, element_type)

                # 过滤空元素
                if self._is_empty_element(element_type, content):
                    logger.debug("跳过空元素：type=%s, page=%d", element_type, page_idx)
                    continue

                # 创建元素（ID 带 doc_id 前缀，区分不同文献）
                element_id = generate_element_id(doc_id, element_index)
                doc_element = DocumentElement(
                    id=element_id,
                    type=element_type,
                    content=content,
                    source=source,
                    metadata=metadata,
                )

                elements.append(doc_element)
                element_index += 1

        return elements

    # ---------------------- 对外公共方法 ----------------------

    async def extract_from_doc(
        self,
        doc_name: str,
        work_dir: Optional[PathLike] = None,
    ) -> ExtractionResult:
        """
        从文档目录提取所有元素。

        Args:
            doc_name: 文档目录名（如 "基于遗传算法的校园路径规划研究"）
            work_dir: MinerU work 目录（默认使用初始化时的目录）

        Returns:
            ExtractionResult：包含元素列表和文档元数据
        """
        doc_dir = (Path(work_dir) if work_dir else self.work_dir) / doc_name

        if not doc_dir.exists():
            raise ElementExtractorError(f"文档目录不存在：{doc_dir}")

        # 查找 UUID（用于构建 content_list.json 和 model.json 文件名）
        uuid = ""
        for item in doc_dir.iterdir():
            if item.is_file() and item.name.endswith("_content_list.json"):
                uuid = item.name.replace("_content_list.json", "")
                break

        if not uuid:
            logger.warning("未找到 {uuid}_content_list.json 文件，将跳过部分字段补充")

        # 加载所有 JSON 文件
        json_data = await self._load_all_json_files(doc_dir, uuid)

        content_list_v2 = json_data.get("content_list_v2", [])
        content_list_json = json_data.get("content_list_json", [])
        layout_json = json_data.get("layout_json", {})

        if not content_list_v2:
            raise ElementExtractorError(f"content_list_v2.json 为空或不存在：{doc_dir}")

        # 提取元素
        elements = await self._extract_elements_from_content_list_v2(
            content_list_v2=content_list_v2,
            content_list_json=content_list_json,
            layout_json=layout_json,
            doc_dir=doc_dir,
            doc_id=doc_name,
        )

        # 检测语言（使用第一个段落的文本）
        language = "en"
        for element in elements:
            if element.type == "paragraph" and element.content.get("text"):
                language = detect_language(element.content["text"])
                break

        # 构建文档元数据
        total_pages = len(content_list_v2)
        # 获取 PDF 文件绝对路径
        pdf_dir = self._project_root / "files" / "file_store" / "pdf_store"
        pdf_path = (
            str((pdf_dir / f"{doc_name}.pdf").resolve()).replace("\\", "/")
            if (pdf_dir / f"{doc_name}.pdf").exists()
            else ""
        )

        metadata = DocumentMetadata(
            doc_id=doc_name,
            doc_title=doc_name,
            parse_stage=STAGE_LAYOUT_JSON_PARSED,
            language=language,
            source_file=f"{doc_name}.pdf",
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_elements=len(elements),
        )

        logger.info(
            "元素提取完成：doc_id=%s, pages=%d, elements=%d, language=%s",
            doc_name,
            total_pages,
            len(elements),
            language,
        )

        return ExtractionResult(elements=elements, metadata=metadata)

    async def extract_and_save(
        self,
        doc_name: str,
        work_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
    ) -> Path:
        """
        提取元素并保存为 JSON 文件。

        Args:
            doc_name: 文档目录名
            work_dir: MinerU work 目录
            output_dir: 输出目录（默认：PROJECT_ROOT/files/file_store/json_store）

        Returns:
            输出 JSON 文件路径
        """
        # 确定输出目录
        if output_dir is None:
            output_dir = self._project_root / "files" / "file_store" / "json_store"

        # 确保输出目录存在
        output_path = Path(output_dir)
        await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)

        # 提取元素
        result = await self.extract_from_doc(doc_name, work_dir)

        # 清理 content.text 为空的块，并重新编号使 id 连续
        elements_out = filter_empty_text_elements_and_renumber(
            result.elements, result.metadata.doc_id
        )

        # 构建输出数据
        output_data = {
            "metadata": {
                "doc_id": result.metadata.doc_id,
                "doc_title": result.metadata.doc_title,
                "parse_stage": result.metadata.parse_stage,
                "language": result.metadata.language,
                "source_file": result.metadata.source_file,
                "pdf_path": result.metadata.pdf_path,
                "total_pages": result.metadata.total_pages,
                "total_elements": len(elements_out),
            },
            "elements": [],
        }

        # 转换元素为字典
        for element in elements_out:
            element_dict = {
                "id": element.id,
                "type": element.type,
                "content": element.content,
                "source": {
                    "file": element.source.file,
                    "page": element.source.page,
                    "bbox": element.source.bbox,
                },
            }

            # 添加 section_title 到 source
            if element.source.section_title:
                element_dict["source"]["section_title"] = element.source.section_title

            # 添加 image_path
            if element.source.image_path:
                element_dict["source"]["image_path"] = element.source.image_path

            if element.metadata:
                metadata_dict = {}
                if element.metadata.confidence is not None:
                    metadata_dict["confidence"] = element.metadata.confidence
                if element.metadata.char_count is not None:
                    metadata_dict["char_count"] = element.metadata.char_count
                if element.metadata.table_type:
                    metadata_dict["table_type"] = element.metadata.table_type
                if element.metadata.row_count is not None:
                    metadata_dict["row_count"] = element.metadata.row_count
                if element.metadata.col_count is not None:
                    metadata_dict["col_count"] = element.metadata.col_count
                if element.metadata.line_count is not None:
                    metadata_dict["line_count"] = element.metadata.line_count
                if element.metadata.language:
                    metadata_dict["language"] = element.metadata.language
                if element.metadata.format:
                    metadata_dict["format"] = element.metadata.format

                if metadata_dict:
                    element_dict["metadata"] = metadata_dict

            output_data["elements"].append(element_dict)

        # 从元素中根据 section_title（摘要/Abstract）提取摘要并写入 metadata
        abstract = extract_abstract_from_elements(
            elements_out, result.metadata.language
        )
        if abstract is not None:
            output_data["metadata"]["abstract"] = abstract

        # 保存文件到 json_store 目录
        # 文件名与 minerU_md 目录下的 md 文件名一致
        output_file_name = f"{doc_name}.json"
        output_path = Path(output_dir) / output_file_name

        json_content = json.dumps(output_data, ensure_ascii=False, indent=2)
        await asyncio.to_thread(output_path.write_text, json_content, encoding="utf-8")

        logger.info("元素已保存到：%s", output_path)

        return output_path

    async def batch_extract(
        self,
        work_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        skip_existing: bool = True,
    ) -> List[Path]:
        """
        批量提取目录下所有文档的元素。

        Args:
            work_dir: MinerU work 目录
            output_dir: 输出目录
            skip_existing: 是否跳过已存在的文件

        Returns:
            输出文件路径列表
        """
        dir_path = Path(work_dir) if work_dir else self.work_dir

        if not dir_path.exists():
            raise ElementExtractorError(f"目录不存在：{dir_path}")

        # 确定输出目录
        if output_dir is None:
            output_dir = self._project_root / "files" / "file_store" / "json_store"
        else:
            output_dir = Path(output_dir)

        results: List[Path] = []
        success_count = 0
        fail_count = 0
        skip_count = 0

        for doc_dir in sorted(dir_path.iterdir()):
            if not doc_dir.is_dir():
                continue

            doc_name = doc_dir.name

            # 检查是否已存在（文件名与 md 文件一致）
            output_file = output_dir / f"{doc_name}.json"
            if skip_existing and output_file.exists():
                logger.info("跳过（已存在）：%s", doc_name)
                skip_count += 1
                continue

            try:
                output_path = await self.extract_and_save(
                    doc_name=doc_name,
                    work_dir=work_dir,
                    output_dir=output_dir,
                )
                results.append(output_path)
                success_count += 1
            except Exception as e:
                logger.error("处理失败：%s, 错误：%r", doc_name, e)
                fail_count += 1

        logger.info(
            "批量提取完成：总数=%d, 成功=%d, 跳过=%d, 失败=%d",
            success_count + skip_count + fail_count,
            success_count,
            skip_count,
            fail_count,
        )

        return results


# ---------------------- 独立运行调试示例 ----------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    extractor = ElementExtractor()

    async def main():
        # 单个文档提取
        result = await extractor.extract_from_doc(
            doc_name="基于遗传算法的校园路径规划研究",
        )
        print(f"提取了 {result.metadata.total_elements} 个元素")

        # 提取并保存
        output_path = await extractor.extract_and_save(
            doc_name="基于遗传算法的校园路径规划研究",
        )
        print(f"已保存到：{output_path}")

        # 批量提取
        results = await extractor.batch_extract()
        print(f"批量处理完成，共 {len(results)} 个文件")

    asyncio.run(main())
