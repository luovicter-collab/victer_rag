# src/data_initialization/processors/json_fragment_merger.py

"""
JSON 片段合并器

对 layout_json_parser 输出的 JSON 进行修复：将本应连贯的句子或单词合并为同一块。

功能：
1. 句子未结束合并：若 paragraph 的 text 未以句末标点结尾（中文 。！？ / 英文 . ! ?），
   则与后续 paragraph 块合并，直到出现完整句子。
2. 英文断词合并：若一块以 "-" 结尾（跨行断词），与下一块合并为一个块（去掉连接处的 "-"）。
3. 合并后重新编号元素 id，并更新 metadata.total_elements 与 metadata.region_division 的序号映射。

执行顺序：在 layout_json_parser 之后、imagedescription_from_json 之前。
"""

import asyncio
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.settings import PROJECT_ROOT, STAGE_FRAGMENT_MERGED

# ---------------------- 类型与日志配置 ----------------------

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

# 句末标点（中英文）
_SENTENCE_END_CHARS = "。！？.?!"
# 英文断词连接符
_HYPHEN = "-"


def _get_text_from_element(el: Dict[str, Any]) -> str:
    """从元素 content 中取出 text，统一为字符串。"""
    content = el.get("content") or {}
    if not isinstance(content, dict):
        return ""
    text = content.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and text:
        return str(text[0]).strip() if text[0] is not None else ""
    return str(text).strip() if text is not None else ""


def _set_text_in_element(el: Dict[str, Any], text: str) -> None:
    """设置元素的 content.text。"""
    if "content" not in el:
        el["content"] = {}
    el["content"]["text"] = text


def _is_mergeable_paragraph(el: Dict[str, Any]) -> bool:
    """是否为可参与合并的 paragraph（有 text）。"""
    if not isinstance(el, dict) or el.get("type") != "paragraph":
        return False
    t = _get_text_from_element(el)
    return t is not None and len(t.strip()) > 0


def _text_ends_with_sentence_end(text: str) -> bool:
    """文本是否以句末标点结尾（去除尾部空白后）。"""
    t = text.rstrip()
    if not t:
        return False
    return t[-1] in _SENTENCE_END_CHARS


def _text_ends_with_hyphen(text: str) -> bool:
    """文本去尾空白后是否以 '-' 结尾（英文断词）。"""
    t = text.rstrip()
    if not t:
        return False
    return t.endswith(_HYPHEN)


def _should_merge_with_next(current_text: str, next_el: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    判断是否应将下一块合并到当前块。

    Returns:
        (should_merge, reason_is_hyphen): 若 reason_is_hyphen 为 True 表示因断词合并（连接时去 hyphen）。
    """
    if not _is_mergeable_paragraph(next_el):
        return False, False
    next_text = _get_text_from_element(next_el)
    if not next_text.strip():
        return False, False

    # 英文断词：当前块以 "-" 结尾
    if _text_ends_with_hyphen(current_text):
        return True, True
    # 句子未结束：当前块不以句末标点结尾
    if not _text_ends_with_sentence_end(current_text):
        return True, False
    return False, False


def _merge_text(current_text: str, next_text: str, use_hyphen_join: bool) -> str:
    """合并两段文本：use_hyphen_join 为 True 时去掉当前块尾的 '-' 并直接拼接下一块。"""
    curr = current_text.rstrip()
    nxt = next_text.lstrip()
    if use_hyphen_join:
        curr = curr.rstrip(_HYPHEN).rstrip()
        return curr + nxt
    return curr + (" " if curr and nxt else "") + nxt


def _merge_elements(elements: List[Dict[str, Any]], doc_id: str) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """
    对 elements 进行片段合并，并生成旧序号到新序号的映射（1-based）。

    - 只合并 type=paragraph 且有 text 的块。
    - 合并后保留第一个块的 source/metadata，仅扩展 content.text；合并后重新编号 id。
    """
    if not elements:
        return [], {}

    new_list: List[Dict[str, Any]] = []
    old_to_new: Dict[int, int] = {}
    i = 0
    new_seq = 1

    while i < len(elements):
        el = elements[i]
        if not _is_mergeable_paragraph(el):
            new_list.append(copy.deepcopy(el))
            old_to_new[i + 1] = new_seq
            new_seq += 1
            i += 1
            continue

        # 从当前块开始向后合并
        merged = copy.deepcopy(el)
        current_text = _get_text_from_element(merged)
        merge_start = i
        j = i + 1

        while j < len(elements):
            should, is_hyphen = _should_merge_with_next(current_text, elements[j])
            if not should:
                break
            next_text = _get_text_from_element(elements[j])
            current_text = _merge_text(current_text, next_text, is_hyphen)
            j += 1

        _set_text_in_element(merged, current_text)
        # 合并后更新 char_count（若存在）
        if merged.get("metadata") is not None and isinstance(merged["metadata"], dict):
            merged["metadata"]["char_count"] = len(current_text)
        new_list.append(merged)
        for k in range(merge_start, j):
            old_to_new[k + 1] = new_seq
        new_seq += 1
        i = j

    return new_list, old_to_new


def _renumber_element_ids(elements: List[Dict[str, Any]], doc_id: str) -> None:
    """将 elements 的 id 重排为 doc_id_elem_000001, doc_id_elem_000002, ..."""
    for idx, el in enumerate(elements):
        if isinstance(el, dict):
            el["id"] = f"{doc_id}_elem_{idx + 1:06d}"


def _remap_region_division(region_division: Dict[str, Any], old_to_new: Dict[int, int]) -> Dict[str, Any]:
    """根据 old_to_new 重映射 region_division 中各区的 start_seq / end_seq。"""
    if not region_division or not old_to_new:
        return region_division
    out = {}
    for key in ("head", "body", "tail"):
        part = region_division.get(key)
        if not isinstance(part, dict):
            out[key] = part
            continue
        start = part.get("start_seq")
        end = part.get("end_seq")
        new_start = old_to_new.get(start, start) if isinstance(start, int) else start
        new_end = old_to_new.get(end, end) if isinstance(end, int) else end
        out[key] = {"start_seq": new_start, "end_seq": new_end}
    return out


# ---------------------- 异常与主类 ----------------------


class JsonFragmentMergerError(Exception):
    """JSON 片段合并过程中的统一异常基类."""
    pass


class JsonFragmentMerger:
    """
    对 layout_json_parser 输出的 JSON 进行片段合并修复。

    - 句子未结束：与后续 paragraph 合并直到出现句末标点。
    - 英文断词：以 "-" 结尾的块与下一块合并（去 hyphen 连接）。
    - 合并后重编号 id、更新 total_elements 与 region_division。
    """

    def __init__(self, json_store_dir: Optional[PathLike] = None) -> None:
        if json_store_dir is None:
            self.json_store_dir = PROJECT_ROOT / "files" / "file_store" / "json_store"
        else:
            self.json_store_dir = Path(json_store_dir).resolve()

    def process_single_file(
        self,
        json_path: PathLike,
        output_dir: Optional[PathLike] = None,
    ) -> bool:
        """
        处理单个 JSON 文件：合并片段并写回。

        Args:
            json_path: 输入 JSON 路径
            output_dir: 输出目录，默认与输入同目录

        Returns:
            是否成功
        """
        path = Path(json_path)
        if not path.is_file() or path.suffix.lower() != ".json":
            logger.warning("跳过非 JSON 文件：%s", path)
            return False

        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.exception("读取 JSON 失败：%s", path)
            raise JsonFragmentMergerError(f"读取 JSON 失败: {path}") from e

        elements = data.get("elements", [])
        if not isinstance(elements, list):
            logger.warning("无有效 elements：%s", path)
            return True

        metadata = data.get("metadata", {})
        doc_id = metadata.get("doc_id", path.stem)

        new_elements, old_to_new = _merge_elements(elements, doc_id)
        _renumber_element_ids(new_elements, doc_id)

        data["elements"] = new_elements
        metadata["total_elements"] = len(new_elements)
        metadata["parse_stage"] = STAGE_FRAGMENT_MERGED

        if "region_division" in metadata and old_to_new:
            metadata["region_division"] = _remap_region_division(
                metadata["region_division"], old_to_new
            )

        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.exception("写入 JSON 失败：%s", out_path)
            raise JsonFragmentMergerError(f"写入 JSON 失败: {out_path}") from e

        logger.debug("片段合并已写回：%s，元素数 %d -> %d", out_path.name, len(elements), len(new_elements))
        return True

    def batch_process(
        self,
        input_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        skip_existing: bool = False,
    ) -> Tuple[int, int, int]:
        """
        批量处理目录下所有 JSON 文件。

        Args:
            input_dir: 输入目录，默认 self.json_store_dir
            output_dir: 输出目录，默认与 input_dir 相同
            skip_existing: 若为 True 则跳过已存在输出（此处与输入同目录时一般不跳过）

        Returns:
            (成功数, 跳过数, 失败数)
        """
        base = Path(input_dir) if input_dir else self.json_store_dir
        if not base.exists():
            raise JsonFragmentMergerError(f"目录不存在：{base}")

        out = Path(output_dir) if output_dir else base
        json_files = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".json")

        success = 0
        skip = 0
        fail = 0

        for jpath in json_files:
            try:
                if skip_existing and (out / jpath.name).exists():
                    skip += 1
                    continue
                self.process_single_file(jpath, output_dir=out)
                success += 1
            except JsonFragmentMergerError:
                fail += 1
            except Exception as e:
                logger.error("处理失败：%s, 错误：%r", jpath.name, e)
                fail += 1

        logger.info(
            "批量片段合并完成：总数=%d, 成功=%d, 跳过=%d, 失败=%d",
            len(json_files),
            success,
            skip,
            fail,
        )
        return success, skip, fail

    async def async_batch_process(
        self,
        input_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        skip_existing: bool = False,
    ) -> Tuple[int, int, int]:
        """异步执行批量片段合并（在线程池中运行同步逻辑）。"""
        return await asyncio.to_thread(
            self.batch_process,
            input_dir=input_dir,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )
