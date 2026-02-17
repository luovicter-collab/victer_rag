# src/data_initialization/processors/region_extractor.py

"""
JSON 区域划分与标题提取

从 json_store 目录下的 JSON 文件中识别标题与小节标记、划分 head/body/tail 三区，并将边界写入 metadata。

功能：
1. 扫描指定目录下所有 .json 文件，解析 elements 中的 type == "title" 及任意带 content.text 的块
2. 正文区域识别：body 从「1 Introduction/1 绪论」到「参考文献/References」前一块
3. 三区边界：head（正文前）、body（正文）、tail（正文后），序号均从 1 开始、含首含尾
4. 将 region_division（head/body/tail 的 start_seq、end_seq）写入每个 JSON 的 metadata 并保存
5. 支持单文件/目录批量提取与控制台输出

普适规则（标题在 text 开头的情况）：
- 小节标记可在 type=title 或 paragraph 等块的 content.text 开头（如 "ABSTRACT: ..."、"参考文献"）。
- 凡以摘要/ABSTRACT、目录/Contents、参考文献/References、附录/致谢等开头的块，均参与划分：
  head 含摘要与目录等，body 不含这些块；tail 含参考文献与附录等，body 在其前结束。
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.settings import PROJECT_ROOT, STAGE_REGION_DIVIDED

# ---------------------- 类型与日志配置 ----------------------

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

# ---------------------- 自定义异常 ----------------------


class TitleExtractorError(Exception):
    """标题提取过程中的统一异常基类。"""

    pass


# ---------------------- 数据结构 ----------------------


@dataclass
class TitleItem:
    """单个标题元素信息。"""

    doc_id: str
    element_id: str
    text: str
    level: int
    seq: int  # JSON 中 elements 数组里的块序号（从 1 开始）
    page: int
    source_file: str


# 从非 title 块（如 paragraph）开头识别的区域标记：seq（1-based）, role
SectionMarker = Tuple[
    int, str
]  # role: "ref" | "toc" | "tail" | "body_start" | "abstract"


@dataclass
class DocTitleResult:
    """单文档的标题提取结果。"""

    json_path: str
    doc_id: str
    titles: List[TitleItem] = field(default_factory=list)
    section_markers: List[SectionMarker] = field(
        default_factory=list
    )  # 非 title 块开头的标记（如 ABSTRACT:、References）
    body_region: Optional["BodyRegion"] = None  # 正文区域，由 detect_body_region 填充


@dataclass
class AllTitlesResult:
    """目录下全部 JSON 的标题汇总结果。"""

    json_store_dir: str
    total_files: int
    total_titles: int
    doc_results: List[DocTitleResult] = field(default_factory=list)


@dataclass
class BodyRegion:
    """正文区域：从 start_seq 到 end_seq 的 JSON 块范围（均含）。"""

    start_seq: int
    end_seq: int
    method: str  # "references" | "max_span"


@dataclass
class RegionDivision:
    """三个区域的边界：head（正文前）、body（正文）、tail（正文后）。序号均从 1 开始，含首含尾。"""

    head_start_seq: int
    head_end_seq: int
    body_start_seq: int
    body_end_seq: int
    tail_start_seq: int
    tail_end_seq: int

    def to_metadata_dict(self) -> Dict[str, Any]:
        """转为可写入 JSON metadata 的字典（不含 method）。"""
        return {
            "head": {"start_seq": self.head_start_seq, "end_seq": self.head_end_seq},
            "body": {"start_seq": self.body_start_seq, "end_seq": self.body_end_seq},
            "tail": {"start_seq": self.tail_start_seq, "end_seq": self.tail_end_seq},
        }


# 正文结束标记：参考文献/References 等（目录与参考文献均不属于正文）
_REF_NORMALIZED_ZH = ("参考文献", "參考文獻", "参考 文献", "引用文献", "参考资料")
_REF_NORMALIZED_EN = (
    "references",
    "bibliography",
    "works cited",
    "references and notes",
)

# 目录标题：正文起始必须在其后（英文不用 "content" 单用，避免 "Content available at" 等误判）
_TOC_NORMALIZED_ZH = ("目录", "目次")
_TOC_NORMALIZED_EN = ("contents", "table of contents")
_TOC_NORMALIZED_EN_STRICT = (
    "contents",
    "table of contents",
)  # 块开头识别时用，不含 content

# 附录/致谢等：也可作为正文结束（若未找到参考文献）
_TAIL_START_ZH = ("附录", "致谢", "致 谢", "鸣谢")
_TAIL_START_EN = (
    "appendix",
    "appendices",
    "acknowledgement",
    "acknowledgments",
    "acknowledgement(s)",
)

# 正文起始：第一个章节标题（排除摘要、目录等）
_BODY_START_1_INTRO_EN = re.compile(
    r"^\s*1\s*[\.．]?\s*introduction\s*$", re.IGNORECASE
)
_BODY_START_1_ZH = re.compile(r"^1\s*[\.．]?\s*绪论\s*[：:]?\s*$")
_BODY_START_CHAPTER = re.compile(r"^\s*(chapter|part)\s+[i1]\s*$", re.IGNORECASE)
_BODY_START_INTRO_STANDALONE = re.compile(r"^\s*introduction\s*$", re.IGNORECASE)
_BODY_START_ZH_CHAPTER = re.compile(
    r"^[一二三1ⅠⅠ]\s*[、．.．]?\s*.*"
)  # 第一章、一、第一部分、1. xxx
_BODY_START_NUMBERED = re.compile(r"^\s*1\s*[\.．]\s*\S+")  # 1. xxx 开头（非 TOC 行）


# ---------------------- 工具类 ----------------------


class JsonTitleExtractor:
    """
    JSON 区域划分与标题提取工具类。

    从 json_store 中的 JSON 文件内识别 title 元素、划分 head/body/tail，
    并将 region_division 写入各 JSON 的 metadata。
    """

    def __init__(self, json_store_dir: Optional[PathLike] = None) -> None:
        if json_store_dir is None:
            self.json_store_dir = PROJECT_ROOT / "files" / "file_store" / "json_store"
        else:
            self.json_store_dir = Path(json_store_dir).resolve()

    @staticmethod
    def _normalize_zh(t: str) -> str:
        """中文标题归一化：去空格、全角。"""
        return t.replace(" ", "").replace("\u3000", "").strip()

    @staticmethod
    def _normalize_en(t: str) -> str:
        """英文标题归一化：小写、去首尾空白。"""
        return t.lower().strip()

    @staticmethod
    def _is_references_title(text: str) -> bool:
        """判断是否为参考文献类标题（正文结束，不属于正文）。支持块开头为「References」「REFERENCES:」等。排除目录行（参考文献 … 22）。"""
        if not text or not text.strip():
            return False
        t = text.strip()
        # 目录行特征：含省略号且像页码结尾（如「参考文献 … 22」），不算正文后的参考文献节
        if ("…" in t or "..." in t) and len(t) < 50 and re.search(r"\d\s*$", t):
            return False
        t_zh = JsonTitleExtractor._normalize_zh(text)
        for p in _REF_NORMALIZED_ZH:
            if p in t_zh or t_zh == p or t_zh.startswith(p):
                return True
        t_en = JsonTitleExtractor._normalize_en(text)
        for p in _REF_NORMALIZED_EN:
            if (
                t_en == p
                or (len(t_en) <= len(p) + 2 and p in t_en)
                or t_en.startswith(p)
            ):
                return True
        return False

    @staticmethod
    def _is_toc_title(text: str) -> bool:
        """判断是否为目录标题（正文起始必须在其后，目录不属于正文）。"""
        if not text or not text.strip():
            return False
        t_zh = JsonTitleExtractor._normalize_zh(text)
        for p in _TOC_NORMALIZED_ZH:
            if p in t_zh or t_zh == p:
                return True
        t_en = JsonTitleExtractor._normalize_en(text)
        for p in _TOC_NORMALIZED_EN:
            if t_en == p or p in t_en:
                return True
        return False

    @staticmethod
    def _is_toc_section_header(text: str) -> bool:
        """判断是否为目录小节头（仅用于块开头），比 _is_toc_title 更严：英文仅认 contents/table of contents，避免 Content 开头的段落误判。"""
        if not text or not text.strip():
            return False
        t_zh = JsonTitleExtractor._normalize_zh(text)
        for p in _TOC_NORMALIZED_ZH:
            if t_zh == p or t_zh.startswith(p) and len(t_zh) <= 10:
                return True
        t_en = JsonTitleExtractor._normalize_en(text)
        for p in _TOC_NORMALIZED_EN_STRICT:
            if t_en == p or t_en.startswith(p) or (len(t_en) <= 20 and p in t_en):
                return True
        return False

    @staticmethod
    def _is_tail_start_title(text: str) -> bool:
        """判断是否为附录/致谢类（正文结束后的 tail 起始，用于无参考文献时的回退）。排除目录行（致谢 … 页码）。"""
        if not text or not text.strip():
            return False
        t = text.strip()
        if ("…" in t or "..." in t) and len(t) < 50 and re.search(r"\d\s*$", t):
            return False
        t_zh = JsonTitleExtractor._normalize_zh(text)
        for p in _TAIL_START_ZH:
            if p in t_zh or t_zh == p:
                return True
        t_en = JsonTitleExtractor._normalize_en(text)
        for p in _TAIL_START_EN:
            if t_en == p or (len(t_en) <= len(p) + 3 and p in t_en):
                return True
        return False

    @staticmethod
    def _is_front_matter_title(text: str) -> bool:
        """判断是否为前置部分（摘要、Abstract 等），不作为正文起始。含块开头为「ABSTRACT:」「摘要：」等情形。"""
        if not text or not text.strip():
            return False
        t_zh = JsonTitleExtractor._normalize_zh(text)
        if t_zh in ("摘要", "abstract", "关键词", "摘要与关键词") or t_zh.startswith(
            "摘要"
        ):
            return True
        t_en = JsonTitleExtractor._normalize_en(text)
        if t_en in ("abstract", "keywords", "key words") or t_en.startswith("abstract"):
            return True
        return False

    @staticmethod
    def _get_leading_section_label(text: str, max_len: int = 100) -> str:
        """取块文本开头的「小节标签」：首行或前 max_len 字符，去尾空白及末尾冒号。用于从 paragraph 等非 title 块识别 ABSTRACT、References 等。"""
        if not text or not text.strip():
            return ""
        first_line = text.strip().split("\n")[0].strip()
        label = (
            first_line[:max_len].strip() if len(first_line) > max_len else first_line
        )
        return label.rstrip("：:").strip()

    @staticmethod
    def _is_body_start_title(text: str) -> bool:
        """判断是否为正文起始标题（多种文献格式）。排除目录行、摘要等。"""
        if not text or not text.strip():
            return False
        t = text.strip()
        # 目录行特征：省略号、结尾冒号或页码、或「1 绪论 ：」这类 TOC 行
        if "…" in t or "..." in t:
            return False
        if re.search(r"[：:]\s*$", t) or re.search(r"\d\s*$", t):
            return False
        # 中文 TOC 行：仅「1 绪论」+ 冒号/空格（无正文），如「1 绪论 ：」
        if "绪论" in t and ("：" in t or ":" in t) and len(t) < 20:
            return False
        if JsonTitleExtractor._is_front_matter_title(t):
            return False
        if JsonTitleExtractor._is_toc_title(
            t
        ) or JsonTitleExtractor._is_references_title(t):
            return False
        # 英文：1 Introduction / 1. Introduction / Introduction / Chapter 1 / Part 1
        if _BODY_START_1_INTRO_EN.search(t) or _BODY_START_INTRO_STANDALONE.search(t):
            return True
        if _BODY_START_CHAPTER.search(t):
            return True
        # 中文：1 绪论（不含冒号的正文章节标题）/ 第一章 / 一、/ 1. xxx
        if _BODY_START_1_ZH.search(t):
            return True
        t_zh = JsonTitleExtractor._normalize_zh(t)
        if t_zh == "1绪论" or (len(t_zh) >= 2 and t_zh[0] == "1" and "绪论" in t_zh):
            return True
        if t_zh.startswith("1") and ("引言" in t_zh or "概述" in t_zh):
            return True
        if _BODY_START_ZH_CHAPTER.search(t) and len(t) <= 30:
            return True
        if _BODY_START_NUMBERED.search(t) and len(t) <= 80:
            return True
        return False

    @staticmethod
    def _is_major_body_start(text: str) -> bool:
        """仅「第一章/1 绪论/1 Introduction」级别，用于栈匹配；排除目录行及 2、3 章等。"""
        if not text or not text.strip():
            return False
        t = text.strip()
        if (
            "…" in t
            or "..." in t
            or re.search(r"[：:]\s*$", t)
            or re.search(r"\d\s*$", t)
        ):
            return False
        if "绪论" in t and ("：" in t or ":" in t) and len(t) < 20:
            return False
        if JsonTitleExtractor._is_front_matter_title(
            t
        ) or JsonTitleExtractor._is_toc_title(t):
            return False
        if JsonTitleExtractor._is_references_title(t):
            return False
        if _BODY_START_1_INTRO_EN.search(t) or _BODY_START_INTRO_STANDALONE.search(t):
            return True
        if _BODY_START_CHAPTER.search(t):
            return True
        if _BODY_START_1_ZH.search(t):
            return True
        t_zh = JsonTitleExtractor._normalize_zh(t)
        if t_zh == "1绪论" or (len(t_zh) >= 2 and t_zh[0] == "1" and "绪论" in t_zh):
            return True
        if (
            t_zh.startswith("1")
            and ("引言" in t_zh or "概述" in t_zh)
            and len(t_zh) < 15
        ):
            return True
        return False

    def _detect_body_region_by_stack(
        self,
        titles: List[TitleItem],
        markers: List[SectionMarker],
        ref_seqs: List[int],
        tail_seqs: List[int],
        all_seqs: List[int],
        after_head: int,
    ) -> Optional[Tuple[int, int]]:
        """
        用栈匹配「正文起」与「正文止」：目录与正文各出现一对起止，取 span 最大的那对。
        - 事件：start = 仅「1 绪论/1 Introduction」级；end = 仅参考文献（正文止于 References 前，不含致谢/附录）。
        - 记录所有配对，取 (end - start) 最大的一对，保证选到正文边界而非目录。
        - 单边括号：无配对时返回 None，走 fallback。
        """
        start_events = [
            (t.seq, "start") for t in titles if self._is_major_body_start(t.text)
        ]
        # 正文止于「参考文献」前一块，只用 ref 作为 end，不用 tail（致谢/附录在 References 之后）
        end_events = [(s, "end") for s in ref_seqs]
        events = start_events + end_events
        if not events:
            return None
        events.sort(key=lambda x: (x[0], (0 if x[1] == "start" else 1)))

        stack: List[int] = []
        pairs: List[Tuple[int, int]] = []
        for seq, kind in events:
            if kind == "start":
                stack.append(seq)
            else:
                if stack:
                    start_seq = stack.pop()
                    pairs.append((start_seq, seq))

        if not pairs:
            return None
        # 取 span（end - start）最大的一对，即正文的起止，而非目录的起止
        best = max(pairs, key=lambda p: p[1] - p[0])
        body_start, end_marker_seq = best
        body_end = max(1, end_marker_seq - 1)
        return (body_start, body_end)

    def detect_body_region(self, doc_result: DocTitleResult) -> BodyRegion:
        """
        识别正文区域 [start_seq, end_seq]（含首含尾）。

        普适规则：目录与正文中「起/止」标题各出现一次，用栈取最内层一对，避免目录被算进正文；
        单边括号（缺起或缺止）时用回退逻辑。
        """
        titles = sorted(doc_result.titles, key=lambda x: x.seq)
        markers: List[SectionMarker] = doc_result.section_markers or []
        all_seqs = [t.seq for t in titles] + [seq for seq, _ in markers]
        if not all_seqs:
            return BodyRegion(start_seq=1, end_seq=1, method="default")

        ref_seqs = [t.seq for t in titles if self._is_references_title(t.text)] + [
            s for s, r in markers if r == "ref"
        ]
        toc_seqs = [t.seq for t in titles if self._is_toc_title(t.text)] + [
            s for s, r in markers if r == "toc"
        ]
        tail_seqs = [t.seq for t in titles if self._is_tail_start_title(t.text)] + [
            s for s, r in markers if r == "tail"
        ]
        abstract_seqs = [s for s, r in markers if r == "abstract"] + [
            t.seq for t in titles if self._is_front_matter_title(t.text)
        ]
        toc_seq = max(toc_seqs) if toc_seqs else 0
        after_head = 0
        if abstract_seqs or toc_seqs:
            after_head = max(abstract_seqs + toc_seqs) + 1

        # 栈匹配得到最内层 (body_start, body_end)
        pair = self._detect_body_region_by_stack(
            titles, markers, ref_seqs, tail_seqs, all_seqs, after_head
        )

        if pair is not None:
            start_seq, end_seq = pair
            if after_head > 0 and after_head <= end_seq:
                start_seq = max(start_seq, after_head)
            return BodyRegion(start_seq=start_seq, end_seq=end_seq, method="detected")

        # 单边括号或无法成对：回退
        if ref_seqs:
            end_seq = max(1, max(ref_seqs) - 1)
        elif tail_seqs:
            end_seq = max(1, min(tail_seqs) - 1)
        else:
            end_seq = max(all_seqs)

        body_start_seqs = [
            t.seq for t in titles if self._is_body_start_title(t.text)
        ] + [s for s, r in markers if r == "body_start"]
        start_candidates = [s for s in body_start_seqs if s > toc_seq and s <= end_seq]
        if start_candidates:
            start_seq = min(start_candidates)
        elif ref_seqs or tail_seqs:
            start_seq = 1
        elif titles:
            max_span = -1
            best_start = titles[0].seq
            best_end = best_start
            for i in range(len(titles)):
                for j in range(i + 1, len(titles)):
                    if titles[j].seq > toc_seq and titles[i].seq > toc_seq:
                        span = titles[j].seq - titles[i].seq
                        if span > max_span:
                            max_span = span
                            best_start = titles[i].seq
                            best_end = titles[j].seq
            start_seq = best_start
            end_seq = best_end
        else:
            start_seq = 1

        if after_head > 0 and after_head <= end_seq:
            start_seq = max(start_seq, after_head)
        return BodyRegion(start_seq=start_seq, end_seq=end_seq, method="detected")

    @staticmethod
    def _region_division_from_body(
        body_region: BodyRegion,
        total_elements: int,
        elements: Optional[List[Dict[str, Any]]] = None,
    ) -> "RegionDivision":
        """
        根据正文区域和元素总数生成 head/body/tail 三区边界。
        不修改 body。若 head 为空或 tail 为空，则用 page 回退：
        - head 为空时：取 source.page==0 的所有块作为 head；
        - tail 为空时：取 source.page==最后一页 的所有块作为 tail。
        """
        b_start = body_region.start_seq
        b_end = body_region.end_seq
        head_end = max(0, b_start - 1)
        tail_start = min(total_elements + 1, b_end + 1)
        tail_end = total_elements

        if elements and total_elements > 0:
            # 收集每个 seq 对应的 page（seq 从 1 开始）
            seq_to_page: Dict[int, int] = {}
            for i, el in enumerate(elements):
                if not isinstance(el, dict):
                    continue
                p = el.get("source", {}).get("page", 0)
                if not isinstance(p, int):
                    try:
                        p = int(p)
                    except (TypeError, ValueError):
                        p = 0
                seq_to_page[i + 1] = p

            last_page = max(seq_to_page.values()) if seq_to_page else 0
            seqs_page0 = [s for s, p in seq_to_page.items() if p == 0]
            seqs_last_page = [s for s, p in seq_to_page.items() if p == last_page]

            # head 为空：用 page=0 的块作为 head
            if head_end < 1 and seqs_page0:
                head_end = max(seqs_page0)

            # tail 为空：用最后一页的块作为 tail
            if tail_start > tail_end and seqs_last_page:
                tail_start = min(seqs_last_page)
                tail_end = total_elements

        return RegionDivision(
            head_start_seq=1,
            head_end_seq=head_end,
            body_start_seq=b_start,
            body_end_seq=b_end,
            tail_start_seq=tail_start,
            tail_end_seq=tail_end,
        )

    def _write_region_division_to_json(
        self, json_path: PathLike, data: Dict[str, Any], division: RegionDivision
    ) -> None:
        """将区域划分写入 JSON 的 metadata 并保存。"""
        path = Path(json_path)
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["region_division"] = division.to_metadata_dict()
        data["metadata"]["parse_stage"] = STAGE_REGION_DIVIDED
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("已写入 region_division 到 %s", path.name)
        except Exception as e:
            logger.warning("写入 region_division 失败 %s: %s", path.name, e)

    def _get_text_from_content(self, content: Any) -> str:
        if not isinstance(content, dict):
            return ""
        text_field = content.get("text")
        if isinstance(text_field, str):
            return text_field.strip()
        if isinstance(text_field, list) and text_field:
            return str(text_field[0]).strip()
        return str(text_field).strip() if text_field is not None else ""

    def _get_level_from_content(self, content: Any) -> int:
        if not isinstance(content, dict):
            return 1
        level = content.get("level")
        if isinstance(level, int) and level >= 1:
            return level
        if isinstance(level, (float, str)):
            try:
                return int(float(level)) if int(float(level)) >= 1 else 1
            except (ValueError, TypeError):
                pass
        return 1

    def extract_from_file(self, json_path: PathLike) -> DocTitleResult:
        """从单个 JSON 文件中提取 title 元素、识别 body 区域，并将 head/body/tail 写入 metadata。"""
        path = Path(json_path)
        if not path.exists():
            raise TitleExtractorError(f"JSON 文件不存在：{path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise TitleExtractorError(f"JSON 解析失败：{path}，错误：{e}") from e

        metadata = data.get("metadata", {})
        doc_id = metadata.get("doc_id", path.stem)
        source_file = metadata.get("source_file", path.name)

        titles: List[TitleItem] = []
        elements = data.get("elements", [])
        if not isinstance(elements, list):
            return DocTitleResult(json_path=str(path), doc_id=doc_id, titles=[])

        for idx, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            if element.get("type") != "title":
                continue

            content = element.get("content", {})
            text = self._get_text_from_content(content)
            level = self._get_level_from_content(content)

            source = element.get("source", {})
            page = source.get("page", 0)
            if not isinstance(page, int):
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    page = 0

            seq = idx + 1

            item = TitleItem(
                doc_id=doc_id,
                element_id=element.get("id", ""),
                text=text,
                level=level,
                seq=seq,
                page=page,
                source_file=source_file,
            )
            titles.append(item)

        # 从非 title 块（如 paragraph）开头识别区域标记，例如 "ABSTRACT: ..."、"References"
        section_markers: List[SectionMarker] = []
        for idx, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            content = element.get("content", {})
            text = self._get_text_from_content(content)
            if not text.strip():
                continue
            label = self._get_leading_section_label(text)
            if not label:
                continue
            seq = idx + 1
            if self._is_references_title(label):
                section_markers.append((seq, "ref"))
            elif self._is_toc_section_header(label):
                section_markers.append((seq, "toc"))
            elif self._is_tail_start_title(label):
                section_markers.append((seq, "tail"))
            elif self._is_front_matter_title(label):
                section_markers.append((seq, "abstract"))
            elif self._is_body_start_title(label):
                section_markers.append((seq, "body_start"))

        logger.debug(
            "从 %s 提取到 %d 个标题、%d 个块开头标记",
            path.name,
            len(titles),
            len(section_markers),
        )
        result = DocTitleResult(
            json_path=str(path),
            doc_id=doc_id,
            titles=titles,
            section_markers=section_markers,
        )
        result.body_region = self.detect_body_region(result)

        if result.body_region:
            total_elements = len(elements)
            division = self._region_division_from_body(
                result.body_region, total_elements, elements=elements
            )
            self._write_region_division_to_json(path, data, division)

        return result

    def extract_from_dir(
        self,
        json_store_dir: Optional[PathLike] = None,
    ) -> AllTitlesResult:
        """从目录下所有 JSON 文件中提取 title、划分区域，并写入各文件 metadata。"""
        base_dir = Path(json_store_dir) if json_store_dir else self.json_store_dir
        if not base_dir.exists():
            raise TitleExtractorError(f"目录不存在：{base_dir}")

        json_files = sorted(
            p.resolve()
            for p in base_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".json"
        )

        doc_results: List[DocTitleResult] = []
        total_titles = 0

        for jpath in json_files:
            try:
                doc_result = self.extract_from_file(jpath)
                doc_results.append(doc_result)
                total_titles += len(doc_result.titles)
            except TitleExtractorError as e:
                logger.warning("跳过文件 %s：%s", jpath.name, e)

        return AllTitlesResult(
            json_store_dir=str(base_dir),
            total_files=len(json_files),
            total_titles=total_titles,
            doc_results=doc_results,
        )

    def print_titles(
        self,
        result: AllTitlesResult,
        indent: str = "  ",
        include_seq: bool = True,
    ) -> None:
        """将 AllTitlesResult 以可读形式输出到控制台。"""
        print(f"目录: {result.json_store_dir}")
        print(f"文件数: {result.total_files}, 标题总数: {result.total_titles}")
        print("-" * 60)

        for doc in result.doc_results:
            print(f"\n【{doc.doc_id}】 (共 {len(doc.titles)} 个标题)")
            if doc.body_region:
                print(
                    f"{indent}正文区域: 序号 {doc.body_region.start_seq} ~ {doc.body_region.end_seq} ({doc.body_region.method})"
                )
            for t in doc.titles:
                seq_str = f" [序号 {t.seq}]" if include_seq else ""
                level_prefix = "  " * (t.level - 1) if t.level > 1 else ""
                line = f"{indent}{level_prefix}- {t.text}{seq_str}"
                try:
                    print(line)
                except UnicodeEncodeError:
                    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
                    sys.stdout.buffer.write(line.encode(enc, errors="replace") + b"\n")

    def get_titles_flat(self, result: AllTitlesResult) -> List[Dict[str, Any]]:
        """将 AllTitlesResult 展平为字典列表。"""
        flat: List[Dict[str, Any]] = []
        for doc in result.doc_results:
            for t in doc.titles:
                flat.append(
                    {
                        "doc_id": t.doc_id,
                        "element_id": t.element_id,
                        "text": t.text,
                        "level": t.level,
                        "seq": t.seq,
                        "page": t.page,
                        "source_file": t.source_file,
                        "json_path": doc.json_path,
                    }
                )
        return flat

    def get_body_regions_flat(self, result: AllTitlesResult) -> List[Dict[str, Any]]:
        """返回每个文档的正文区域。"""
        out: List[Dict[str, Any]] = []
        for doc in result.doc_results:
            if doc.body_region:
                out.append(
                    {
                        "doc_id": doc.doc_id,
                        "json_path": doc.json_path,
                        "start_seq": doc.body_region.start_seq,
                        "end_seq": doc.body_region.end_seq,
                        "method": doc.body_region.method,
                    }
                )
        return out


# ---------------------- 独立运行示例 ----------------------

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    extractor = JsonTitleExtractor()
    result = extractor.extract_from_dir()
    extractor.print_titles(result)
