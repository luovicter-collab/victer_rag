"""
LangGraph Agentic-RAG 状态定义模块（对话滚动摘要 + 工作记忆 / 语义记忆）。

本模块实现与《Agentic-RAG 多轮对话上下文与记忆管理方案书》一致的状态建模：

1. 不直接在状态中拼接完整历史对话，只保留：
   - 结构化的短期工作记忆（Working Memory）
   - 来自长期语义记忆的召回视图（Semantic Memory View）
   - 最近若干轮“最终问答”摘要（qa_turns + conversation_summary）

2. 将记忆系统拆为三层（其中 Raw Dialogue Log 建议由外部持久化系统负责）：
   - Raw Dialogue Log：完整日志，不进入 prompt（不在此模块建模）；
   - Short-term / Working Memory：当前任务相关的聚焦、要素、开放问题等；
   - Long-term / Semantic Memory：稳定、经验证的原子事实视图（本轮召回）。

3. 与 Agentic-RAG 三阶段对齐：
   - 阶段一：问题理解 & Working Memory 初始化
   - 阶段二：子任务拆解 + 混合检索 + 证据评估 + Retry
   - 阶段三：答案生成 + 阶段摘要（episodic summary）+ 是否写入长期记忆
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Literal, Optional, TypedDict, Union

# ---------------------- 类型与日志配置 ----------------------

logger = logging.getLogger(__name__)

# ---------------------- 常量定义（集中管理枚举值） ----------------------

PhaseLiteral = Literal["query_understanding", "retrieval", "answer_generation"]
RetrievalModeLiteral = Literal["hybrid", "hyde", "step_back"]
RetryStrategyLiteral = Literal["hyde", "step_back", "expand_sub_questions"]

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_PARENT_CHILD_THRESHOLD: int = 3
DEFAULT_SUB_QUESTION_COUNT: int = 5

# 滚动摘要 / 对话管理相关默认值
SUMMARY_BLOCK_SIZE: int = 5  # 每多少问答轮就进行一次“summary + 清空 qa_turns”
MAX_QA_TURNS: int = 5  # qa_turns 最大长度（与你的策略一致，就是 5）

# ---------------------- 数据结构定义 ----------------------


class RetrievedDoc(TypedDict, total=False):
    """
    单条检索结果的通用结构。

    说明：
        - 不强制依赖特定向量库 / 检索实现；
        - 只抽象出 Agent 可能使用到的公共字段。
    """

    doc_id: str  # 文档 / 片段所属的文档 ID（建议与预处理 JSON 的 doc_id 对齐）
    score: float  # 检索得分（向量相似度 / BM25 分数等，统一为“越大越相关”）
    content: str  # 片段内容文本
    metadata: Dict[str, object]  # 任意额外元数据（页码、标题、路径等）


# ---------------------- 阶段一：问题理解与优化 ----------------------


class QueryUnderstandingState(TypedDict, total=False):
    """
    阶段一：问题理解与优化相关状态。

    Attributes:
        original_query: 当前这一轮的用户原始查询（必须存在）。
        rewritten_query: 语义重写后的查询（如果有）。
        final_query: 最终用于检索 / 回答的问题：
                     一般为 rewritten_query，否则回退到 original_query。
        query_quality_score: 问题质量评分（0-1 之间的浮点数）。
        is_query_clear: 问题是否清晰（Agent 决策结果）。
        needs_human_intervention: 是否需要人工介入补充信息。
        human_supplemented_info: 人工补充的关键信息（如果有）。
        query_rewrite_attempts: 查询重写尝试次数。
    """

    original_query: str
    rewritten_query: Optional[str]
    final_query: Optional[str]
    query_quality_score: Optional[float]
    is_query_clear: Optional[bool]
    needs_human_intervention: Optional[bool]
    human_supplemented_info: Optional[str]
    query_rewrite_attempts: int


class RetrievalState(TypedDict, total=False):
    """
    阶段二：文档检索相关状态。

    注意：检索步骤默认采用混合检索（关键字 + 向量），由 Agent 决定权重。

    Attributes:
        sub_questions: 任务拆解后的子问题列表（必做，默认 5 个）。
        sub_question_count: 子问题数量（默认 5，Retry 时可增加额外子问题）。
        hybrid_retrieval_results:
            混合检索结果（关键字 + 向量加权），格式：
            {sub_question: [RetrievedDoc, ...]}。
        sparse_retrieval_results:
            稀疏检索（关键字）结果，格式：
            {sub_question: [RetrievedDoc, ...]}。
        dense_retrieval_results:
            稠密检索（向量）结果，格式：
            {sub_question: [RetrievedDoc, ...]}。
        retrieval_weights:
            Agent 决定的检索权重，格式 {"sparse": float, "dense": float}，总和应约等于 1.0。
        parent_doc_replacements:
            父文档替换结果，格式 {sub_question: parent_doc_id}，
            当某子问题命中同一父文档的片段数量 >= parent_child_threshold 时，可替换为父文档。
        hit_sub_blocks_count:
            命中的子块数量，格式 {sub_question: 命中块数量}。
        parent_child_threshold:
            父文档替换阈值（命中子块数 >= 阈值时使用父文档）。
        evidence_sufficiency_score:
            证据充分性评分（0-1 之间的浮点数）。
        is_evidence_sufficient:
            检索结果是否足以回答（Agent 决策结果，用于触发 Retry）。
        retrieval_mode:
            当前检索模式：
                - "hybrid": 默认混合检索（关键字 + 向量）。
                - "hyde": Retry 时启用的假设性文档嵌入检索。
                - "step_back": Retry 时启用的抽象提升检索。
    """

    sub_questions: List[str]
    sub_question_count: int
    hybrid_retrieval_results: Dict[str, List[RetrievedDoc]]
    sparse_retrieval_results: Dict[str, List[RetrievedDoc]]
    dense_retrieval_results: Dict[str, List[RetrievedDoc]]
    retrieval_weights: Dict[str, float]  # 约定只包含 "sparse" 与 "dense" 两个键
    parent_doc_replacements: Dict[str, Optional[str]]
    hit_sub_blocks_count: Dict[str, int]
    parent_child_threshold: int
    evidence_sufficiency_score: Optional[float]
    is_evidence_sufficient: Optional[bool]
    retrieval_mode: RetrievalModeLiteral


# ---------------------- Retry / 补偿机制 ----------------------


class RetryState(TypedDict, total=False):
    """
    Retry / 补偿机制相关状态。

    注意：Retry 机制用于在「证据不足」时追加检索或改变检索策略。
    可用策略包括：
    - HyDE（假设性文档嵌入检索）
    - Step-back（抽象提升检索）
    - 增加额外的子问题进行检索

    Attributes:
        retry_triggered: 是否触发了 Retry 机制（通常在 is_evidence_sufficient=False 时触发）。
        retry_count: Retry 次数。
        retry_strategy: 当前选择的 Retry 补偿策略。
        hyde_enabled: 是否启用 HyDE（假设性文档嵌入检索）。
        step_back_enabled: 是否启用 Step-back（抽象提升检索）。
        step_back_query: Step-back 抽象后的查询。
        expanded_sub_questions_enabled:
            是否启用扩充子问题（增加额外的子问题进行检索）。
        additional_sub_questions:
            Retry 时新增的额外子问题列表。
        expanded_top_k:
            扩充后的 Top-K 检索规模（如果启用扩充检索）。
    """

    retry_triggered: bool
    retry_count: int
    retry_strategy: Optional[RetryStrategyLiteral]
    hyde_enabled: bool
    step_back_enabled: bool
    step_back_query: Optional[str]
    expanded_sub_questions_enabled: bool
    additional_sub_questions: List[str]
    expanded_top_k: Optional[int]


# ---------------------- 阶段三：答案生成与聚合 ----------------------


class AnswerGenerationState(TypedDict, total=False):
    """
    阶段三：答案生成与聚合相关状态。

    Attributes:
        selected_prompt_template: 动态选择的 Prompt 模板名称。
        prompt_template_type: Prompt 模板类型（基于问题类型或子问题类型）。
        sub_answers:
            每个子问题的答案映射，格式 {sub_question: answer}。
        aggregated_answer: 聚合后的初步答案（多子问题答案融合）。
        refined_answer: 润色后的最终答案（面向用户输出）。
        generation_attempts: 答案生成尝试次数（用于控制重试）。
    """

    selected_prompt_template: Optional[str]
    prompt_template_type: Optional[str]
    sub_answers: Dict[str, str]
    aggregated_answer: Optional[str]
    refined_answer: Optional[str]
    generation_attempts: int


# ---------------------- 系统级状态（流程控制、错误处理） ----------------------


class SystemState(TypedDict, total=False):
    """
    系统级状态（流程控制、错误处理等）。

    Attributes:
        current_phase: 当前执行阶段。
        phase_history: 阶段执行历史记录（按时间顺序追加）。
        error_message: 错误信息（如果有）。
        max_retries: 最大 Retry 次数。
        execution_start_time: 执行开始时间戳（time.time()）。
        execution_end_time: 执行结束时间戳（time.time()），尚未结束则为 None。
    """

    current_phase: PhaseLiteral
    phase_history: List[str]
    error_message: Optional[str]
    max_retries: int
    execution_start_time: Optional[float]
    execution_end_time: Optional[float]


# ---------------------- 多轮对话与滚动摘要管理（仅问答轮） ----------------------


class QATurn(TypedDict, total=False):
    """
    单轮「问答」结构（只记录最终问题和最终回答）。

    Attributes:
        question: 用户最终问题（本轮的 original_query / rewritten_query）。
        answer: AI 给出的最终回答文本。
        timestamp: 这一轮交互完成时的时间戳（秒）。
    """

    question: str
    answer: str
    timestamp: float


class ConversationState(TypedDict, total=False):
    """
    对话与上下文管理状态（滚动摘要 + 最近最多 5 轮问答）。

    设计约束：
        - 只记录“问答轮”（QATurn），不记录澄清性问题、工具调用或中间步骤；
        - qa_turns 的长度始终不超过 max_qa_turns（默认 5）；
        - 所有更早的问答内容被压缩到一个滚动更新的 conversation_summary 中；
        - 每累计 summary_block_size 个问答轮，就用
          「旧 summary + 这批 qa_turns」生成新 summary，并清空 qa_turns。

    Attributes:
        session_id: 会话 ID（可选）。
        qa_turns:
            最近尚未进入摘要的问答轮列表，长度最多 5。
        current_user_input:
            当前这一轮用户输入（方便节点直接使用，无需从别处推断）。
        current_answer:
            当前这一轮生成的最终答案（在答案生成节点写入）。
        conversation_summary:
            截至目前为止所有历史问答轮的滚动摘要。
        qa_turn_count:
            累计问答轮数量（仅统计已完成的 QATurn）。
        summary_block_size:
            每多少问答轮就进行一次“summary + 清空 qa_turns”。
        max_qa_turns:
            qa_turns 的最大长度（建议与 summary_block_size 一致）。
    """

    session_id: Optional[str]
    qa_turns: List[QATurn]
    current_user_input: Optional[str]
    current_answer: Optional[str]
    conversation_summary: Optional[str]
    qa_turn_count: int
    summary_block_size: int
    max_qa_turns: int


# ---------------------- Agentic-RAG 全局状态 ----------------------


class AgenticRAGState(
    QueryUnderstandingState,
    RetrievalState,
    RetryState,
    AnswerGenerationState,
    SystemState,
    ConversationState,
):
    """
    Agentic-RAG 完整状态定义。

    整合了：
        - 问题理解与优化（QueryUnderstandingState）
        - 文档检索（RetrievalState）
        - Retry / 补偿机制（RetryState）
        - 答案生成与聚合（AnswerGenerationState）
        - 系统级控制与错误处理（SystemState）
        - 多轮对话与滚动摘要管理（ConversationState）

    所有字段都是可选的（total=False），允许状态在流程中逐步填充。
    """

    pass


# ---------------------- 辅助函数：状态初始化与校验 ----------------------


def create_initial_state(
    original_query: str,
    session_id: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    parent_child_threshold: int = DEFAULT_PARENT_CHILD_THRESHOLD,
    sub_question_count: int = DEFAULT_SUB_QUESTION_COUNT,
    summary_block_size: int = SUMMARY_BLOCK_SIZE,
    max_qa_turns: int = MAX_QA_TURNS,
) -> AgenticRAGState:
    """
    创建初始状态。

    注意：
        - 初始时还没有答案，所以不会立即生成第一条 QATurn；
        - qa_turns 由“答案生成节点”在生成最终回答后追加。

    Args:
        original_query: 用户本轮原始查询（必填）。
        session_id: 会话 ID（可选，不同请求串到同一对话时很有用）。
        max_retries: 最大 Retry 次数，默认 3。
        parent_child_threshold: 父文档替换阈值（命中子块数），默认 3。
        sub_question_count: 初始子问题数量，默认 5。
        summary_block_size: 每多少问答轮就进行一次“summary + 清空 qa_turns”。
        max_qa_turns: qa_turns 的最大长度（建议与 summary_block_size 一致）。
    """
    # 参数防御性校验
    if max_retries < 0:
        logger.warning("max_retries 小于 0，已重置为 0：%r", max_retries)
        max_retries = 0
    if parent_child_threshold < 1:
        logger.warning(
            "parent_child_threshold 小于 1，已重置为 1：%r", parent_child_threshold
        )
        parent_child_threshold = 1
    if sub_question_count < 1:
        logger.warning("sub_question_count 小于 1，已重置为 1：%r", sub_question_count)
        sub_question_count = 1
    if summary_block_size < 1:
        logger.warning("summary_block_size 小于 1，已重置为 1：%r", summary_block_size)
        summary_block_size = 1
    if max_qa_turns < 1:
        logger.warning("max_qa_turns 小于 1，已重置为 1：%r", max_qa_turns)
        max_qa_turns = 1

    now = time.time()

    return AgenticRAGState(
        # -------- Query Understanding（阶段一） --------
        original_query=original_query,
        rewritten_query=None,
        final_query=original_query,
        query_quality_score=None,
        is_query_clear=None,
        needs_human_intervention=None,
        human_supplemented_info=None,
        query_rewrite_attempts=0,
        # -------- Retrieval（阶段二） --------
        sub_questions=[],
        sub_question_count=sub_question_count,
        hybrid_retrieval_results={},
        sparse_retrieval_results={},
        dense_retrieval_results={},
        retrieval_weights={"sparse": 0.5, "dense": 0.5},
        parent_doc_replacements={},
        hit_sub_blocks_count={},
        parent_child_threshold=parent_child_threshold,
        evidence_sufficiency_score=None,
        is_evidence_sufficient=None,
        retrieval_mode="hybrid",
        # -------- Retry --------
        retry_triggered=False,
        retry_count=0,
        retry_strategy=None,
        hyde_enabled=False,
        step_back_enabled=False,
        step_back_query=None,
        expanded_sub_questions_enabled=False,
        additional_sub_questions=[],
        expanded_top_k=None,
        # -------- Answer Generation（阶段三） --------
        selected_prompt_template=None,
        prompt_template_type=None,
        sub_answers={},
        aggregated_answer=None,
        refined_answer=None,
        final_answer=None,
        generation_attempts=0,
        # -------- System --------
        current_phase="query_understanding",
        phase_history=[],
        error_message=None,
        max_retries=max_retries,
        execution_start_time=now,
        execution_end_time=None,
        # -------- Conversation（滚动摘要 + 最近问答轮） --------
        session_id=session_id,
        qa_turns=[],
        current_user_input=original_query,
        current_answer=None,
        conversation_summary=None,
        qa_turn_count=0,
        summary_block_size=summary_block_size,
        max_qa_turns=max_qa_turns,
    )


def validate_state(state: AgenticRAGState) -> bool:
    """
    验证状态的基本有效性。

    该函数主要用于在关键节点 / 调试阶段做快速一致性检查，
    不会抛出异常，只记录 warning 并返回 False。
    """
    # 1. 原始查询
    original_query = state.get("original_query")
    if not original_query or not isinstance(original_query, str):
        logger.warning(
            "状态验证失败：缺少 original_query 或类型错误，值=%r", original_query
        )
        return False

    # 2. final_query 类型（如果存在）
    final_query = state.get("final_query")
    if final_query is not None and not isinstance(final_query, str):
        logger.warning(
            "状态验证失败：final_query 类型错误，应为 str 或 None，值=%r",
            type(final_query),
        )
        return False

    # 3. 阶段
    current_phase = state.get("current_phase")
    if current_phase and current_phase not in (
        "query_understanding",
        "retrieval",
        "answer_generation",
    ):
        logger.warning("状态验证失败：无效的 current_phase: %r", current_phase)
        return False

    # 4. 检索权重
    retrieval_weights = state.get("retrieval_weights")
    if retrieval_weights:
        sparse_weight = float(retrieval_weights.get("sparse", 0.0) or 0.0)
        dense_weight = float(retrieval_weights.get("dense", 0.0) or 0.0)
        total_weight = sparse_weight + dense_weight
        if total_weight <= 0:
            logger.warning(
                "状态验证失败：检索权重总和 <= 0 (sparse=%r, dense=%r)",
                sparse_weight,
                dense_weight,
            )
            return False
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                "状态验证警告：检索权重总和不为 1.0 "
                "(sparse=%r, dense=%r, total=%r)，建议在节点中做归一化处理",
                sparse_weight,
                dense_weight,
                total_weight,
            )

    # 5. 检索模式
    retrieval_mode = state.get("retrieval_mode", "hybrid")
    if retrieval_mode not in ("hybrid", "hyde", "step_back"):
        logger.warning("状态验证失败：无效的 retrieval_mode: %r", retrieval_mode)
        return False

    # 6. Retry 次数
    retry_count = int(state.get("retry_count", 0) or 0)
    max_retries = int(
        state.get("max_retries", DEFAULT_MAX_RETRIES) or DEFAULT_MAX_RETRIES
    )
    if retry_count < 0 or max_retries < 0:
        logger.warning(
            "状态验证失败：retry_count 或 max_retries 为负数: retry=%r, max=%r",
            retry_count,
            max_retries,
        )
        return False
    if retry_count > max_retries:
        logger.warning(
            "状态验证失败：retry_count (%r) 超过 max_retries (%r)",
            retry_count,
            max_retries,
        )
        return False

    retry_strategy = state.get("retry_strategy")
    if retry_strategy and retry_strategy not in (
        "hyde",
        "step_back",
        "expand_sub_questions",
    ):
        logger.warning("状态验证失败：无效的 retry_strategy: %r", retry_strategy)
        return False

    # 7. 对话 & 滚动摘要
    qa_turns = state.get("qa_turns") or []
    if not isinstance(qa_turns, list):
        logger.warning(
            "状态验证失败：qa_turns 类型错误，应为 list，实际=%r",
            type(qa_turns),
        )
        return False

    max_qa_turns = int(state.get("max_qa_turns", MAX_QA_TURNS) or MAX_QA_TURNS)
    if max_qa_turns < 1:
        logger.warning("状态验证失败：max_qa_turns < 1: %r", max_qa_turns)
        return False
    if len(qa_turns) > max_qa_turns:
        logger.warning(
            "状态验证警告：qa_turns 数量 (%r) 超过 max_qa_turns (%r)，"
            "说明对话管理节点可能没有正确截断 / 触发摘要。",
            len(qa_turns),
            max_qa_turns,
        )

    qa_turn_count = int(state.get("qa_turn_count", 0) or 0)
    if qa_turn_count < 0:
        logger.warning("状态验证失败：qa_turn_count 为负数: %r", qa_turn_count)
        return False

    summary_block_size = int(
        state.get("summary_block_size", SUMMARY_BLOCK_SIZE) or SUMMARY_BLOCK_SIZE
    )
    if summary_block_size < 1:
        logger.warning("状态验证失败：summary_block_size < 1: %r", summary_block_size)
        return False

    # 8. final_answer 类型（如果存在）
    final_answer = state.get("final_answer")
    if final_answer is not None and not isinstance(final_answer, str):
        logger.warning(
            "状态验证失败：final_answer 类型错误，应为 str 或 None，值=%r",
            type(final_answer),
        )
        return False

    return True