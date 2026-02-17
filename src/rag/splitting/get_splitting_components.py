"""
文本切分组件模块。

提供文本分割器的初始化方法，支持中英文混合文档的智能切分。

功能：
1. 递归字符文本分割器（RecursiveCharacterTextSplitter）
2. 支持中英文混合分隔符，保证不切分完整句子
3. 从配置文件读取切分参数

参考：
- LangChain RecursiveCharacterTextSplitter
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import (
    TEXT_CHUNK_SON_SIZE,
    TEXT_CHUNK_SON_OVERLAP,
)


# ===================================
# 通用文本分隔符列表
# ===================================
# 策略：按段落、子句、句子的优先级切分
# - 不切断完整句子（按 . 。! ？ 等句子结束符切分）
# - 尽量合并短句成接近 chunk_size 的块
#
SEPARATORS = [
    "\n\n",  # 段落分隔符
    "\n",  # 行分隔符
    # 句子结束符（确保不切断完整句子）
    ". ",  # 英文句号 + 空格
    "! ",  # 英文感叹号 + 空格
    "? ",  # 英文问号 + 空格
    "。",  # 中文句号
    "！",  # 中文感叹号
    "？",  # 中文问号
    # 子句分隔符（用于合并短句）
    "; ",  # 英文分号 + 空格
    ": ",  # 英文冒号 + 空格
    ", ",  # 英文逗号 + 空格
    "；",  # 中文分号
    "：",  # 中文冒号
    "，",  # 中文逗号
    "、",  # 中文顿号
    # 引号和括号边界
    '"',  # 双引号
    "'",  # 单引号
    "（",  # 中文左括号
    "）",  # 中文右括号
    "(",  # 英文左括号
    ")",  # 英文右括号
    # 兜底方案
    " ",  # 空格
    "",  # 按字符切分（最后手段）
]


# ===================================
# 获取 文本分割器
# ===================================
def get_text_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> RecursiveCharacterTextSplitter:
    """
    获取文本分割器。

    特点：
    - 按段落、子句、句子的优先级切分
    - 不切断完整句子（以句子结束符为切分点）
    - 多个短句会自动合并，更接近 chunk_size

    Args:
        chunk_size: 单个文本块的最大字符数。若为 None，使用配置文件中的 CHUNK_SON_SIZE。
        chunk_overlap: 相邻文本块之间的重叠字符数。若为 None，使用配置文件中的 CHUNK_SON_OVERLAP。

    Returns:
        RecursiveCharacterTextSplitter: 配置好的文本分割器实例。
    """
    if chunk_size is None:
        chunk_size = TEXT_CHUNK_SON_SIZE if TEXT_CHUNK_SON_SIZE is not None else 512
    if chunk_overlap is None:
        chunk_overlap = TEXT_CHUNK_SON_OVERLAP if TEXT_CHUNK_SON_OVERLAP is not None else 128

    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    chunk_size = TEXT_CHUNK_SON_SIZE if TEXT_CHUNK_SON_SIZE is not None else 256
    chunk_overlap = TEXT_CHUNK_SON_OVERLAP if TEXT_CHUNK_SON_OVERLAP is not None else 128

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 测试文本
    test_text = """
    Our approach not only achieves superior verification accuracy compared to existing baseline methods, such as deductive verifier (Ling et al., 2023), backward verifier (Weng et al., 2023), but also enhances reasoning accuracy by integrating detailed feedback derived from the verification process. Furthermore, VerifiAgent can be effectively applied to inference scaling, requiring significantly fewer computational resources compared to standard Process Reward Models (PRMs), thereby providing a practical approach to improve LLM performance during inference. Through extensive experiments across three types of reasoning tasks, we summarise two key empirical findings: 1) An LLM reasoner can improve via inference scaling methods like Majority Vote, PRMs, or VerifiAgent, but VerifiAgent achieves higher accuracy at lower cost. 2) VerifiAgent's capabilities scale alongside improvements in its backbone LLM, enabling consistent performance gains on the same reasoner.
    The image is a line chart illustrating the accuracy performance of different reasoning methods on MATH 350 test cases using the GPT-4o reasoner. The chart is designed to compare the effectiveness of various approaches in terms of accuracy percentage as a function of the number of samples.\n**Visible Elements:**\n1. **Axes and Labels:**\n- The x-axis is labeled \"Number of Samples,\" ranging from 0 to 10.\n- The y-axis is labeled \"Accuracy (%)\" and spans from 68% to 74%.\n2. **Data Series:**\n- Five distinct data series are plotted, each represented by a different color and marker style:\n- **Blue Line with Circles:** Represents \"Majority Vote @N.\"\n- **Orange Line with Circles:** Represents \"Best-of-N (Qwen2.5-Math-PRM-7B).\"\n- **Green Line with Circles:** Represents \"Best-of-N (Qwen2.5-Math-7B-PRM800K).\"\n- **Red Star Marker:** Represents \"GPT-4o+VerifiAgent.\"\n- **Purple Triangle Marker:** Represents \"GPT-4o.\"\n3. **Legend:**\n- Located in the lower right corner, the legend identifies each data series by color and marker type, providing a clear reference for interpreting the chart.\n4. **Grid Lines:**\n- The chart includes horizontal and vertical grid lines to enhance readability and facilitate the comparison of data points across different series.\n**Structure and Spatial Layout:**\n- The chart is organized with the x-axis representing the number of samples and the y-axis indicating accuracy. The data series are plotted across this grid, showing trends and comparisons.\n- The lines generally show an upward trend, indicating that accuracy tends to increase with the number of samples.\n- The red star and purple triangle markers are distinct from the lines, highlighting specific data points for \"GPT-4o+VerifiAgent\" and \"GPT-4o,\" respectively.\n**Patterns and Notable Features:**\n- The blue line (\"Majority Vote @N\") shows a consistent upward trend, reaching the highest accuracy at 10 samples.\n- The orange and green lines (\"Best-of-N\" methods) also show improvement with more samples, but at a slightly lower rate compared to the blue line.\n- The red star marker for \"GPT-4o+VerifiAgent\" is positioned at the top of the chart, indicating a high accuracy level, suggesting its superior performance.\n- The purple triangle for \"GPT-4o\" is positioned lower, indicating a baseline performance without additional verification.\n**Overall Purpose and Context:**\nThe chart's primary purpose is to visually compare the accuracy of different verification and reasoning methods applied to the MATH 350 test cases using the GPT-4o reasoner. It highlights the effectiveness of the VerifiAgent framework, as indicated by the red star, which achieves high accuracy with fewer samples. This aligns with the document's theme of enhancing the reliability and efficiency of large language models through advanced verification techniques. The chart effectively demonstrates the comparative advantage of VerifiAgent over other methods, supporting the document's claims of improved verification accuracy and reasoning performance."
    """

    chunks = splitter.create_documents([test_text])
    print(f"配置：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"分割结果（共 {len(chunks)} 个块）：")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] 长度={len(chunk.page_content)} 字符: {chunk.page_content}")
