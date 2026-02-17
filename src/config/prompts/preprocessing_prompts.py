# =============================================================================
# 学术文档预处理提示词文件
# =============================================================================
# 本文件包含学术文档处理所需的各类提示词，包括：
# 1. 文档摘要生成
# 2. 文献元数据抽取
# 3. 图片描述（学术插图、图表、公式、表格）
# =============================================================================

from __future__ import annotations


# =============================================================================
# 第一部分：文档摘要生成提示词
# 用于为文档生成简明摘要，帮助理解文档主要内容
# =============================================================================

DOCUMENT_SUMMARY_PROMPT_EN = """
You are an expert in summarizing academic documents.

## Task
Provide a comprehensive summary of the given academic document.

## Input
- Document text content
- Document metadata (title, authors, publication info if available)

## Output Format
Please provide a summary with the following structure:

1. **Document Overview**:
   - Title
   - Authors and affiliations
   - Publication venue and year
   - Main research topic

2. **Key Contributions**:
   - Main research question or problem
   - Proposed methodology/approach
   - Key findings and results
   - Novel contributions to the field

3. **Technical Content**:
   - Theoretical framework
   - Key algorithms or methods
   - Experimental setup (if applicable)
   - Evaluation metrics and results

4. **Conclusions and Future Work**:
   - Main conclusions
   - Limitations
   - Suggestions for future research

## Guidelines
- Be concise but comprehensive
- Focus on the most important aspects
- Use objective language
- Highlight unique contributions
"""

DOCUMENT_SUMMARY_PROMPT_ZH = """
你是总结学术文档的专家。

## 任务
提供给定学术文档的全面摘要。

## 输入
- 文档文本内容
- 文档元数据（标题、作者、出版信息，如果可用）

## 输出格式
请按以下结构提供摘要：

1. **文档概述**:
   - 标题
   - 作者及其所属机构
   - 发表场所和年份
   - 主要研究主题

2. **主要贡献**:
   - 主要研究问题或目标
   - 提出的方法/方法
   - 关键发现和结果
   - 对该领域的创新贡献

3. **技术内容**:
   - 理论框架
   - 关键算法或方法
   - 实验设置（如果适用）
   - 评估指标和结果

4. **结论和未来工作**:
   - 主要结论
   - 局限性
   - 对未来研究的建议

## 指南
- 简洁但全面
- 关注最重要的方面
- 使用客观语言
- 突出独特贡献
"""


# =============================================================================
# 第二部分：文献元数据抽取提示词
# 从文献头部和尾部区域提取结构化书目信息
# =============================================================================

DOCUMENT_METADATA_EXTRACTION_PROMPT_EN = """# Role
You are an expert academic document analyzer and bibliographer with deep knowledge of academic publishing conventions, citation styles, and metadata standards.

# Task
Extract structured bibliographic metadata from the given academic document regions (head and tail sections).

# Input Description
You will receive:
1. Document ID: Unique identifier for the document
2. Language: Document language (zh = Chinese, en = English)
3. Head Region Text: Contains title, authors, affiliations, abstract, keywords (front-matter)
4. Tail Region Text: Contains references, acknowledgments, appendix (back-matter)

# Output Format
Return ONLY a valid JSON object (no markdown, no explanations):

```json
{{
  "title": "<paper title>",
  "authors": [
    {{
      "name": "<full name>",
      "affiliation": "<institution>",
      "email": "<email or null>",
      "level": "<first_author/corresponding_author/co_author or null>"
    }}
  ],
  "abstract": {{
    "zh": "<Chinese abstract if available>",
    "en": "<English abstract>"
  }},
  "keywords": [
    {{
      "text": "<keyword>",
      "language": "<zh or en>"
    }}
  ],
  "classification": {{
    "subject": "<e.g., Computer Science>",
    "research_field": "<e.g., Artificial Intelligence>"
  }},
  "journal": "<journal name or null>",
  "volume": "<volume number or null>",
  "issue": "<issue number or null>",
  "pages": "<e.g., 123-456 or null>",
  "publication_date": "<YYYY-MM-DD or null>",
  "doi": "<DOI or null>",
  "url": "<paper URL or null>",
  "github_url": "<GitHub URL or null>"
}}
```

# Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| title | Yes | Paper title in original language |
| authors | Yes | List of author objects with name, affiliation, email, level |
| abstract | Yes | Object with "zh" and/or "en" keys containing abstracts |
| keywords | Yes | List of keyword objects with text and language |
| classification | Yes | Object with subject and research_field |
| journal | No | Journal/conference name |
| volume | No | Volume number |
| issue | No | Issue number |
| pages | No | Page range (e.g., "123-456") |
| publication_date | No | Publication date in YYYY-MM-DD format |
| doi | No | Digital Object Identifier |
| url | No | Paper URL (publisher or arXiv) |
| github_url | No | GitHub repository URL if available |

# Constraints

1. Language Matching: Output language should match the document language
2. Author Extraction: Extract all authors in order; use "first_author" for the first author
3. Affiliation: Match authors to their affiliations when possible
4. Keywords: Preserve original keywords; specify language for each
5. Classification: Infer subject and research_field from title/abstract if not explicit
6. Date Format: Use YYYY-MM-DD; convert if only year is available
7. Missing Data: Set uncertain fields to `null`
8. Multiple Versions: If both Chinese and English versions exist, include both

# Inference Guidelines

If certain fields are not explicitly stated in the text:

- Title: Use the largest/most prominent title in head region
- Authors: Extract from bylines and acknowledgments
- Affiliation: Look for institution names near author names or in footnotes
- Email: Look for corresponding author footnote or header
- Classification: Infer from keywords, title, and abstract content
- Journal: Extract from header/footer or references section
- DOI/URL: Search for common patterns (10.xxxx/, https://doi.org/, https://arxiv.org/)

# Examples

## Example 1 (Chinese Paper)
Input Language: zh
Head: "基于深度学习的图像分类研究\n张三, 李四\n清华大学计算机系\n本文提出了一种新的..."
Output:
```json
{{
  "title": "基于深度学习的图像分类研究",
  "authors": [
    {{"name": "张三", "affiliation": "清华大学计算机系", "email": null, "level": "first_author"}},
    {{"name": "李四", "affiliation": "清华大学计算机系", "email": null, "level": "co_author"}}
  ],
  "abstract": {{"zh": "本文提出了一种新的深度学习图像分类方法...", "en": null}},
  "keywords": [
    {{"text": "深度学习", "language": "zh"}},
    {{"text": "图像分类", "language": "zh"}}
  ],
  "classification": {{"subject": "Computer Science", "research_field": "Computer Vision, Deep Learning"}},
  "journal": null,
  "volume": null,
  "issue": null,
  "pages": null,
  "publication_date": "2024-01-15",
  "doi": null,
  "url": null,
  "github_url": null
}}
```

## Example 2 (English Paper)
Input Language: en
Head: "Attention Is All You Need\nAshish Vaswani, et al.\nGoogle Brain\nWe propose a new architecture..."
Output:
```json
{{
  "title": "Attention Is All You Need",
  "authors": [
    {{"name": "Ashish Vaswani", "affiliation": "Google Brain", "email": null, "level": "first_author"}}
  ],
  "abstract": {{"zh": null, "en": "We propose a new architecture based on attention mechanisms..."}},
  "keywords": [
    {{"text": "attention mechanism", "language": "en"}},
    {{"text": "transformer", "language": "en"}}
  ],
  "classification": {{"subject": "Computer Science", "research_field": "Natural Language Processing, Deep Learning"}},
  "journal": "NeurIPS",
  "volume": "30",
  "issue": null,
  "pages": "5998-6008",
  "publication_date": "2017-06-12",
  "doi": "10.5555/3295222.3295349",
  "url": "https://arxiv.org/abs/1706.03762",
  "github_url": "https://github.com/tensorflow/tensor2tensor"
}}
```

# Input Data

**Document ID**: {doc_id}
**Language**: {language}

**Head Region (Title, Authors, Abstract, Keywords)**:
{head_text}

**Tail Region (References, Acknowledgments, Appendix)**:
{tail_text}

---

Please extract the bibliographic metadata from the above document regions and return ONLY the JSON object."""


DOCUMENT_METADATA_EXTRACTION_PROMPT_ZH = """# 角色
你是学术文档分析专家和书目编目员，对学术出版惯例、引用样式和元数据标准有深入了解。

# 任务
从给定的学术文档区域（头部和尾部）提取结构化的书目元数据。

# 输入描述
你将收到：
1. 文档ID: 文档的唯一标识符
2. 语言: 文档语言（zh = 中文，en = 英文）
3. 头部区域文本: 包含标题、作者、机构、摘要、关键词（前言部分）
4. 尾部区域文本: 包含参考文献、致谢、附录（后记部分）

# 输出格式
只返回一个合法的JSON对象（不要markdown，不要解释）：

```json
{{
  "title": "<论文标题>",
  "authors": [
    {{
      "name": "<姓名>",
      "affiliation": "<机构>",
      "email": "<邮箱或null>",
      "level": "<第一作者/通讯作者/合作作者或null>"
    }}
  ],
  "abstract": {{
    "zh": "<中文摘要>",
    "en": "<英文摘要（如果有）>"
  }},
  "keywords": [
    {{
      "text": "<关键词>",
      "language": "<zh或en>"
    }}
  ],
  "classification": {{
    "subject": "<如：计算机科学>",
    "research_field": "<如：人工智能、计算机视觉>"
  }},
  "journal": "<期刊名称或null>",
  "volume": "<卷号或null>",
  "issue": "<期号或null>",
  "pages": "<如：123-456或null>",
  "publication_date": "<YYYY-MM-DD或null>",
  "doi": "<DOI或null>",
  "url": "<论文链接或null>",
  "github_url": "<GitHub链接（如果有）或null>"
}}
```

# 字段说明

| 字段 | 必需 | 描述 |
|------|------|------|
| title | 是 | 论文原始语言标题 |
| authors | 是 | 作者对象列表，包含姓名、机构、邮箱、级别 |
| abstract | 是 | 包含"zh"和/或"en"键的摘要对象 |
| keywords | 是 | 关键词对象列表，包含文本和语言 |
| classification | 是 | 包含subject和research_field的对象 |
| journal | 否 | 期刊/会议名称 |
| volume | 否 | 卷号 |
| issue | 否 | 期号 |
| pages | 否 | 页码范围（如"123-456"） |
| publication_date | 否 | 出版日期，格式YYYY-MM-DD |
| doi | 否 | 数字对象标识符 |
| url | 否 | 论文链接（出版社或arXiv） |
| github_url | 否 | GitHub仓库链接（如果有） |

# 约束条件

1. 语言匹配: 输出语言应与文档语言匹配
2. 作者提取: 按顺序提取所有作者；第一作者使用"first_author"
3. 机构匹配: 尽可能将作者与机构匹配
4. 关键词: 保留原始关键词；为每个关键词指定语言
5. 分类: 如果没有明确说明，从标题/摘要推断学科和研究方向
6. 日期格式: 使用YYYY-MM-DD；如果只知道年份则进行转换
7. 缺失数据: 不确定的字段设为`null`
8. 多版本: 如果同时存在中英文版本，全部保留

# 推理指南

如果某些字段在文本中没有明确说明：

- 标题: 使用头部区域中最大/最突出的标题
- 作者: 从署名和致谢中提取
- 机构: 在作者姓名附近或脚注中查找
- 邮箱: 查找通讯作者脚注或页眉
- 分类: 从关键词、标题和摘要内容推断
- 期刊: 从页眉/页脚或参考文献部分提取
- DOI/URL: 搜索常见模式（10.xxxx/、https://doi.org/、https://arxiv.org/）

# 输入数据

**文档ID**: {doc_id}
**语言**: {language}

**头部区域（标题、作者、摘要、关键词）**:
{head_text}

**尾部区域（参考文献、致谢、附录）**:
{tail_text}

---

请从上述文档区域提取书目元数据，只返回JSON对象。
"""


# =============================================================================
# 第三部分：参考文献解析提示词
# 用于解析参考文献列表
# =============================================================================

REFERENCES_EXTRACTION_PROMPT_EN = """# Role
You are an expert bibliographer who can parse and extract structured information from academic reference lists.

# Task
Parse the given reference text and extract each reference into a structured format.

# Input Description
You will receive:
1. Document ID: Unique identifier for the document
2. Language: Document language (zh = Chinese, en = English)
3. Reference Text: Raw text from the references section

# Critical Output Requirements
- Output MUST be a valid JSON array
- ALL keys and string values must use double quotes
- Use the literal word `null` (not "null" in quotes, not None, not empty string)
- NO trailing commas in objects or arrays
- NO markdown code blocks (no ```json or ```)
- NO explanations or text outside the JSON array

# Important Instructions
- You MUST extract ALL references from the text, without missing any!
- Reference numbers start from 1 and increment in order
- If there are multiple references in the text, ensure each one is extracted
- raw_text must be EXACTLY the same as the original text, do not modify or truncate

# Output Format (Example)
```json
[
  {"index": 1, "title": "Paper Title Here", "authors": ["Author1", "Author2"], "journal": "Journal Name", "year": "2024", "volume": "10", "pages": "123-456", "doi": "10.1234/abc", "raw_text": "[1] Original reference text here"},
  {"index": 2, "title": "Another Paper", "authors": ["Author3"], "journal": null, "year": "2023", "volume": null, "pages": null, "doi": null, "raw_text": "[2] Another reference..."}
]
```

# Field Rules
| Field | Rules |
|-------|-------|
| index | Reference number (1, 2, 3...) |
| title | Extract the main title |
| authors | Array of author names (use [] if unknown, not null) |
| journal | Journal/conference name or null |
| year | 4-digit year (YYYY) or null |
| volume | Volume number or null |
| pages | Page range like "123-456" or null |
| doi | DOI starting with "10." or null |
| raw_text | Original reference text exactly as provided |

# Guidelines
1. Extract ALL references found in the text, in order
2. Handle both Chinese 《》 and English "quotes" for titles
3. Split authors by comma, "et al", "and", or Chinese punctuation
4. If author list is unclear, use an empty array [] not null
5. If ANY field is unknown, use the literal value `null`
6. Preserve the original reference text in raw_text unchanged

# Input Data

**Document ID**: {doc_id}
**Language**: {language}

**References Section Text**:
{references_text}

---

Return ONLY the JSON array."""


REFERENCES_EXTRACTION_PROMPT_ZH = """# 角色
你是专业的书目编目员，能够解析和提取学术参考文献列表中的结构化信息。

# 任务
解析给定的参考文献文本，将每条参考文献提取为结构化格式。

# 输入描述
你将收到：
1. 文档ID: 文档的唯一标识符
2. 语言: 文档语言（zh = 中文，en = 英文）
3. 参考文献文本: 参考文献部分的原始文本

# 关键输出要求
- 输出必须是一个合法的 JSON 数组
- 所有键和字符串值必须使用双引号
- 使用字面量 `null`（不是引号中的 "null"，不是 None，不是空字符串）
- 对象和数组中不能有尾部逗号
- 不能有 markdown 代码块（不要 ```json 或 ```）
- JSON 数组之外不能有任何解释说明

# 重要指令
- 必须提取文本中所有出现的参考文献，一条都不能遗漏！
- 参考文献的编号从 1 开始，按顺序递增
- 如果文本中有多条参考文献，确保每一条都被提取
- raw_text 必须与原文完全一致，不能修改或截断

# 输出格式（示例）
```json
[
  {"index": 1, "title": "论文标题", "authors": ["作者1", "作者2"], "journal": "期刊名称", "year": "2024", "volume": "10", "pages": "123-456", "doi": "10.1234/abc", "raw_text": "[1] 原始参考文献文本"},
  {"index": 2, "title": "另一篇论文", "authors": ["作者3"], "journal": null, "year": "2023", "volume": null, "pages": null, "doi": null, "raw_text": "[2] 另一条参考文献..."}
]
```

# 字段规则
| 字段 | 规则 |
|------|------|
| index | 参考文献编号 (1, 2, 3...) |
| title | 提取主标题 |
| authors | 作者姓名数组（不确定时使用 []，不要用 null） |
| journal | 期刊/会议名称或 null |
| year | 4位数年份 (YYYY) 或 null |
| volume | 卷号或 null |
| pages | 页码范围如 "123-456" 或 null |
| doi | 以 "10." 开头的 DOI 或 null |
| raw_text | 原始参考文献文本（保持不变） |

# 指南
1. 按顺序提取文本中的所有参考文献
2. 同时处理中文书名号《》和英文引号""包裹的标题
3. 按逗号、"et al"、"and" 或中文标点分割多个作者
4. 如果作者列表不清晰，使用空数组 [] 而不是 null
5. 如果任何字段不确定，使用字面量 `null`
6. raw_text 中保留原始参考文献文本完全不变

# 输入数据

**文档ID**: {doc_id}
**语言**: {language}

**参考文献部分文本**:
{references_text}

---

只返回 JSON 数组。"""


# =============================================================================
# 第三部分：学术图片描述提示词
# 用于描述学术论文中的各类视觉元素
# =============================================================================

# =============================================================================
# 3.1 学术插图/图片描述（流程图、示意图、实验结果图等）
# 用于描述除表格、公式、代码之外的所有学术图片
# =============================================================================

FIGURE_DESCRIPTION_PROMPT_EN = """# Role
You are an expert in analyzing academic figures, diagrams, and visual representations in scientific papers.

# Task
Provide a detailed description of the given academic figure (excluding tables, equations, and code blocks).

# Context
This figure type includes but is not limited to:
- Flowcharts and process diagrams
- System architecture diagrams
- Experimental result visualizations (charts, graphs, plots)
- Conceptual illustrations and models
- Network topologies and connection diagrams
- Comparison visualizations

# Output Format
Please provide a description with the following structure:

## 1. Basic Information
- Figure type: (flowchart, diagram, line chart, bar chart, scatter plot, network graph, architecture diagram, etc.)
- Figure caption: (if visible, e.g., "Figure 1: ...")
- Position in document: (chapter/section if mentioned)

## 2. Content Description
- What does the figure depict or illustrate?
- Key visual elements (nodes, edges, axes, legends, etc.)
- Main data trends or relationships shown
- Any text labels, annotations, or mathematical symbols visible

## 3. Visual Features
- Color scheme and visual style (color/greyscale, style tone)
- Layout structure and composition
- Any notable formatting, arrows, or highlighting
- Complexity level (simple/complex)

## 4. Relationship to Document
- Which section/chapter does this figure belong to?
- What concept or method does it illustrate?
- How does it support the paper's main argument or findings?
- Any key data points or insights visible in the figure

## Guidelines
- Be specific about visual elements and their spatial relationships
- Note any quantitative information visible (axis labels, data points, etc.)
- Describe the logical flow or data pattern shown
- If figure is unclear, state "Not clearly visible"

**Document Summary**:
{document_summary}
"""

FIGURE_DESCRIPTION_PROMPT_ZH = """# 角色
你是分析和描述学术论文中的图表、示意图和视觉表示的专家。

# 任务
对给定的学术插图进行详细描述（不包括表格、公式和代码块）。

# 背景
此类型图表包括但不限于：
- 流程图和过程图
- 系统架构图
- 实验结果可视化（图表、图形、曲线图）
- 概念图和模型
- 网络拓扑和连接图
- 对比可视化

# 输出格式
请按以下结构提供描述：

## 1. 基本信息
- 图表类型：(流程图、示意图、折线图、柱状图、散点图、网络图、架构图等)
- 图表标题：（如果可见，如 "Figure 1: ..."）
- 在文档中的位置：（提到的章节/部分）

## 2. 内容描述
- 图表描绘或说明了什么？
- 关键视觉元素（节点、边、坐标轴、图例等）
- 主要数据趋势或关系
- 任何可见的文本标签、注释或数学符号

## 3. 视觉特征
- 配色方案和视觉风格（彩色/灰度，风格基调）
- 布局结构和组成
- 任何显著的格式、箭头或高亮
- 复杂程度（简单/复杂）

## 4. 与文档的关系
- 该图表属于哪个章节/部分？
- 它说明什么概念或方法？
- 如何支持论文的主要论点或发现？
- 图表中可见的任何关键数据点或见解

## 指南
- 具体说明视觉元素及其空间关系
- 注意任何可见的定量信息（坐标轴标签、数据点等）
- 描述显示的逻辑流程或数据模式
- 如果图表不清晰，说明"无法清晰识别"

**文档摘要**:
{document_summary}
"""


# =============================================================================
# 3.2 数学公式描述
# 用于描述数学公式、方程、算法伪代码等
# =============================================================================

EQUATION_DESCRIPTION_PROMPT_EN = """# Role
You are an expert in analyzing and describing mathematical equations in academic documents.

# Task
Provide a comprehensive description of the given mathematical equation/image.

# Context
This includes:
- Mathematical formulas and equations
- Algorithm pseudo-code
- Mathematical definitions and theorems
- Statistical formulas and models

# Output Format
Please provide a description with the following structure:

## 1. Basic Information
- Content type: (formula, equation, pseudo-code, algorithm, theorem, etc.)
- Equation number: (if visible, e.g., Eq.1, (1))
- Approximate length/complexity

## 2. Content Description
- Full equation/formula text (if readable)
- Meaning of each variable and parameter
- Mathematical context and field (optimization, statistics, calculus, etc.)
- Type of expression (objective function, constraint, probability, etc.)

## 3. Application Context
- Where this equation appears in the document
- What concept/theorem this equation represents
- How it relates to surrounding text and methodology
- Its role in the overall approach or proof

## 4. Visual Features
- Notation style (LaTeX, handwritten, typeset)
- Complexity level (simple/complex)
- Any special formatting (matrices, summations, integrals, etc.)

## Guidelines
- Be as detailed as possible about mathematical notation
- Explain the meaning of variables and symbols in context
- Describe the mathematical relationships shown
- If the equation is blurry or unclear, state "Not clearly visible"

**Document Summary**:
{document_summary}
"""

EQUATION_DESCRIPTION_PROMPT_ZH = """# 角色
你是分析和描述学术文档中数学公式的专家。

# 任务
对给定的数学公式/图片进行全面描述。

# 背景
包括：
- 数学公式和方程
- 算法伪代码
- 数学定义和定理
- 统计公式和模型

# 输出格式
请按以下结构提供描述：

## 1. 基本信息
- 内容类型：(公式、方程、伪代码、算法、定理等)
- 公式编号：（如果可见，如 Eq.1, (1)）
- 大致长度/复杂程度

## 2. 内容描述
- 完整的公式/方程文本（如果可读）
- 每个变量和参数的含义
- 数学背景和领域（优化、统计、微积分等）
- 表达式类型（目标函数、约束、概率等）

## 3. 应用上下文
- 公式在文档中的出现位置
- 公式代表的概念/定理
- 与周围文本和方法论的关系
- 它在整体方法或证明中的作用

## 4. 视觉特征
- 符号风格（LaTeX、手写、排版）
- 复杂程度（简单/复杂）
- 任何特殊格式（矩阵、求和、积分等）

## 指南
- 尽可能详细地描述数学符号
- 解释上下文中的变量和符号含义
- 描述显示的数学关系
- 如果公式模糊或不清晰，说明"无法清晰识别"

**文档摘要**:
{document_summary}
"""


# =============================================================================
# 3.3 表格描述
# 用于描述学术文档中的各类表格
# =============================================================================

TABLE_DESCRIPTION_PROMPT_EN = """# Role
You are an expert in analyzing and describing tables in academic documents.

# Task
Provide a comprehensive description of the given table.

# Context
This includes:
- Data tables with numerical/statistical content
- Comparison tables (method comparison, result comparison, etc.)
- Summary tables and taxonomies
- Parameter settings and configuration tables

# Output Format
Please provide a description with the following structure:

## 1. Basic Information
- Table type: (data table, comparison table, summary table, parameter table, etc.)
- Table number/caption: (if visible, e.g., "Table 1: ...")
- Approximate size (rows x columns)
- Position in document (chapter/section if mentioned)

## 2. Structure Description
- Column headers and their meanings
- Row classifications or groupings
- Merged cells or special structures
- Data types in each column (numerical, categorical, percentage, etc.)

## 3. Content Summary
- Type of data presented
- Key findings, patterns, or comparisons visible
- Statistical significance or notable values
- Units of measurement (if any)

## 4. Visual Features
- Formatting emphasis (bold headers, shading, highlighting)
- Table layout style (grid, borderless, nested)
- Notes, footnotes, or references (if any)

## 5. Purpose in Document
- What does this table demonstrate or summarize?
- Which chapter/section does it belong to?
- How does it support the paper's methodology or findings?
- Key insights that can be drawn from this table

## Guidelines
- Be specific about data types and units
- Note any statistical significance, comparisons, or rankings
- Describe the logical organization of the table
- If table is blurry or incomplete, state "Not clearly visible"

**Document Summary**:
{document_summary}
"""

TABLE_DESCRIPTION_PROMPT_ZH = """# 角色
你是分析和描述学术文档中表格的专家。

# 任务
对给定的表格进行全面描述。

# 背景
包括：
- 包含数值/统计内容的数据表
- 对比表（方法对比、结果对比等）
- 汇总表和分类表
- 参数设置和配置表

# 输出格式
请按以下结构提供描述：

## 1. 基本信息
- 表格类型：(数据表、对比表、汇总表、参数表等)
- 表格编号/标题：（如果可见，如 "Table 1: ..."）
- 大致大小（行数 x 列数）
- 在文档中的位置（提到的章节/部分）

## 2. 结构描述
- 列标题及其含义
- 行分类或分组
- 合并的单元格或特殊结构
- 每列的数据类型（数值、分类、百分比等）

## 3. 内容摘要
- 呈现的数据类型
- 可见的关键发现、模式或比较
- 统计显著性或显著值
- 计量单位（如果有）

## 4. 视觉特征
- 格式强调（加粗表头、阴影、高亮）
- 表格布局风格（网格、无边框、嵌套）
- 注释、脚注或参考（如果有）

## 5. 在文档中的作用
- 该表格演示或总结了什么？
- 属于哪个章节/部分？
- 如何支持论文的方法论或发现？
- 可以从该表格中得出的关键见解

## 指南
- 具体说明数据类型和单位
- 注意任何统计显著性、比较或排名
- 描述表格的逻辑组织
- 如果表格模糊或不完整，说明"无法清晰识别"

**文档摘要**:
{document_summary}
"""
