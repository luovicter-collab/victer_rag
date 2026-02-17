# Agent Graph RAG 项目说明

基于 MinerU 解析与 LangChain 的 RAG（检索增强生成）项目：从 PDF 文档到结构化 JSON、再到向量检索与 Agent 对话的完整流水线。本文档对仓库内**主要文档与代码文件**逐一说明。

---

## 一、项目简介与目录结构

### 1.1 项目目标

- **数据初始化**：PDF → MinerU 解析 → 多源 JSON 融合 → 元素级结构化 JSON（含 head/body/tail 区域划分），供 RAG 嵌入。
- **RAG 与 Agent**：文本切分、向量库、LLM/嵌入模型封装；LangGraph Agentic-RAG 状态定义（问题理解、检索、答案生成、滚动摘要与记忆）。

### 1.2 顶层目录概览

```
Agent_gragh_rag/
├── README.md                 # 本说明文档
├── requirements.txt         # Python 依赖
├── setup.py                  # 包安装配置
├── test/                     # 测试或临时脚本
├── src/                      # 项目主代码
├── files/                    # 文件与数据存储
├── model/                    # MinerU 与嵌入模型等
└── .vscode/                  # VS Code 调试配置
```

---

## 二、根目录文件说明

| 文件 | 说明 |
|------|------|
| **README.md** | 项目与文件说明文档（本文件）。 |
| **requirements.txt** | Python 依赖：LangChain 系列、sentence-transformers、chromadb、openai、oss2、pymupdf、aiohttp、python-dotenv 等。 |
| **setup.py** | setuptools 配置：包名为 `agent-graph-rag`，`find_packages()` 发现所有子包，依赖中含 `oss2`，Python ≥3.8。 |

---

## 三、`src/` 包与文件说明

### 3.1 `src/config/` — 配置与提示词

| 文件 | 说明 |
|------|------|
| **settings.py** | 统一配置入口：自动把 `src` 加入 `sys.path`、加载 `.env`；定义 `PROJECT_ROOT`、`CONFIG_DIR`、`SRC_DIR`；设备（CPU/CUDA）；MinerU API/URL、OSS 密钥与桶；LLM 模型（main/fast/vision/high_precision）及超参；嵌入模型路径与 Chroma 持久化目录；关系型 DB 路径（SQLite）；文本/图片 chunk 与 overlap；各阶段并发与超时；**处理阶段常量**（`STAGE_LAYOUT_JSON_PARSED`、`STAGE_FRAGMENT_MERGED`、`STAGE_REGION_DIVIDED`、`STAGE_METADATA_EXTRACTED`、`STAGE_IMAGE_DESCRIPTION`、`STAGE_RAG_EMBEDDING`）及 `PROCESS_STAGES` 列表。 |
| **prompts/preprocessing_prompts.py** | 学术文档预处理用提示词：文档摘要（中/英）、文献元数据抽取、参考文献解析、图片/公式/表格描述（中英双语），供 data_initialization 中的 LLM 调用（如图片描述）。 |

---

### 3.2 `src/data_initialization/` — 数据初始化管线（基于 layout.json）

从 MinerU 的 JSON 产出构建 RAG 用结构化 JSON，各步会更新 `metadata.parse_stage`。

| 文件 | 说明 |
|------|------|
| **__init__.py** | 包说明；导出 `ElementExtractor`、`DocumentMetadata`、`DocumentElement` 等（来自 layout_json_parser）；不导入 pipeline 以避免循环依赖。 |
| **pipeline.py** | **数据初始化主入口**：`async_run_data_initialization_pipeline()` 依次执行：① PDF→MD（保留 work_dir 中的 layout.json）② 元素提取 ③ JSON 片段合并 ④ 区域划分 head/body/tail ⑤ 可选图片描述；统计各步耗时并打印。 |
| **converters/__init__.py** | 子包说明（MinerU 转换器）。 |
| **converters/pdf_to_md.py** | PDF→Markdown 转换：调用 MinerU（API 或本地）、上传/下载 OSS、处理 zip；**数据初始化专用**接口 `async_batch_convert_pdfs_with_layout()` 保证输出 work_dir 中含 layout.json，供后续元素提取使用。 |
| **processors/__init__.py** | 从 settings 导入各 `STAGE_*` 与 `PROCESS_STAGES`；提供 `update_parse_stage()`、`get_parse_stage()`、`is_stage_completed()`、`should_skip_stage()`，用于按阶段更新/查询 JSON 的 `parse_stage`。 |
| **processors/layout_json_parser.py** | **元素提取器**：从 MinerU 多 JSON（content_list_v2、content_list、model、layout）融合数据，输出 RAG 嵌入格式：`metadata`（doc_id、doc_title、parse_stage、language、source_file、pdf_path、total_pages、total_elements）+ `elements`（id、type、content、source、metadata），类型含 paragraph/title/table/image/code/equation。 |
| **processors/json_fragment_merger.py** | **片段合并**：对 paragraph 做“句末标点未结束则与下一块合并”、英文断词“-”合并；合并后重编元素 id，更新 total_elements 及后续区域序号。 |
| **processors/region_extractor.py** | **区域划分与标题提取**：根据 type=title 及 content.text 识别摘要/目录/参考文献/附录等；划定 body（从“1 Introduction/绪论”到“参考文献/References”前）；写出 head/body/tail 的 start_seq、end_seq 到 `metadata.region_division`。 |
| **processors/imagedescription_from_json.py** | **图片描述（可选）**：读取 JSON 中带 `source.image_path` 的元素，优先用 metadata.abstract，否则用 LLM 生成摘要；按中/英文调用 Vision LLM 生成描述，写入 `content.description`。 |
| **utils/__init__.py** | 工具函数子包说明。 |

---

### 3.3 `src/rag/` — RAG 组件

| 文件 | 说明 |
|------|------|
| **splitting/get_splitting_components.py** | **文本切分**：基于 LangChain `RecursiveCharacterTextSplitter`，提供中英文分隔符列表（段落、句末标点、逗号等），从 settings 读取 chunk_size、overlap，用于 RAG 前对文档分块。 |

---

### 3.4 `src/graphs/` — Agent 图状态

| 文件 | 说明 |
|------|------|
| **state.py** | **LangGraph Agentic-RAG 状态定义**：与“多轮对话上下文与记忆管理”方案一致；包含阶段一（问题理解与优化）、检索、阶段三（答案生成与聚合）、Retry 策略、滚动摘要与 qa_turns、Working Memory 与 Semantic Memory 视图等 TypedDict；不直接拼接完整历史，只保留工作记忆与召回视图。 |

---

### 3.5 `src/models/` — 模型与存储封装

| 文件 | 说明 |
|------|------|
| **get_models.py** | 统一获取：**LLM**（fast/main/vision/high_precision，基于 `settings` 中对应 config）、**嵌入模型**（HuggingFaceEmbeddings）、**向量库**（Chroma，持久化目录与 collection 来自 settings）、**关系型 DB**（SQLite 连接）。 |

---

## 四、`files/` 目录说明

| 路径 | 说明 |
|------|------|
| **files/file_store/pdf_store** | 原始 PDF 存放目录，作为数据初始化管线的输入。 |
| **files/file_store/md_store/minerU_md** | MinerU 产出的 Markdown。 |
| **files/file_store/md_store/minerU_work** | MinerU 工作目录，每文档子目录内含 layout.json、content_list_v2.json 等，供 data_initialization 元素提取。 |
| **files/file_store/md_store/minerU_zip** | MinerU 产出的 zip。 |
| **files/file_store/json_store** | 数据初始化管线产出的元素级 JSON（含 region_division），供 RAG 嵌入与检索。 |
| **files/file_store/zip_store/minerU_zip** | 数据初始化管线中 zip 输出目录（若使用）。 |
| **files/vector_store/rag** | Chroma 向量库持久化目录（见 settings）。 |
| **files/relation_store/rag.db** | SQLite 关系库路径（见 settings）。 |

---

## 五、`model/` 目录说明

| 路径 | 说明 |
|------|------|
| **model/MinerU/** | MinerU 相关代码与文档（PDF 解析、OCR、布局等），含 projects（如 mineru_tianshu、multi_gpu_v2、mcp）及 mineru 核心与文档；本仓库通过 API 或本地调用使用，不在此文档逐文件展开。 |
| **model/embeddings/** | 嵌入模型存放目录（如 sentence-transformers/all-mpnet-base-v2、BAAI/bge-large-zh-v1 等），路径在 settings 中配置。 |

---

## 六、其他文件说明

| 文件 | 说明 |
|------|------|
| **test/11.py** | 测试或临时脚本（当前仅含 pip 安装 langchain_neo4j 的注释）。 |
| **.vscode/launch.json** | VS Code 调试配置：运行 `src/data_initialization/pipeline.py`，cwd 为工作区根目录，`PYTHONPATH=${workspaceFolder}`。 |

---

## 七、数据初始化流水线概览

| 步骤 | 说明 |
|------|------|
| 1 | PDF → MD（MinerU，保留 work_dir 中的 layout.json） |
| 2 | 元素提取（多 JSON 融合 → RAG 元素格式） |
| 3 | JSON 片段合并（句末标点 / 英文断词） |
| 4 | 区域划分 head/body/tail（写入 metadata.region_division） |
| 5 | 可选：图片描述写入 JSON |
| 输出 | json_store（elements + region_division），供 RAG 嵌入与 Agent 检索。 |

---

## 八、运行与依赖

- 安装依赖：`pip install -r requirements.txt`；从项目根目录运行脚本时需保证 `PYTHONPATH` 包含项目根（或通过 `python -m` 运行）。
- 运行数据初始化管线：执行 `src/data_initialization/pipeline.py`（可直接用 .vscode 中的 “Python: pipeline.py” 配置）。
- 环境变量与密钥：见 `src/config/settings.py`（OSS、MinerU API、OpenAI/LLM、向量库路径等），建议使用 `.env` 配置。

以上为仓库内主要文档与代码的逐文件说明，便于维护与二次开发。
