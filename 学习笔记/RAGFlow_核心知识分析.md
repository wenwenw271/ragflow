# RAGFlow 核心知识分析

> 基于RAGFlow源码的分析笔记 - 用于自研RAG框架参考

---

## 一、项目概述

RAGFlow是一款开源的检索增强生成(RAG)引擎，核心特点：
- **深度文档理解**：从复杂格式的非结构化数据中提取信息
- **基于模板的文本切片**：可控可解释的分块策略
- **Agent工作流**：支持编排复杂的RAG流程
- **多数据源支持**：支持多种异构数据源接入

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGFlow 系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │   Web    │    │   API    │    │   DeepDoc│    │   Agent  │  │
│   │ (前端)   │◄──►│ (后端)   │◄──►│ (文档解析)│◄──►│ (工作流) │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                          │                    │                  │
│                          ▼                    ▼                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                      RAG Core Engine                      │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│   │  │ Embedding│  │  Rerank  │  │ Retrieval│  │   LLM     │ │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │                    基础设施层                              │   │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌─────┐ │   │
│   │  │ MySQL  │ │ ES/     │ │ Redis  │ │ MinIO  │ │Ocea-│ │   │
│   │  │        │ │Infinity│ │        │ │(文件存储)│ │nbase│ │   │
│   │  └────────┘ └────────┘ └────────┘ └────────┘ └─────┘ │   │
│   └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块说明

| 模块 | 目录 | 功能描述 |
|------|------|----------|
| api | `/api` | Flask/Quart后端API服务 |
| web | `/web` | React+TypeScript前端 |
| deepdoc | `/deepdoc` | 文档解析、OCR、布局识别 |
| rag | `/rag` | 核心RAG逻辑 |
| agent | `/agent` | Agent工作流引擎 |

---

## 三、核心技术模块详解

### 3.1 文档解析 (DeepDoc)

**位置**: `deepdoc/`

#### 3.1.1 视觉处理模块
- **OCR识别** (`vision/ocr.py`): 光学字符识别
- **布局识别** (`vision/layout_recognizer.py`): 文档版面分析
- **表格结构识别** (`vision/table_structure_recognizer.py`): TSR

**识别的10种基本布局组件**:
1. 文本(Text)
2. 标题(Title)
3. 配图(Figure)
4. 配图标题(Figure caption)
5. 表格(Table)
6. 表格标题(Table caption)
7. 页头(Header)
8. 页尾(Footer)
9. 参考引用(Reference)
10. 公式(Formula)

#### 3.1.2 解析器模块
支持的文档格式 (`parser/`):
- PDF: `pdf_parser.py`
- Word: `docx_parser.py`
- Excel: `excel_parser.py`
- PPT: `ppt_parser.py`
- Markdown: `markdown_parser.py`
- HTML: `html_parser.py`
- JSON: `json_parser.py`
- 简历专用: `resume/`
- 支持MinerU和Docling解析

---

### 3.2 文本分块 (Chunking)

**位置**: `rag/flow/splitter/splitter.py`

#### 核心参数
```python
class SplitterParam:
    chunk_token_size = 512      # 分块token数
    delimiters = ["\n"]          # 分隔符
    overlapped_percent = 0       # 重叠百分比
    children_delimiters = []     # 子分隔符
    table_context_size = 0      # 表格上下文大小
    image_context_size = 0      # 图像上下文大小
```

#### 分块策略
1. **Naive分块**: 简单按分隔符切分
2. **基于模板的分块**: 预定义模板规则
3. **语义分块**: 使用LLM理解语义边界

---

### 3.3 向量化 (Embedding)

**位置**: `rag/llm/embedding_model.py`

#### 支持的Embedding模型
- **内置模型**: BAAI/bge-m3, BAAI/bge-small-en-v1.5, Qwen/Qwen3-Embedding-0.6B
- **支持自定义**: OpenAI, Ollama, Claude, Gemini等

#### 核心接口
```python
class Base(ABC):
    def encode(self, texts: list): pass
    def encode_queries(self, text: str): pass
```

#### Token限制
```python
MAX_TOKENS = {
    "Qwen/Qwen3-Embedding-0.6B": 30000,
    "BAAI/bge-m3": 8000,
    "BAAI/bge-small-en-v1.5": 500
}
```

---

### 3.4 检索模块 (Retrieval)

**位置**: `rag/nlp/search.py`, `rag/nlp/query.py`

#### 3.4.1 全文检索
```python
class FulltextQueryer:
    query_fields = [
        "title_tks^10",      # 标题权重最高
        "title_sm_tks^5",
        "important_kwd^30",   # 关键词高权重
        "important_tks^20",
        "question_tks^20",
        "content_ltks^2",     # 内容次要
        "content_sm_ltks",
    ]
```

#### 3.4.2 检索流程
1. **Query预处理**: 同义词扩展、繁简体转换
2. **多路召回**: 全文检索 + 向量检索
3. **融合重排序**: 使用Rerank模型

#### 3.4.3 向量检索
```python
async def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
    # 使用cosine相似度
    return MatchDenseExpr(
        vector_column_name, 
        embedding_data, 
        'float', 
        'cosine', 
        topk, 
        {"similarity": similarity}
    )
```

---

### 3.5 重排序 (Rerank)

**位置**: `rag/llm/rerank_model.py`

#### 支持的Rerank模型
- Jina Rerank
- BAAI Rerank
- CoHERE Rerank

#### 算法原理
1. **多路召回**: 获取多批次候选结果
2. **粗排**: 使用向量相似度初筛
3. **精排**: 使用Rerank模型重排
4. **归一化处理**:
```python
@staticmethod
def _normalize_rank(rank: np.ndarray) -> np.ndarray:
    min_rank = np.min(rank)
    max_rank = np.max(rank)
    if not np.isclose(min_rank, max_rank, atol=1e-3):
        rank = (rank - min_rank) / (max_rank - min_rank)
    else:
        rank = np.zeros_like(rank)
    return rank
```

---

### 3.6 Agent工作流引擎

**位置**: `agent/`, `rag/flow/pipeline.py`

#### 3.6.1 核心概念
- **Graph**: 有向无环图(DAG)定义工作流
- **Component**: 组件(节点)
- **Pipeline**: 流程执行器

#### 3.6.2 DSL定义
```python
dsl = {
    "components": {
        "begin": {
            "obj": {"component_name": "Begin", "params": {}},
            "downstream": ["retrieval_0"],
            "upstream": [],
        },
        "retrieval_0": {
            "obj": {"component_name": "Retrieval", "params": {}},
            "downstream": ["generate_0"],
            "upstream": ["begin"],
        }
    },
    "history": [],
    "path": ["begin"],
    "retrieval": {"chunks": [], "doc_aggs": []},
    "globals": {...}
}
```

#### 3.6.3 内置组件 (`agent/component/`)
- **Begin**: 流程开始
- **Retrieval**: 知识检索
- **Generate**: LLM生成
- **Categorize**: 分类
- **Loop/Iteration**: 循环迭代
- **Switch**: 条件分支
- **LLM**: 大语言模型调用
- **Message**: 消息处理
- **ExcelProcessor**: Excel处理
- **CodeExec**: 代码执行

#### 3.6.4 工具集成 (`agent/tools/`)
- **搜索**: Google, DuckDuckGo, SearXNG, Tavily
- **数据库**: ExeSQL (自然语言转SQL)
- **金融**: Tushare, AkShare, YahooFinance, Jin10
- **学术**: ArXiv, GoogleScholar, PubMed
- **工具**: Crawler, Wikipedia, Github
- **MCP支持**: Model Context Protocol

---

### 3.7 高级RAG技术

**位置**: `rag/advanced_rag/`

#### Tree-structured Query Decomposition
```python
from rag.advanced_rag import DeepResearcher

# 将复杂问题分解为子问题
# 递归检索并整合答案
```

---

## 四、数据流与处理流程

### 4.1 文档处理流程

```
文档上传
    │
    ▼
┌─────────────────┐
│  文件解析       │ ◄── DeepDoc (OCR/布局识别)
│ (PDF/DOCX...)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  文本提取       │ ◄── 多种解析器
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  文档分块       │ ◄── 模板/语义分块
│  (Chunking)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  向量化          │ ◄── Embedding Model
│  (Embedding)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  索引存储        │ ◄── ES/Infinity
│                 │
└─────────────────┘
```

### 4.2 问答处理流程

```
用户Query
    │
    ▼
┌─────────────────┐
│  Query预处理    │ ◄── 同义词/繁简转换
│                 │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│全文检索│ │向量检索│
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  结果融合       │ ◄── Rerank
│  与重排序       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM生成答案    │
│  (带引用)       │
└────────┬────────┘
         │
         ▼
   返回结果
```

---

## 五、数据库设计

### 5.1 核心实体

| 实体 | 说明 |
|------|------|
| Tenant | 租户 |
| User | 用户 |
| KnowledgeBase | 知识库 |
| Document | 文档 |
| Chunk | 文本块 |
| Conversation | 对话 |
| Message | 消息 |
| Agent | Agent配置 |

### 5.2 存储方案
- **MySQL**: 业务数据存储
- **Elasticsearch/Infinity**: 向量检索
- **Redis**: 缓存、任务队列
- **MinIO**: 文件存储

---

## 六、自研RAG框架建议

### 6.1 核心组件设计

1. **文档解析层**
   - 统一解析接口
   - 支持多种格式
   - OCR集成

2. **分块策略层**
   - 固定大小分块
   - 滑动窗口重叠
   - 语义分块

3. **索引层**
   - 全文索引(ES)
   - 向量索引(Milvus/Infinity)
   - 混合检索

4. **检索层**
   - Query理解
   - 多路召回
   - 重排序

5. **生成层**
   - 上下文组装
   - Prompt模板
   - 引用追溯

### 6.2 关键技术选型

| 功能 | 推荐方案 |
|------|----------|
| 文档解析 | MinerU, Docling, PyMuPDF |
| 向量存储 | Milvus, Qdrant, Infinity |
| 全文检索 | Elasticsearch, OpenSearch |
| Embedding | BGE-M3,gte-Qwen2 |
| Rerank | BAAI Rerank, Jina Rerank |
| LLM | OpenAI, Claude, 本地部署 |

---

## 七、总结

RAGFlow是一个功能完整的RAG系统，核心优势：
1. **深度文档理解**: 解决了非结构化文档解析难题
2. **可控的分块**: 模板化分块策略
3. **Agent编排**: 灵活的工作流支持
4. **多数据源**: 丰富的接入方式
5. **可观测性**: 完整的日志和追踪

对于自研RAG框架，可以参考其模块化设计思想，重点突破文档解析和检索两个核心环节。

---

*文档生成时间: 2025-02-26*
