# RAGFlow 检索流程完整分析

本文围绕一次标准的 **Retrieval** 请求，按真实调用链梳理「用户问题 → 最终 chunks 返回」的全过程，并回答向量化、关键词与混合检索、检索优化、Rerank、以及入库阶段自动关键词/问题提取对检索的影响。

---

## 一、检索请求的入口与主调用链

### 1.1 入口

- **API 路由**：`POST /retrieval`
- **处理函数**：`api.apps.sdk.doc.retrieval_test(tenant_id)`
- **位置**：`api/apps/sdk/doc.py` 中由 `@manager.route("/retrieval", methods=["POST"])` 注册。

### 1.2 主调用链（概览）

```
doc.retrieval_test()
  → 参数校验、权限、question 清洗（strip）、metadata_condition → doc_ids
  → [可选] cross_languages(question)  → question 多语言扩展
  → [可选] keyword_extraction(chat_mdl, question)  → question += 关键词
  → label_question(question, kbs)  → rank_feature（标签权重）
  → settings.retriever.retrieval(question, embd_mdl, tenant_ids, kb_ids, page, size,
        similarity_threshold, vector_similarity_weight, top, doc_ids,
        rerank_mdl=..., highlight=..., rank_feature=...)
       → Dealer.search(req, ...)   // 问题向量化 + 混合检索
       → [若有 rerank_mdl] Dealer.rerank_by_model(...) 或 [否则] Dealer.rerank(...)
       → 相似度阈值过滤、分页、组装 ranks
  → [可选] retrieval_by_toc(question, ranks["chunks"], ...)  // TOC 增强
  → retrieval_by_children(ranks["chunks"], ...)                 // 子 chunk 召回
  → [可选] kg_retriever.retrieval(...)                         // 知识图谱增强
  → 去掉 vector、重命名字段 → 返回 get_result(data=ranks)
```

- **retriever** 实例：`common.settings` 中 `retriever = search.Dealer(docStoreConn)`，即 **`rag.nlp.search.Dealer`**。
- **docStoreConn**：根据配置为 ES/Infinity 等文档库连接，由 `common.settings` 初始化。

---

## 二、问题向量化机制

### 2.1 在哪个模块、哪个方法中完成向量化？

- **模块**：`rag.nlp.search`
- **类/方法**：`Dealer.get_vector(self, txt, emb_mdl, topk=10, similarity=0.1)`
- **调用位置**：在 `Dealer.search()` 内，当 `req` 带 `question` 且传入了 `emb_mdl` 时，会调用：
  - `matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))`
- **向量化实现**：内部通过线程池调用 **`emb_mdl.encode_queries(txt)`**，得到一维向量，再封装为 `MatchDenseExpr` 用于稠密检索（如 cosine + topk + similarity 阈值）。
- **Embedding 模型**：来自请求对应的知识库配置 `kb.embd_id`，在 `doc.retrieval_test` 中构造为 `LLMBundle(kb.tenant_id, LLMType.EMBEDDING, llm_name=kb.embd_id)`，在 `retrieval()` 中作为 `embd_mdl` 传入 `search()`。

因此：**用户问题的向量化在 `rag.nlp.search.Dealer.get_vector` 中完成，实际计算由 `emb_mdl.encode_queries(question)` 执行。**

### 2.2 向量化之前的预处理

- **在 API 层（doc.py）**：
  - 对 `question` 做 `strip()`，空或仅空白则直接返回空结果。
  - **多语言扩展（可选）**：若请求带 `cross_languages` 语言列表，会先调用 `rag.prompts.generator.cross_languages(tenant_id, None, question, langs)`，用 Chat 模型将问题扩展为多语言版本（多句用 `\n` 或 `===` 分隔），**整段替换为扩展后的字符串**，再参与后续检索。
  - **关键词拼接到 question（可选）**：若请求 `keyword=True`，会调用 `rag.prompts.generator.keyword_extraction(chat_mdl, question)`，用 LLM 抽取关键词，返回值**直接拼接到 question 后面**（`question += await keyword_extraction(...)`），**不做单独字段**。拼接后的整串再传入 `retrieval()`，因此会一起被向量化和关键词匹配。

- **在 search 层**：
  - `Dealer.search()` 使用的 `qst` 来自 `req.get("question", "")`，即上面已经预处理（含可选多语言、关键词拼接）的字符串。
  - `Dealer.get_vector(txt, ...)` 对 `txt` **不再做额外清洗或语言转换**，直接交给 `emb_mdl.encode_queries(txt)`。即**向量化前的“预处理”都在 doc.py 完成**。

### 2.3 向量化之后的额外处理

- **检索侧**：
  - 得到的向量仅用于构建 `MatchDenseExpr`，与关键词表达式一起送入文档库的 **weighted_sum** 融合（见下节）。代码中**没有**对 query 向量做归一化、缓存或权重融合的额外步骤；归一化/权重由底层引擎（如 Infinity/ES）在 fusion 时处理。
- **Embedding 模型内部**：是否归一化、是否区分 query/passage 由 `rag.llm.embedding_model` 各实现决定（如 `encode_queries` 与 `encode` 的差异），不在检索流程里再处理。

---

## 三、关键词提取与混合检索机制

### 3.1 检索前是否对问题进行关键词提取？

- **两种“关键词”来源**：
  1. **请求级“关键词”**：即上一节所述，当 `req.get("keyword", False)` 为 True 时，用 **LLM 做 keyword_extraction**，结果**拼接到 question**，再整体参与向量化与全文检索。这是“检索前对问题的扩展”，而不是单独的关键词检索字段。
  2. **检索内部的关键词/全文**：在 `Dealer.search()` 里，对当前 `qst`（可能已含扩展）调用 **`self.qryr.question(qst, min_match=0.3)`**，得到：
     - `matchText`：用于全文/关键词匹配的表达式（`MatchTextExpr`）；
     - `keywords`：用于高亮等。

- 因此：**检索前**可以有一次 **LLM 关键词提取并拼接到 question**；**检索时**一定会用 **FulltextQueryer.question()** 对 question 做分词、权重、同义词等得到全文查询，**不是**“先单独关键词检索再合并”，而是**同一条 question 同时驱动关键词匹配和向量匹配**。

### 3.2 关键词与向量检索如何结合？拼接到 query 还是 hybrid？

- **结合方式**：**Hybrid Search（向量 + 关键词/全文）**，在**同一次文档库 search** 里完成。
- **模块**：`rag.nlp.search.Dealer.search()`；底层查询构建依赖 `rag.nlp.query.FulltextQueryer.question()` 与 `common.doc_store.doc_store_base` 的 `MatchTextExpr`、`MatchDenseExpr`、`FusionExpr`。

**具体逻辑**：

1. **关键词/全文侧**：  
   `matchText, keywords = self.qryr.question(qst, min_match=0.3)`  
   - `FulltextQueryer.question()` 会对 `qst` 做：繁简转换、全角转半角、小写、去特殊字符、分词、词权重、同义词扩展、短语与细粒度分词等，生成针对多字段的 **MatchTextExpr**（字段含 `title_tks`、`important_kwd`、`question_tks`、`content_ltks` 等及权重）。
2. **向量侧**：  
   `matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))`  
   - 使用同一 `qst` 的向量做稠密检索。
3. **融合**：  
   `fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})`  
   `matchExprs = [matchText, matchDense, fusionExpr]`  
   - 文档库一次 search 同时做：全文匹配 + 向量匹配，再按 **0.05（关键词）: 0.95（向量）** 做加权和，得到融合分数与候选列表。

所以：**不是**“把关键词拼接到 query 再做一次 embedding”，而是**同一条 question 既生成全文查询又生成向量，在引擎内做 hybrid search（向量 + 关键词）**。请求里的 `keyword=True` 只是**在进入检索前**用 LLM 把关键词**拼接到 question 文本**，从而让后续的全文与向量都基于“问题+关键词”的整串。

### 3.3 相关模块与方法小结

| 内容           | 模块                     | 方法/位置 |
|----------------|--------------------------|-----------|
| 检索前关键词扩展 | `rag.prompts.generator`  | `keyword_extraction(chat_mdl, content, topn=3)`，在 doc.py 中拼接到 question |
| 全文/关键词查询构建 | `rag.nlp.query`          | `FulltextQueryer.question(txt, min_match=0.6)`，被 `Dealer.search` 调用 |
| 问题向量化     | `rag.nlp.search`          | `Dealer.get_vector(txt, emb_mdl, topk, similarity)` → `emb_mdl.encode_queries(txt)` |
| 混合检索与融合 | `rag.nlp.search`          | `Dealer.search()` 中 `matchExprs = [matchText, matchDense, fusionExpr]`，weights `"0.05,0.95"` |
| 文档库执行     | `common.doc_store` / 各引擎封装 | `dataStore.search(..., matchExprs, ...)` |

---

## 四、检索阶段的优化机制

以下按“检索前 / 检索中 / 检索后”归纳，并标明模块与方法。

### 4.1 相似度阈值过滤

- **位置**：`rag.nlp.search.Dealer.retrieval()` 内，在拿到 `search()` 结果并完成 rerank（或直接用引擎分数）之后。
- **逻辑**：用 `vector_similarity_weight` 得到 `post_threshold`（当向量权重为 0 时阈值为 0，否则为 `similarity_threshold`），再 `valid_idx = [i for i in sorted_idx if sim_np[i] >= post_threshold]`，只保留超过阈值的候选。
- **参数**：请求中的 `similarity_threshold`（默认 0.2）、`vector_similarity_weight`（默认 0.3）。

### 4.2 向量权重控制

- **检索阶段（引擎内）**：在 `Dealer.search()` 中，`FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})` 固定为 **关键词 0.05、向量 0.95**，由文档库在检索时计算融合分。
- **Rerank 阶段**：在 `retrieval()` 里调用 `rerank()` 或 `rerank_by_model()` 时，传入 `tkweight=1 - vector_similarity_weight`、`vtweight=vector_similarity_weight`，用于**重算**每个候选的混合分（词权 + 向量/rerank 分），从而用请求级的 `vector_similarity_weight` 控制**最终排序**时的向量 vs 关键词权重。

### 4.3 Rerank 前的候选集裁剪

- **位置**：`rag.nlp.search.Dealer.retrieval()`。
- **逻辑**：先构造 `RERANK_LIMIT = max(30, ceil(64/page_size)*page_size)`，在 `req` 里设 `"size": RERANK_LIMIT`，即**只向文档库取前 RERANK_LIMIT 条**作为候选，再在这批候选上做 rerank 与阈值过滤。因此“候选集裁剪”是通过 **size 上限** 在 search 阶段完成的。

### 4.4 TOC 增强

- **位置**：`api.apps.sdk.doc.retrieval_test()` 中，在 `retriever.retrieval()` 返回之后；若 `req.get("toc_enhance", False)` 为 True 则执行。
- **方法**：`settings.retriever.retrieval_by_toc(question, ranks["chunks"], tenant_ids, chat_mdl, size)`（`rag.nlp.search.Dealer.retrieval_by_toc()`）。
- **逻辑**：按文档聚合得分选出一个主文档，拉取该文档的 TOC 块（`toc_kwd: "toc"`），用 `rag.prompts.generator.relevant_chunks_with_toc`（Chat 模型）根据 question 从 TOC 中选出相关 chunk id 与得分，再把这些 chunk 并入或替换现有 chunks，最后按相似度排序取 topn。属于**检索后、基于 TOC 的再排序与扩充**。

### 4.5 子 chunk 召回

- **位置**：同上，在 `retrieval_test()` 里，紧接 TOC 增强之后（无论是否开启 TOC）。
- **方法**：`settings.retriever.retrieval_by_children(ranks["chunks"], tenant_ids)`（`rag.nlp.search.Dealer.retrieval_by_children()`）。
- **逻辑**：遍历当前 chunks，若存在 `mom_id`（父块 id），则把这些 chunk 从列表中移出，按父 id 分组，再根据父 id 从文档库取父块内容，将同一父块下的子块内容合并成一个“父级”块（内容拼接、相似度取子块平均等），用父级块替代子块列表。即**子块聚合成父块再返回**，避免只返回零散子块。

### 4.6 Metadata 过滤

- **位置**：`api.apps.sdk.doc.retrieval_test()` 开头。
- **逻辑**：若请求带了 `metadata_condition` 且未指定 `document_ids`，则通过 `DocMetadataService.get_meta_by_kbs(kb_ids)` 与 `meta_filter(..., convert_conditions(metadata_condition), ...)` 得到满足条件的 `doc_ids`，再把这些 `doc_ids` 传入 `retrieval()`。在 `Dealer.search()` 里通过 `get_filters(req)` 将 `doc_ids` 转为 `doc_id` 过滤条件，从而**只在这些文档的 chunk 中检索**。

### 4.7 多语言增强

- **位置**：`doc.retrieval_test()`，在调用 `retrieval()` 之前。
- **方法**：`rag.prompts.generator.cross_languages(tenant_id, None, question, langs)`。
- **逻辑**：用 Chat 模型把用户问题扩展成多种语言的等价表述，返回的整段字符串**替换原 question**，后续向量化与关键词检索都基于扩展后的问题，相当于**查询扩展**。

### 4.8 知识图谱增强

- **位置**：`doc.retrieval_test()`，在 TOC、children 处理之后；当 `req.get("use_kg", False)` 为 True 时执行。
- **方法**：`settings.kg_retriever.retrieval(question, tenant_ids, kb_ids, embd_mdl, chat_mdl)`（`rag.graphrag.search.KGSearch.retrieval()`）。
- **逻辑**：基于知识图谱做实体、关系、多跳路径等检索，返回结构化的 KG 结果（如 `content_with_weight` 等），并**插入到 ranks["chunks"] 头部**，与普通文档 chunk 一起返回。

### 4.9 小结表

| 策略               | 阶段     | 模块/位置              | 方法/要点 |
|--------------------|----------|------------------------|-----------|
| 相似度阈值过滤     | 检索后   | `rag.nlp.search.Dealer` | `retrieval()` 内按 `post_threshold` 过滤 |
| 向量权重控制       | 检索中/后 | `rag.nlp.search`       | search 内 0.05/0.95；rerank 用 `vector_similarity_weight` |
| Rerank 前候选裁剪  | 检索中   | `rag.nlp.search.Dealer` | `retrieval()` 中 `size=RERANK_LIMIT` |
| TOC 增强           | 检索后   | `rag.nlp.search.Dealer` | `retrieval_by_toc()` + `relevant_chunks_with_toc` |
| 子 chunk 召回      | 检索后   | `rag.nlp.search.Dealer` | `retrieval_by_children()` 按 mom_id 聚合成父块 |
| Metadata 过滤      | 检索前   | `api.apps.sdk.doc` + metadata 服务 | `meta_filter` → doc_ids → `get_filters` |
| 多语言增强         | 检索前   | `rag.prompts.generator` | `cross_languages()` 扩展 question |
| 知识图谱增强       | 检索后   | `rag.graphrag.search`   | `KGSearch.retrieval()` 结果插入 chunks 头部 |

---

## 五、重排序（Rerank）机制

### 5.1 是否存在 Rerank？

- **有**。在 `Dealer.retrieval()` 中，在 `search()` 返回候选后，一定会做一轮“打分/排序”：
  - 若请求传入了 **rerank_mdl**（即 `req.get("rerank_id")` 有值）：使用 **rerank 模型** 再算一次分；
  - 否则：使用**内置的混合相似度**（基于已有向量与词权）或引擎返回的 `_score` 作为分。

### 5.2 Rerank 模型如何加载？

- **加载位置**：`api.apps.sdk.doc.retrieval_test()` 中，`if req.get("rerank_id"): rerank_mdl = LLMBundle(kb.tenant_id, LLMType.RERANK, llm_name=req["rerank_id"])`。
- **类型**：`rag.llm.rerank_model` 下的具体实现（如 Jina、XInference、Nvidia、BGE 等），通过 LLM 配置与工厂创建，与普通 Embedding/Chat 一样由租户与模型名决定。

### 5.3 基于 Cross-Encoder 还是 Embedding 再计算？

- **基于 Cross-Encoder 风格**：Rerank 模型接口为 `similarity(query: str, texts: list)`，传入**一条 query** 和**多条候选文本**，返回每条文本与 query 的相关性分数（或排序）。实现里多为调用各厂商的 rerank API（如 Jina、BGE、Nvidia 等），这些 API 一般是 **query-document 联合编码的 cross-encoder**，而不是“再算一遍 embedding 再算 cosine”。
- **无 rerank 模型时**：使用 `Dealer.rerank()` 或引擎分数。`rerank()` 内部用**已有 chunk 向量**与 **query 向量**（来自 search 结果里的 `sres.query_vector`）算 cosine，再与**词权相似度**按 `tkweight/vtweight` 加权，并加上 **rank_feature（标签分）**，属于 **embedding + 词权** 的混合，不是 cross-encoder。

### 5.4 重排序在哪个阶段执行？

- **阶段**：在 **向量召回 + 关键词融合** 之后、**相似度阈值过滤与分页** 之前。
- **顺序**：`retrieval()` 内：  
  `sres = await self.search(...)` → 若有 `rerank_mdl` 则 `sim, tsim, vsim = self.rerank_by_model(rerank_mdl, sres, question, ...)`，否则 `sim, tsim, vsim = self.rerank(sres, question, ...)` 或（Infinity）直接用 `_score` → 再用 `sim` 做 `argsort`、阈值过滤、分页、组装 chunks。

---

## 六、存储阶段“自动关键词提取 / 自动问题提取”对检索的影响

### 6.1 入库时这些字段如何产生？

- **自动关键词提取**：在 `rag.svr.task_executor` 的解析流程中，若 `parser_config.get("auto_keywords")` 有值，会对每个 chunk 调用 `keyword_extraction(chat_mdl, d["content_with_weight"], topn)`，结果写入：
  - `d["important_kwd"]`：关键词列表（如 split by `","`）；
  - `d["important_tks"]`：对关键词拼接后做 `rag_tokenizer.tokenize(...)`，用于全文检索。
- **自动问题提取**：若 `parser_config.get("auto_questions", 0)` 有值，会对每个 chunk 调用 `question_proposal(chat_mdl, d["content_with_weight"], topn)`，结果写入：
  - `d["question_kwd"]`：问题列表（如 split by `"\n"`);
  - `d["question_tks"]`：对问题拼接后做 tokenize，用于全文检索。

### 6.2 入库时向量用哪些内容构建？

- **方法**：`rag.svr.task_executor.embedding(docs, mdl, parser_config, callback)`。
- **正文来源**：对每个 doc，若存在 `question_kwd`，则 **c = "\n".join(question_kwd)**；否则 **c = content_with_weight**。随后对 `c` 做简单清洗（去掉表格标签等），再 **truncate 后送入 mdl.encode()** 得到正文向量。
- **标题**：用 `docnm_kwd`（文件名）encode 得到标题向量。
- **最终 chunk 向量**：`filename_embd_weight * title_vec + (1 - filename_embd_weight) * content_vec`，即**自动问题提取的 question_kwd 会参与向量构建**（作为正文的代表）；**自动关键词 important_kwd 不参与向量构建**，只写入字段供检索用。

### 6.3 检索时这些字段是否参与向量构建？

- **不直接参与**：检索时只对**当前用户 question**（及可选扩展）做一次 `encode_queries(question)`，得到 query 向量；**不会**对 chunk 的 important_kwd/question_kwd 再算一遍向量。
- **间接影响**：chunk 的向量在**入库时**若开启了自动问题提取，则已用 question_kwd 代表正文参与 encode，因此**与“问题式”查询更对齐**；检索时用 question 的向量与这些 chunk 向量做相似度，相当于“问题-问题”语义匹配。

### 6.4 是否参与关键词匹配？

- **参与**。`rag.nlp.query.FulltextQueryer` 的 `query_fields` 中包含：
  - `important_kwd^30`、`important_tks^20`；
  - `question_tks^20`、`question_kwd` 对应字段也会被引擎索引。
- `Dealer.search()` 里用 `self.qryr.question(qst, ...)` 生成的 **MatchTextExpr** 会在这批字段上做全文/关键词匹配，因此**自动关键词和自动问题对应的 token 都会参与关键词匹配**，且权重较高。

### 6.5 是否影响 Rerank？

- **会**。在 `Dealer.rerank()` 和 `Dealer.rerank_by_model()` 中，会从 `sres.field[id]` 里取：
  - `content_ltks`、`title_tks`、`important_kwd`、`question_tks`
- 在 **rerank()** 中：  
  `tks = content_ltks + title_tks*2 + important_kwd*5 + question_tks*6`，用这些 token 与 query 的 keywords 做 **token_similarity**，再与向量相似度加权，得到最终 sim。因此 **important_kwd、question_tks 在 rerank 阶段被赋予更高权重**（5 倍、6 倍），直接影响最终排序。
- 在 **rerank_by_model()** 中：  
  用 `content_ltks + title_tks + important_kwd` 等拼成文本，与 query 一起送进 rerank 模型的 `similarity(query, texts)`，因此**自动关键词和问题对应的内容会进入 rerank 模型的输入**，影响 cross-encoder 打分。

### 6.6 相关逻辑所在模块

| 内容           | 模块                     | 说明 |
|----------------|--------------------------|------|
| 自动关键词/问题 | `rag.svr.task_executor`  | 解析流程中调用 `keyword_extraction` / `question_proposal`，写 `important_kwd`、`important_tks`、`question_kwd`、`question_tks` |
| 入库向量构建   | `rag.svr.task_executor.embedding()` | 正文优先用 `question_kwd` 拼接后 encode，再与标题向量加权 |
| 检索关键词匹配 | `rag.nlp.query.FulltextQueryer` | `query_fields` 含 important_kwd、question_tks 等；`Dealer.search()` 用其做 MatchTextExpr |
| Rerank 中使用  | `rag.nlp.search.Dealer`  | `rerank()` 中 token 权重含 important_kwd*5、question_tks*6；`rerank_by_model()` 中拼入这些字段送模型 |

---

## 七、从“用户问题”到“最终 chunks”的简要心智模型

1. **入口**：`doc.retrieval_test()` 收到 question、dataset_ids、可选 document_ids、metadata_condition、similarity_threshold、vector_similarity_weight、top_k、rerank_id、keyword、toc_enhance、cross_languages、use_kg 等。
2. **Question 预处理**：strip；可选 cross_languages 扩展；可选 keyword_extraction 拼接到 question；`label_question` 得到 rank_feature。
3. **检索**：`Dealer.retrieval()` 内构造 req（含 question、doc_ids、similarity、size=RERANK_LIMIT 等）→ `Dealer.search()`：
   - 全文：`FulltextQueryer.question(question)` → MatchTextExpr（含 important_kwd、question_tks 等字段）；
   - 向量：`get_vector(question, emb_mdl)` → MatchDenseExpr；
   - 融合：weighted_sum 0.05/0.95，一次 search 得到候选。
4. **Rerank**：对候选用 rerank_mdl（cross-encoder）或内置 rerank（embedding+词权+rank_feature）得到最终 sim，再阈值过滤、分页。
5. **后处理**：可选 retrieval_by_toc、retrieval_by_children、kg_retriever.retrieval；去掉 vector、重命名字段后返回。

这样即可完整理解 RAGFlow 的检索架构、向量与关键词的融合方式、rerank 与各类增强在整体中的位置，以及入库阶段自动关键词/问题提取对向量构建、关键词匹配和 rerank 的影响。
