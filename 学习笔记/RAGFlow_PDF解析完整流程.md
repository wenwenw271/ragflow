# RAGFlow PDF 解析完整流程

本文档基于代码分析，梳理从文档上传到生成 Chunk 的完整 PDF 解析链路。

---

## 一、整体架构概览

```
用户上传 PDF
    → API 写入存储 + 创建文档记录
    → queue_tasks() 按页范围拆成多个 Task，推入 Redis
    → task_executor 消费 Task
        → build_chunks() 调用 naive.chunk()
            → 根据 layout_recognize 选择 PDF 解析器 (DeepDOC / Plain Text / MinerU / Docling / TCADP / PaddleOCR / Vision)
            → 解析得到 sections（文本块+位置）和 tables（表格）
            → naive_merge() 按 chunk_token_num 合并成 chunks
            → tokenize_table() + tokenize_chunks() 生成最终 chunk 列表（含 content、position、image 等）
        → embedding → insert_chunks 写入向量库
```

---

## 二、入口与任务拆分

### 2.1 文档上传与任务入队

- **入口**：文档上传后由 `DocumentService` / 数据集相关 API 触发解析。
- **任务拆分**：`api/db/services/task_service.py` 中的 `queue_tasks(doc, bucket, name, priority)`：
  - 若 `doc["type"] == FileType.PDF`：
    - 用 `PdfParser.total_page_number(doc["name"], file_bin)` 获取总页数。
    - 按 `task_page_size`（默认 12 页）将 PDF 切成多个页范围。
    - 每个页范围生成一个 Task：`from_page`, `to_page`，并计算 `digest` 用于复用。
  - 其他类型（如 Excel）按行范围或单任务处理。
- **执行**：Task 写入 DB 后，未完成的任务被推入 Redis 队列，由 `rag/svr/task_executor.py` 中的消费者拉取执行。

### 2.2 任务执行器中的 Chunk 构建

- **入口函数**：`task_executor.build_chunks(task, progress_callback)`。
- **解析器选择**：`chunker = FACTORY[task["parser_id"].lower()]`，PDF 文档通常使用 `parser_id = "general"` 或 `"naive"`，即 `rag.app.naive` 模块。
- **实际调用**：
  ```python
  cks = await thread_pool_exec(
      chunker.chunk,
      task["name"],
      binary=binary,
      from_page=task["from_page"],
      to_page=task["to_page"],
      lang=task["language"],
      callback=progress_callback,
      kb_id=task["kb_id"],
      parser_config=task["parser_config"],
      tenant_id=task["tenant_id"],
  )
  ```
- **返回值**：`cks` 为 chunk 字典列表；后续会做 image 上传 MinIO、embedding、写入向量库。

---

## 三、naive.chunk() 中的 PDF 分支

文件：`rag/app/naive.py`。

### 3.1 解析器选择（layout_recognize）

- 从 `parser_config.get("layout_recognize", "DeepDOC")` 得到配置，经 `normalize_layout_recognizer()` 得到 `layout_recognizer` 和可选的 `parser_model_name`。
- 根据名称映射到具体解析函数：
  - `PARSERS = {
      "deepdoc": by_deepdoc,
      "mineru": by_mineru,
      "docling": by_docling,
      "tcadp": by_tcadp,
      "paddleocr": by_paddleocr,
      "plaintext": by_plaintext,
    }`
- 若配置为「DeepDOC」或未配置，走 **by_deepdoc**，内部使用 `Pdf`（继承自 `deepdoc.parser.pdf_parser.RAGFlowPdfParser`）做版面+表格+OCR。
- 若为「Plain Text」，走 **by_plaintext**：`PlainParser()` 或带视觉模型的 `VisionParser()`，只做纯文本或简单版面。
- 其他如 MinerU、Docling、TCADP、PaddleOCR 为独立解析管线，返回的也是 `(sections, tables, pdf_parser)` 形态，便于后续统一分块。

### 3.2 调用解析器得到 sections 与 tables

```python
sections, tables, pdf_parser = parser(
    filename=filename,
    binary=binary,
    from_page=from_page,
    to_page=to_page,
    lang=lang,
    callback=callback,
    layout_recognizer=layout_recognizer,
    mineru_llm_name=parser_model_name,
    paddleocr_llm_name=parser_model_name,
    **kwargs,
)
```

- **sections**：列表，每项为 `(text, position_tag)`，text 为一段文本，position_tag 为 `@@页码\tx0\tx1\ttop\tbottom##` 形式的位置信息，用于后续 crop 图片、写回 chunk 的 positions。
- **tables**：表格列表（结构因解析器而异），DeepDOC 下为带 HTML/文本和位置的表格数据。

### 3.3 表格与正文分块

- **表格**：`res = tokenize_table(tables, doc, is_english)`，将每个表格转成独立 chunk（含表格文本/HTML 等）。
- **正文**：
  - 若为 tcadp/docling/mineru/paddleocr，会设 `parser_config["chunk_token_num"] = 0`，仅用表格分块逻辑。
  - 否则对 sections 做：
    - `chunks = naive_merge(sections, chunk_token_num, delimiter, overlapped_percent)`：按 token 数和分隔符合并成若干 chunk。
    - `res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser, ...))`：为每个 chunk 生成 content、positions、可选 image（通过 `pdf_parser.crop(position_tag)` 从页面图裁剪），并追加到 `res`。
- 若有 `section_images`（如部分解析器带图），则走 `naive_merge_with_images` + `tokenize_chunks_with_images`。
- 最终 `res` 即为该 PDF 在本 Task 页范围内的全部 chunk 列表，返回给 `build_chunks()`。

---

## 四、解析得到文本块后的向量化与存储

在 `task_executor.do_handle_task()` 中，`build_chunks()` 返回 chunk 列表之后，依次进行：**图片上传 MinIO** → **可选：关键词/问题生成** → **向量化（embedding）** → **写入向量库（insert_chunks）**。以下按执行顺序说明。

### 4.1 图片上传 MinIO（build_chunks 内）

- **位置**：`rag/svr/task_executor.py` 的 `build_chunks()`，在 `chunker.chunk()` 返回 `cks` 之后立即执行。
- **逻辑**：
  - 为每个 chunk 补全 `doc_id`、`kb_id`、`create_time`、`id`（由 `content_with_weight` + `doc_id` 的 xxhash 生成）。
  - 若 chunk 带有 `image` 字段（正文/表格裁剪图），则调用 `image2id(...)` 将图片上传到 MinIO，得到 `img_id` 写入 chunk；无图则 `img_id = ""`。
  - 使用 `asyncio.gather` 并发执行所有 chunk 的 `upload_to_minio`，完成后得到 `docs`（即带 `img_id` 的 chunk 列表），作为后续 embedding 与存储的输入。

### 4.2 可选：关键词与问题生成

- 若 `parser_config.auto_keywords` 非 0：对每个 chunk 调用 `keyword_extraction()`，结果写入 `important_kwd`、`important_tks`。
- 若 `parser_config.auto_questions` 非 0：对每个 chunk 调用 `question_proposal()`，结果写入 `question_kwd`、`question_tks`。
- 二者均可能使用 LLM 缓存（`get_llm_cache` / `set_llm_cache`）避免重复调用。

### 4.3 向量化（embedding）

- **入口**：`token_count, vector_size = await embedding(chunks, embedding_model, task_parser_config, progress_callback)`（`task_executor.do_handle_task`）。
- **实现**：`rag/svr/task_executor.py` 中的 `async def embedding(docs, mdl, parser_config=None, callback=None)`。

**与增强字段的关系（先有分块+增强，再为其中部分文本算向量，最后整条存储）**：

- 不是「把关键字、自动提问、原文等整段拼成一大段再整体向量化」；而是：**分块上先已有原文和增强字段**（`content_with_weight`、可选的 `question_kwd`、`important_kwd` 等），然后**只取其中一部分文本**去算**一个**向量，把这个向量作为新字段（`q_*_vec`）加进这条分块，最后**整条 chunk（原文 + 关键字 + 自动提问 + 向量 + 元数据）一起写入向量库**。
- 参与向量化的文本（每条 chunk 只对应一个向量）：
  - **正文**：若有 `question_kwd`（自动提问），则用 `"\n".join(question_kwd)` 作为“正文”送去编码；否则用 `content_with_weight`（原文/加权正文）。**关键字 `important_kwd` 不参与向量化**，只作为字段与原文一起存储，供检索高亮、排序等使用。
  - **标题**：`docnm_kwd`，与正文向量按 `filename_embd_weight` 加权合并后得到该 chunk 的最终向量。
- 因此流程是：**分块信息（含增强） → 从中选定“用于嵌入的文本” → 向量化得到一条向量 → 向量写回 chunk → 整条存储（原文 + 增强 + 向量）**。

**输入与文本准备**（代码对应）：

- 对每个 doc：标题取 `doc.get("docnm_kwd", "Title")`；正文取 `"\n".join(d.get("question_kwd", []))`，若为空则用 `d["content_with_weight"]`。
- 对正文做简单清洗：去掉表格标签 `</?table|td|caption|tr|th...>`，空则置为 `"None"`。

**编码与加权**：

- **标题向量**：若标题与正文数量一致，对标题列表取首条调用一次 `mdl.encode`，得到向量后按行复制为与正文等长的矩阵 `tts`。
- **正文向量**：按 `settings.EMBEDDING_BATCH_SIZE` 分批调用 `mdl.encode`（内部对单条做 `truncate(..., max_length - 10)`），得到矩阵 `cnts`；同时累计 token 消耗并回调进度（约 0.7～0.9）。
- **加权合并**：`filename_embd_weight`（默认 0.1）来自 `parser_config`，若标题与正文向量维度一致，则  
  `vects = title_w * tts + (1 - title_w) * cnts`；否则仅用正文向量。

**写回 chunk**：

- 对每个 doc：`d["q_%d_vec" % len(v)] = v`，即按向量维度生成字段名（如 `q_768_vec`），将加权后的向量写入 chunk。
- 返回 `(tk_count, vector_size)` 供上游统计与索引创建使用。

**4.3.1 以 OpenAI 嵌入模型为例：创建与使用**

- **实现类**：`rag/llm/embedding_model.py` 中的 `OpenAIEmbed`，`_FACTORY_NAME = "OpenAI"`。构造函数 `__init__(self, key, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1")`，内部使用 `OpenAI(api_key=key, base_url=base_url)` 创建官方 SDK 客户端；默认模型名可在配置中覆盖（如 `text-embedding-3-small`、`text-embedding-3-large`）。

- **注册到 EmbeddingModel**（具体过程）：
  - 在 `rag/llm/__init__.py` 中，先得到空字典：`EmbeddingModel = globals().get("EmbeddingModel", {})`，再通过 `MODULE_MAPPING` 把模块名 `"embedding_model"` 和这个字典对应起来。
  - 对 `module_name = "embedding_model"` 执行 `importlib.import_module("rag.llm.embedding_model")`，加载 `embedding_model.py`，此时该模块里已有 `Base`、`OpenAIEmbed`、`AzureEmbed` 等类（都尚未注册）。
  - 用 `inspect.getmembers(module)` 扫描该模块里所有对象，找到名为 `"Base"` 的类，记作 `base_class`（即 `embedding_model.Base`）。
  - 再遍历一次 `inspect.getmembers(module)`：对每一个 **类** `obj`，若满足「是 `base_class` 的子类」「不是 `base_class` 本身」「有属性 `_FACTORY_NAME`」，则执行 `mapping_dict[obj._FACTORY_NAME] = obj`。这里 `mapping_dict` 就是上面的 `EmbeddingModel`。
  - 因此对 `OpenAIEmbed`（继承 `Base`，且 `_FACTORY_NAME = "OpenAI"`）会执行 `EmbeddingModel["OpenAI"] = OpenAIEmbed`；同理 `AzureEmbed` 的 `_FACTORY_NAME = "Azure-OpenAI"` 会得到 `EmbeddingModel["Azure-OpenAI"] = AzureEmbed`。若 `_FACTORY_NAME` 是列表（如 `["VLLM", "OpenAI-API-Compatible"]`），则用列表中每个字符串各注册一次，都指向同一类。
  - 总结：**注册是“按模块扫描子类 + 用 _FACTORY_NAME 当 key 写进字典”**，无需手写 `EmbeddingModel["OpenAI"] = OpenAIEmbed`，新增一个继承 `Base` 并定义 `_FACTORY_NAME` 的类就会在导入时自动进 `EmbeddingModel`。

- **配置来源**：知识库/任务侧使用「嵌入模型 ID」`embd_id`，格式可为 `模型名@厂商`（如 `text-embedding-3-small@OpenAI`）。该 ID 来自知识库的 `embd_id` 或租户默认嵌入，对应在 **TenantLLM** 表中为该租户绑定的 API Key、API Base、模型名等；厂商名（如 `OpenAI`）用于查找 `EmbeddingModel`。
- **创建实例流程**（以任务执行时为例）：
  1. `do_handle_task` 中从 task 取 `task_embedding_id = task["embd_id"]`（来自知识库的 `Knowledgebase.embd_id`）。
  2. 调用 `embedding_model = LLMBundle(task_tenant_id, LLMType.EMBEDDING, llm_name=task_embedding_id, lang=task_language)`。
  3. `LLMBundle` 继承 `LLM4Tenant`，在 `__init__` 中执行 `self.mdl = TenantLLMService.model_instance(tenant_id, LLMType.EMBEDDING, llm_name)`。
  4. `model_instance` 内部：先 `get_model_config(tenant_id, EMBEDDING, llm_name)`，从 TenantLLM 查出该模型的 `api_key`、`llm_name`、`api_base`、`llm_factory`（如 `"OpenAI"`）；再执行 `EmbeddingModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])`，即 `OpenAIEmbed(api_key, model_name, base_url=api_base)`，得到真正的嵌入模型实例。
- **使用**：`embedding_model.encode(texts)` 在 `LLMBundle` 中先对文本做长度截断（`max_length`），再调用 `self.mdl.encode(safe_texts)`，即 `OpenAIEmbed.encode`：按批大小 16 调用 `self.client.embeddings.create(input=texts[...], model=self.model_name, encoding_format="float", extra_body={"drop_params": True})`，汇总 `res.data[].embedding` 与 token 消耗并返回；检索时用 `encode_queries(text)` 对单条查询编码。Token 消耗会通过 `TenantLLMService.increase_usage` 记入租户用量。
- **预置与初始化**：`conf/llm_factories.json` 中 OpenAI 工厂下预置了 `text-embedding-ada-002`、`text-embedding-3-small`、`text-embedding-3-large` 等；`api/db/init_data.py` 的 `init_llm_factory()` 会为已有 OpenAI 的租户插入上述两个 3.x 嵌入模型到 TenantLLM，供前端选择并绑定 API Key 后作为知识库的 `embd_id` 使用。

### 4.4 写入向量库（insert_chunks）

- **入口**：`insert_result = await insert_chunks(task_id, task_tenant_id, task_dataset_id, chunks, progress_callback)`。
- **实现**：`rag/svr/task_executor.py` 中的 `async def insert_chunks(task_id, task_tenant_id, task_dataset_id, chunks, progress_callback)`。

**索引与存储抽象**：

- 索引名：`search.index_name(task_tenant_id)`，即 `ragflow_{tenant_id}`；数据集/知识库维度由 `task_dataset_id`（即 `kb_id`）区分。
- 实际存储由 `settings.docStoreConn` 完成，支持 **Elasticsearch**、**Infinity**、**OpenSearch**、**OB** 等；同一索引名 + `kb_id` 对应一个逻辑表/索引。

**mother chunk 先行**：

- 若 chunk 带有 `mom` 或 `mom_with_weight`，会先聚合出唯一的 “mother” 记录（`mom_id`、`content_with_weight` 等），按 `DOC_BULK_SIZE` 批量 `docStoreConn.insert` 写入，再写普通 chunk。

**批量写入与进度**：

- 普通 chunk 按 `settings.DOC_BULK_SIZE` 分批调用 `settings.docStoreConn.insert(chunks[b:b + DOC_BULK_SIZE], index_name, task_dataset_id)`。
- 每批成功后调用 `TaskService.update_chunk_ids(task_id, chunk_ids_str)` 将本批 chunk id 追加到 Task 记录；若 Task 已不存在（如被删除），则回滚：删除刚写入的 chunk 并删除 MinIO 上对应图片，然后返回 `False`。
- 进度回调在每 128 条左右更新（约 0.8～0.9）。

**索引的创建时机**：

- **Elasticsearch**：索引通常在首次写入前由业务侧按需 `create_idx`（如 `document_service.doc_upload_and_parse` 中先 `index_exist` 再 `create_idx`）；task_executor 路径下若知识库已有其他文档则索引已存在。
- **Infinity**：若表不存在，`docStoreConn.insert` 内部会捕获 `TABLE_NOT_EXIST`，根据首条 chunk 的 `q_*_vec` 推断 `vector_size` 及可选 `parser_id`，再调用 `create_idx` 建表后重试插入，因此可延迟到首次 insert 时建表。

### 4.5 后续收尾

- 调用 `DocumentService.increment_chunk_num(task_doc_id, task_dataset_id, token_count, chunk_count, 0)` 更新文档的 chunk 数与 token 统计。
- 若开启 `toc_extraction`，会异步生成目录 chunk（`build_TOC`），再执行一次 `insert_chunks` 写入并增加 chunk 计数。
- 若任务被取消，在 `finally` 中会按 `doc_id` 删除该文档在向量库中的 chunk，并清理 MinIO 上对应图片。

---

## 五、DeepDOC 核心：RAGFlowPdfParser

文件：`deepdoc/parser/pdf_parser.py`。  
`naive.Pdf` 继承自 `RAGFlowPdfParser`，其 `__call__` 在 naive 里被当作「解析一次 PDF」的入口；Flow 里则常用 `parse_into_bboxes()`（见下）。

### 5.1 主流程 __call__(fnm, ...)

```text
1. __images__(fnm, zoomin)           # 页转图 + 文本/OCR
2. _layouts_rec(zoomin)              # 版面分析
3. _table_transformer_job(zoomin)   # 表格检测与结构识别（含可选旋转）
4. _text_merge()                    # 文本块合并
5. _concat_downward()               # 向下拼接
6. _filter_forpages()               # 过滤页眉页脚等
7. _extract_table_figure(...)       # 表格与图表抽取
8. __filterout_scraps(boxes, zoomin)# 过滤碎片，得到最终 boxes
返回: (boxes 转成的 sections, tbls)
```

### 5.2 __images__：页转图 + 字符 + OCR

- **输入**：PDF 路径或二进制、`zoomin`（默认 3）、`page_from`/`page_to`。
- **步骤**：
  1. 用 **pdfplumber** 打开 PDF，将指定页范围转为高分辨率图：  
     `resolution=72*zoomin`，得到 `self.page_images`。
  2. 用 pdfplumber 的 `dedupe_chars().chars` 提取每页**已有文本**的字符框（坐标+文字），得到 `self.page_chars`，用于后续与 OCR 结果融合。
  3. 用 **pypdf** 读取大纲，得到 `self.outlines`（目录结构）。
  4. 根据 `page_chars` 中英文比例判断 `self.is_english`，决定是否主要依赖 PDF 内嵌文字。
  5. **逐页 OCR**（异步或同步）：
     - 调用 `self.__ocr(pagenum, img, chars, zoomin, device_id)`：
       - **检测**：`self.ocr.detect(img)`（`deepdoc/vision/ocr.py`）得到文本行框。
       - 将 pdfplumber 的 chars 按重叠关系合并到这些框里，补全/替换文字。
       - 无内嵌文字的框再调用 `self.ocr.recognize_batch()` 做识别。
     - 结果按页存入 `self.boxes`，每个元素为带 `x0,x1,top,bottom,text,page_number` 等的框。
  6. 若整页无框且 `zoomin<9`，会以更大 `zoomin` 重试一次 `__images__`。
- **OCR 实现**：`deepdoc/vision/ocr.py`，基于 PaddleOCR 等，提供 `detect` + `recognize_batch`。

### 5.3 _layouts_rec(zoomin)：版面分析

- **调用**：`self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)`。
- **layouter**：`LayoutRecognizer` 或 `AscendLayoutRecognizer`（由环境变量 `LAYOUT_RECOGNIZER_TYPE` 选择），在 `deepdoc/vision/layout_recognizer.py`。
- **作用**：对每页图像做版面检测，给每个区域打标签：Text、Title、Figure、Table、Header、Footer、Equation 等；并与当前页的 OCR boxes 做匹配，给每个 box 赋予 `layout_type` 等；同时得到 `page_layout`（每页的表格/图等区域框），供表格步骤使用。
- **坐标**：布局框坐标会按 `page_cum_height` 累加为「跨页纵向坐标」，便于后续排序与合并。

### 5.4 _table_transformer_job(zoomin, auto_rotate)：表格结构识别（TSR）

- **输入**：`page_layout` 中类型为 table 的区域。
- **步骤**：
  1. 按页、按表格裁剪出表格图像，可选 **auto_rotate**：
     - 对每个表格图做多角度（如 0°/90°/180°/270°）的 OCR 评估，选置信度最高的角度，得到 `rotated_table_imgs`。
  2. **表格结构识别**：`recos = self.tbl_det(imgs)`，`tbl_det` 为 `TableStructureRecognizer()`（`deepdoc/vision/`），输出每个单元格的框和类型（如 header、row、column、spanning cell）。
  3. 若开启了 auto_rotate：对旋转后的表格图重新 OCR，用新框更新 `self.boxes` 中对应表格区域内的框（`_ocr_rotated_tables`），保证表格内文字与旋转后图像一致。
  4. 根据 TSR 结果给 `self.boxes` 中表格类型的 box 打上 R/H/C/SP 等标记（行、表头、列、跨格），便于后续表格重组与导出。

### 5.5 _text_merge / _concat_downward / _filter_forpages

- **_text_merge**：先 `_assign_column()` 做列聚类（KMeans），再按同页、同列、同 layout 将相邻水平框合并成段落。
- **_concat_downward**：按阅读顺序做纵向拼接与顺序调整。
- **_naive_vertical_merge**（在 naive 的 Pdf 中会调用）：跨页合并页码、列表等短行。
- **_filter_forpages**：过滤页眉、页脚、页码等噪音。

### 5.6 _extract_table_figure 与 __filterout_scraps

- **_extract_table_figure**：根据 `page_layout` 和 TSR 结果，把表格和图表区域裁剪成图像并转为 HTML 或文本，得到 `tbls`（以及可选 figures）。
- **__filterout_scraps**：去掉过小、无意义的框，返回「正文 boxes」和「表格列表」；naive 中再通过 `_line_tag(b, zoomin)` 为每个 box 生成 position_tag，得到 `(text, position_tag)` 的 sections 和 tables。

### 5.7 naive 中的 Pdf.__call__ 与 RAGFlowPdfParser 的差异

- **naive.Pdf**：在 `__images__` 后依次调用 `_layouts_rec` → `_table_transformer_job` → `_text_merge` → `_extract_table_figure` → `_naive_vertical_merge` → `_concat_downward`，返回 `[(text, position_tag), ...], tbls`，供 `naive_merge` + `tokenize_chunks` 使用。
- **RAGFlowPdfParser.__call__**：少一次 `_naive_vertical_merge`，多 `_filter_forpages`，返回形态一致。
- **parse_into_bboxes**（用于 Flow Parser 节点）：在 `__images__` + `_layouts_rec` + `_table_transformer_job` + `_text_merge` + `_concat_downward` + `_naive_vertical_merge` 后，再 `_extract_table_figure`，并把表格/图插入到 `self.boxes` 的合适位置，为每个 box 生成 `position_tag`、`image`（crop）、`positions`，最后返回 bbox 列表（带 text、layout_type、image、positions），供 Flow 输出 JSON/Markdown。

---

## 六、Flow 中的 PDF 解析（Parser 节点）

文件：`rag/flow/parser/parser.py`。

- 当 DAG 中 Parser 的输入文件为 PDF 时，会走 `_pdf(name, blob)`。
- **解析方式** 由 `conf.get("parse_method")` 决定：
  - **deepdoc**：`RAGFlowPdfParser().parse_into_bboxes(blob, callback=self.callback)`，即上文的 bbox 管线。
  - **plain_text**：`PlainParser()(blob)`，只输出纯文本行。
  - **mineru / paddleocr**：通过 `LLMBundle(..., LLMType.OCR)` 拿到对应解析器，调用其 `parse_pdf(...)`，将返回的 `(lines)` 转成带 `image`、`positions` 的 bbox。
  - **tcadp parser**：`TCADPParser().parse_pdf(...)`，解析结果转 bbox。
  - **其他**：当作「视觉模型」名称，用 `VisionParser(vision_model=...)(blob)` 做版面+识别，再转 bbox。
- 输出格式由 `conf["output_format"]` 决定：**json**（bbox 列表）或 **markdown**（拼接成一段 Markdown）。

该路径与「知识库文档解析」共享同一套 DeepDOC/Plain/Vision 等能力，但入口是 Flow 的 `_invoke`，不经过 `naive.chunk()`。

---


## 七、关键数据结构与工具

### 7.1 position_tag 与 positions

- **position_tag**：字符串，形如 `@@{page}\t{x0}\t{x1}\t{top}\t{bottom}##`，可多个拼接，表示该 chunk 在 PDF 上的位置。
- **positions**：列表，每项 `[page, x0, x1, top, bottom]`（整数页号），由 `RAGFlowPdfParser.extract_positions(position_tag)` 解析得到。
- **crop**：`pdf_parser.crop(position_tag, ZM, need_position=False)` 根据 position_tag 从 `page_images` 裁剪出对应区域图像，用于 chunk 的 image 字段或预览。

### 7.2 表格与图表

- 表格：TSR 得到单元格级框和类型，再在 `_extract_table_figure` 中生成 HTML 或纯文本，并带位置信息插入 boxes 或单独返回 tables。
- 图表：作为 figure 区域裁剪成图，可交给视觉模型做描述（如 vision_figure_parser_pdf_wrapper），与正文 chunk 关联。

---

## 八、流程小结（按执行顺序）

| 阶段 | 位置 | 说明 |
|------|------|------|
| 上传与任务拆分 | task_service.queue_tasks | PDF 按页范围拆 Task，写入 DB 并推 Redis |
| 消费 Task | task_executor.build_chunks | 从 MinIO 取二进制，调用 naive.chunk |
| 解析器选择 | naive.chunk | layout_recognize → PARSERS[name] → parser() |
| 页转图+OCR | pdf_parser.__images__ | pdfplumber 转图+抽字，OCR detect+recognize，得到 boxes |
| 版面分析 | _layouts_rec | LayoutRecognizer 打 layout_type，得到 page_layout |
| 表格识别 | _table_transformer_job | 表格裁剪、可选旋转、TSR、必要时旋转表再 OCR |
| 文本合并与过滤 | _text_merge / _concat_downward / _filter_forpages | 列分配、段落合并、页眉页脚过滤 |
| 表格与图抽取 | _extract_table_figure | 表格/图转 HTML 或文本，带位置 |
| 输出 sections/tables | Pdf.__call__ / parse_into_bboxes | (text, position_tag) 列表 + tables |
| 分块 | naive_merge | 按 chunk_token_num、delimiter、overlap 合并成 chunks |
| Chunk 生成 | tokenize_table + tokenize_chunks | 表格 chunk + 正文 chunk（含 content、positions、image），返回给 build_chunks |
| 图片上传 | build_chunks 内 upload_to_minio | chunk 带 image 则上传 MinIO 得 img_id，并发执行 |
| 可选增强 | auto_keywords / auto_questions | 关键词抽取、问题生成，写入 important_kwd、question_kwd 等 |
| 向量化 | task_executor.embedding | 标题+正文分批 encode，标题权重 filename_embd_weight，写回 q_*_vec |
| 写入向量库 | insert_chunks → docStoreConn.insert | 索引名 ragflow_{tenant_id}，按 kb_id 分表，批量写入 ES/Infinity 等 |
| 收尾 | do_handle_task | increment_chunk_num、可选 TOC 写入、取消时按 doc_id 删除并清图 |

以上即 RAGFlow 解析 PDF 的完整流程，从存储与任务调度、解析分块、到向量化与写入向量库均包含在内。
