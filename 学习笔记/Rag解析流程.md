现在我已经了解了完整流程。让我为你总结RAGFlow的文档解析全流程：
RAGFlow 文档解析全流程
1. API入口 (api/apps/sdk/doc.py:782)
POST /datasets/<dataset_id>/chunks
- 验证用户权限
- 获取文档列表
- 调用 queue_tasks() 将任务放入Redis队列
2. 任务入队 (api/db/services/task_service.py:360)
- 根据文档类型(PDF/Excel等)创建不同任务
- PDF按页范围分片(默认每12页一个任务)
- 计算任务digest用于去重和缓存
3. 任务执行器 (rag/svr/task_executor.py)
核心处理在 do_handle_task() 函数 (第949行):
3.1 构建Chunks (build_chunks, 第242行)
chunker = FACTORY[task["parser_id"].lower()]  # naive, paper, book等
cks = await chunker.chunk(...)
- PDF解析: 使用 deepdoc/parser/pdf_parser.py 进行:
  - OCR识别 (vision/ocr.py)
  - 布局分析 (vision/layout_recognizer.py) 
  - 表格识别 (vision/table_structure_recognizer.py)
- 其他格式: docx, excel, txt, markdown 等各有专门parser
3.2 向量化 (embedding, 第569行)
token_count, vector_size = await embedding(chunks, embedding_model, ...)
- 调用Embedding模型将文本转为向量
- 支持批量处理(EMBEDDING_BATCH_SIZE)
- 标题和内容可加权组合
3.3 存储到向量数据库 (insert_chunks, 第870行)
settings.docStoreConn.insert(chunks, index_name, dataset_id)
- 支持多种向量存储: Elasticsearch / Infinity / OpenSearch / OB / MySQL
- 存储格式包含: content_with_weight, q_256_vec(向量), doc_id, page_num等
流程图
API调用 → Redis队列 → task_executor
                            ↓
                      build_chunks (解析)
                            ↓
                      embedding (向量化)
                            ↓
                      insert_chunks (存储到ES/Infinity)
如需深入了解某个具体环节(如OCR原理、表格识别、embedding模型等)，我可以继续分析。