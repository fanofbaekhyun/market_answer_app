from app.utils import extract_elements_with_qwen, faiss_search, attribute_filter, bm25_search, merge_dedup, rerank, generate_strategy_with_qwen

def rag_pipeline(user_query):
    # 1. LLM 提取四要素
    elements = extract_elements_with_qwen(user_query)
    
    # 2. 向量检索
    vector_results = faiss_search(elements)
    
    # 3. 属性过滤 & BM25
    filtered_results = attribute_filter(vector_results, elements)
    bm25_results = bm25_search(user_query)
    
    # 4. 合并去重
    merged = merge_dedup(filtered_results, bm25_results)
    
    # 5. Cross-Encoder 精排
    top5 = rerank(user_query, merged)
    
    # 6. LLM 生成活动策略
    return generate_strategy_with_qwen(elements, top5)
