from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import faiss
import json

# ====== LLM 提取商家问题四要素 ======
def extract_elements_with_qwen(query):
    # 调用 Qwen 模型做信息抽取
    return {
        "activity_type": "满减",
        "target_effect": "提升客单价",
        "customer_segment": "新用户",
        "scenario": "双十一大促"
    }

# ====== FAISS 向量检索 ======
def faiss_search(elements):
    # 根据 elements 查询向量索引
    return ["案例1", "案例2", "案例3"]

# ====== 属性过滤 ======
def attribute_filter(results, elements):
    return results

# ====== BM25 检索 ======
def bm25_search(query):
    return ["文本案例4", "文本案例5"]

# ====== 合并去重 ======
def merge_dedup(list1, list2):
    return list(set(list1 + list2))

# ====== Cross-Encoder 精排 ======
def rerank(query, candidates):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = cross_encoder.predict([(query, doc) for doc in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:5]]

# ====== LLM 生成策略 ======
def generate_strategy_with_qwen(elements, history_cases):
    prompt = f"""
    你是商家活动决策助手。
    活动类型: {elements['activity_type']}
    目标效果: {elements['target_effect']}
    客群: {elements['customer_segment']}
    场景: {elements['scenario']}
    历史活动案例:
    {json.dumps(history_cases, ensure_ascii=False, indent=2)}
    """
    return f"[生成策略] {prompt}"
