import os
import numpy as np
import faiss
from app.model_loader import get_simcse_model

INDEX_PATH = "/app/models/faiss_index.index"

def build_faiss_index():
    if os.path.exists(INDEX_PATH):
        print("FAISS index already exists.")
        return
    
    print("Building FAISS index...")
    model = get_simcse_model()

    # 示例数据向量（实际应替换成历史活动文本 embedding）
    dummy_vectors = np.random.rand(100, 768).astype("float32")

    d = 768
    m = 16
    nbits = 8
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, 10, m, nbits)
    index.train(dummy_vectors)
    index.add(dummy_vectors)
    faiss.write_index(index, INDEX_PATH)
    print("FAISS index built successfully.")
