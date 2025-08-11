import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder
import faiss

MODELS_DIR = "/app/models"

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Qwen-2.5 7B Instruct
    qwen_path = os.path.join(MODELS_DIR, "qwen2.5-7b")
    if not os.path.exists(qwen_path):
        print("Downloading Qwen-2.5-7B-Instruct...")
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=qwen_path)
        AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", torch_dtype="auto", cache_dir=qwen_path)

    # SimCSE-RoBERTa-zh
    simcse_path = os.path.join(MODELS_DIR, "simcse")
    if not os.path.exists(simcse_path):
        print("Downloading SimCSE-RoBERTa-zh...")
        AutoTokenizer.from_pretrained("shibing624/text2vec-base-chinese", cache_dir=simcse_path)
        AutoModel.from_pretrained("shibing624/text2vec-base-chinese", cache_dir=simcse_path)

    # Cross-Encoder
    cross_path = os.path.join(MODELS_DIR, "cross_encoder")
    if not os.path.exists(cross_path):
        print("Downloading Cross-Encoder...")
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", cache_dir=cross_path)

def get_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "qwen2.5-7b"))

def get_qwen_model():
    return AutoModel.from_pretrained(os.path.join(MODELS_DIR, "qwen2.5-7b"), device_map="auto", torch_dtype="auto")

def get_simcse_model():
    return AutoModel.from_pretrained(os.path.join(MODELS_DIR, "simcse"))

def get_cross_encoder():
    return CrossEncoder(os.path.join(MODELS_DIR, "cross_encoder"))
