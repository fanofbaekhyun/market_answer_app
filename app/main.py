from fastapi import FastAPI, Query
from app.rag_pipeline import rag_pipeline
from app.model_loader import download_models
from app.index_builder import build_faiss_index

app = FastAPI(title="Merchant Activity Decision Assistant")

@app.on_event("startup")
def startup_event():
    download_models()
    build_faiss_index()

@app.get("/generate_strategy")
async def generate_strategy(query: str = Query(..., description="商家需求描述")):
    strategy = rag_pipeline(query)
    return {"strategy": strategy}

@app.get("/health")
async def health():
    return {"status": "ok"}
