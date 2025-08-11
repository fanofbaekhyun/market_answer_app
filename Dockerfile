FROM nvidia/cuda:12.1.105-cudnn8-devel-ubuntu22.04

WORKDIR /app

# 安装 Python
RUN apt-get update && apt-get install -y python3 python3-pip git

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY app ./app

# 暴露端口
EXPOSE 8000

# 启动 FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
