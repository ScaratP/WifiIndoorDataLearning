FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /workspace

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 創建必要的目錄結構
RUN mkdir -p models/building_classifier \
    models/floor_classifiers \
    models/position_predictors \
    data/processed \
    logs

# 設置環境變數
ENV PYTHONPATH=/workspace
ENV TF_CPP_MIN_LOG_LEVEL=2

# 複製源代碼
COPY src/ ./src/

CMD ["bash"]
