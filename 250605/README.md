# HADNN 模型拆分與重構專案 - 250605

## 專案概述
本專案從現有的 HADNN2/HADNNh2 模型中拆分出專門的建築物/樓層分類模型，並為每個建築樓層組合訓練獨立的位置預測模型。

## 模型架構
1. **建築物分類器** - 從 WiFi 訊號預測建築物
2. **樓層分類器** - 根據建築物和 WiFi 訊號預測樓層  
3. **位置預測器** - 針對每個建築樓層組合的獨立模型

## 數據處理
- 只使用指定的 WiFi 網路：Ap-nttu, Ap2-nttu, eduroam, Guest, NTTU
- 從 scan13 資料夾的 JSON 文件中提取訓練數據
- 按建築物和樓層分組訓練獨立模型

## 文件結構
```
250605/
├── README.md
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data_processor.py
│   ├── model_splitter.py  
│   ├── building_classifier.py
│   ├── floor_classifier.py
│   ├── position_predictor.py
│   └── train_pipeline.py
├── models/
│   ├── building_classifier/
│   ├── floor_classifiers/
│   └── position_predictors/
└── data/
    └── processed/
```

## 使用方法
1. 建立 Docker 環境：`docker build -t hadnn-split .`
2. 執行訓練：`docker run -v $(pwd):/workspace hadnn-split python src/train_pipeline.py`
