# WiFi 室內定位深度學習系統250713

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

基於深度學習的 WiFi 室內定位系統，使用 HADNN (Hierarchical Attention Deep Neural Network) 架構實現建築物分類、樓層分類和精確位置預測。

## 📋 目錄

- [專案概述](#專案概述)
- [系統架構](#系統架構)
- [資料集](#資料集)
- [模型架構](#模型架構)
- [安裝與環境設定](#安裝與環境設定)
- [使用指南](#使用指南)
- [檔案結構](#檔案結構)
- [模型評估](#模型評估)
- [結果分析](#結果分析)
- [視覺化圖表說明](#視覺化圖表說明)
- [故障排除](#故障排除)
- [參考資料](#參考資料)

## 🎯 專案概述

本專案實現了一個完整的 WiFi 室內定位系統，主要功能包括：

### 核心功能
- **建築物分類**：識別使用者所在的建築物 (SEA, SEB, SEC)
- **樓層分類**：確定使用者所在的樓層 (1-5樓)
- **精確定位**：預測使用者的 (x, y) 座標位置
- **點位分類**：識別特定房間或走廊位置

### 主要特色
- 🏗️ **階層式架構**：先建築物 → 樓層 → 精確位置的邏輯順序
- 🎯 **注意力機制**：自動識別對定位最重要的 WiFi AP
- 📊 **多模型比較**：支援 MLP、HADNN、整合模型的性能比較
- 🔍 **全面評估**：提供詳細的模型評估報告和視覺化分析

## 🏗️ 系統架構

```
WiFi 信號數據 → 資料預處理 → 特徵工程 → 深度學習模型 → 定位結果
     ↓              ↓           ↓            ↓           ↓
  掃描數據        標準化      注意力機制     分類/回歸    建築物/樓層/位置
```

### 模型架構圖

```
輸入層 (RSS 向量)
      ↓
共享特徵提取層 (Dense + BatchNorm + Dropout)
      ↓
注意力機制 (AttentionLayer)
      ↓
   ┌─────────────────┬─────────────────┬─────────────────┐
   ↓                 ↓                 ↓                 ↓
建築物分類分支     樓層分類分支      位置預測分支       點位分類分支
   ↓                 ↓                 ↓                 ↓
 建築物預測        樓層預測          (x,y) 座標        點位 ID 預測
```

## 📊 資料集

### 資料來源
- **建築物**：台東大學知本校區理工學院 (SEA, SEB, SEC)
- **樓層**：1-5樓
- **採集點**：房間 + 走廊位置
- **Wi-Fi AP**：校園網路 (ap-nttu, ap2-nttu, eduroam, nttu)

### 資料格式
- **RSS 向量**：各 WiFi AP 的信號強度
- **位置標籤**：(x, y) 座標
- **分類標籤**：建築物 ID、樓層 ID、點位 ID

### 資料統計
```
總參考點數：[自動計算]
建築物數：3 (SEA, SEB, SEC)
樓層數：5 (1-5樓)
WiFi AP數：[自動計算]
```

## 🧠 模型架構

### 1. MLP 模型 (`mlp.py`)
```python
輸入層 → 共享特徵層 → 3個分支 (建築物/樓層/位置)
特色：簡潔高效，適合快速原型開發
```

### 2. HADNN 模型 (`original_hadnn.py`)
```python
輸入層 → 深度特徵提取 → 注意力機制 → 階層式分類 → 條件依賴輸出
特色：階層式設計，條件依賴，注意力機制
```

### 3. 整合模型 (`hadnn_n_random_forest.py`)
```python
輸入層 → 共享特徵 → 注意力機制 → 多任務學習 → 聯合優化
特色：多任務共享特徵，注意力機制，端到端訓練
```

## 🔧 安裝與環境設定

### 環境需求
```bash
Python >= 3.8
TensorFlow >= 2.8
NumPy >= 1.21
Pandas >= 1.3
Scikit-learn >= 1.0
Matplotlib >= 3.5
```

### 安裝步驟
```bash
# 1. 複製專案
git clone [repository-url]
cd WifiIndoorDataLearning/250713

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 準備資料
mkdir -p points/scan13
# 將 WiFi 掃描數據放入 points/scan13/ 目錄

# 4. 執行資料預處理
cd preprocess
python run_preprocessing.py
```

## 📖 使用指南

### 快速開始

#### 1. 資料預處理
```bash
cd preprocess
python run_preprocessing.py
```

#### 2. 模型訓練
```bash
# 訓練 MLP 模型
python mlp.py

# 訓練 HADNN 模型
python original_hadnn.py

# 訓練整合模型
python hadnn_n_random_forest.py
```

#### 3. 模型評估
```bash
# 單一模型評估
python evaluate_model.py

# 多模型比較
python compare_all_models.py

# 統一評估
python evaluate_models.py
```

### 進階使用

#### 自訂模型參數
```python
# 修改 mlp.py 中的模型參數
model = mlp_model(
    input_dim=rss_features,
    n_buildings=3,
    n_floors=5,
    hidden_units=[128, 64],  # 可調整
    dropout_rate=0.3         # 可調整
)
```

#### 調整訓練參數
```python
# 修改訓練參數
model.fit(
    train_x, train_y,
    epochs=100,        # 可調整
    batch_size=32,     # 可調整
    validation_split=0.2
)
```

## 📁 檔案結構

```
250713/
├── preprocess/                 # 資料預處理模組
│   ├── run_preprocessing.py   # 主要預處理流程
│   ├── extract_and_filter.py  # 資料提取與過濾
│   ├── create_vector_format.py # 向量格式轉換
│   └── hadnn_adapter.py       # HADNN 格式適配器
├── models/                    # 訓練好的模型
│   ├── mlp.h5                # MLP 模型
│   ├── original_hadnn.h5     # HADNN 模型
│   ├── hadnn_n_random_forest.h5 # 整合模型
│   ├── mlp.txt               # MLP 評估報告
│   ├── original_hadnn.txt    # HADNN 評估報告
│   └── hadnn_n_random_forest.txt # 整合模型評估報告
├── results/                   # 評估結果
│   ├── model_comparison.png   # 模型比較圖
│   ├── evaluation_results.json # 評估結果
│   └── model_comparison.md    # 比較報告
├── processed_data/            # 處理後的資料
│   ├── nttu_wifi_data.csv    # 主要資料集
│   ├── bssid_mapping.csv     # WiFi AP 映射
│   └── label_mappings.json   # 標籤映射
├── hadnn_data/               # HADNN 格式資料
│   ├── train_x.npy          # 訓練特徵
│   ├── train_b.npy          # 訓練建築物標籤
│   ├── train_y.npy          # 訓練樓層標籤
│   ├── train_c.npy          # 訓練位置標籤
│   ├── test_*.npy           # 測試資料
│   └── dataset_config.json  # 資料集配置
├── mlp.py                   # MLP 模型實現
├── original_hadnn.py        # HADNN 模型實現
├── hadnn_n_random_forest.py # 整合模型實現
├── evaluate_model.py        # 單一模型評估
├── evaluate_models.py       # 統一模型評估
├── compare_all_models.py    # 多模型比較
└── README.md               # 本文件
```

## 📊 模型評估

### 評估指標

#### 分類指標
- **建築物分類準確率**：建築物識別的準確度
- **樓層分類準確率**：樓層識別的準確度
- **混淆矩陣**：詳細的分類結果分析

#### 回歸指標
- **平均定位誤差**：位置預測的平均歐氏距離誤差
- **中位數誤差**：位置預測的中位數誤差
- **誤差分佈**：不同誤差範圍的樣本分佈

#### 綜合指標
- **綜合得分**：結合分類和回歸性能的綜合評分
- **模型評級**：優秀/良好/一般/需改進

### 評估報告

每個模型訓練後會自動生成評估報告：
- `models/mlp.txt` - MLP 模型評估報告
- `models/original_hadnn.txt` - HADNN 模型評估報告
- `models/hadnn_n_random_forest.txt` - 整合模型評估報告

### 視覺化結果

系統會自動生成以下視覺化結果：
- 建築物分類混淆矩陣
- 樓層分類混淆矩陣
- 位置預測誤差分佈圖
- 模型性能比較圖

## 📈 結果分析

### 典型性能表現

| 模型類型 | 建築物準確率 | 樓層準確率 | 整體平均定位誤差 | 條件平均定位誤差* |
|---------|-------------|-----------|-----------------|------------------|
| MLP     | 85-95%      | 80-90%    | 2-4 公尺        | 1.5-3 公尺       |
| HADNN   | 90-98%      | 85-95%    | 1.5-3 公尺      | 1-2.5 公尺       |
| 整合模型 | 92-99%      | 88-96%    | 1.2-2.5 公尺    | 0.8-2 公尺       |

*條件平均定位誤差：只計算建築物和樓層都預測正確的樣本的位置誤差

### 位置誤差說明

本系統提供兩種位置誤差計算方式：

1. **整體位置誤差**：對所有測試樣本計算位置預測誤差
2. **條件位置誤差**：只對建築物和樓層都預測正確的樣本計算位置誤差

條件位置誤差更能反映在正確建築物和樓層判斷基礎上的精確定位能力，通常會比整體誤差更小。

### 性能分析

#### 建築物分類
- **最佳表現**：HADNN 和整合模型通常達到 95% 以上準確率
- **影響因素**：建築物間的 WiFi 信號差異、AP 覆蓋範圍

#### 樓層分類
- **挑戰**：同建築物內樓層間信號相似度高
- **解決方案**：階層式架構，以建築物分類為條件

#### 位置預測
- **精度**：多數模型可達到 2-3 公尺的平均誤差
- **影響因素**：WiFi 信號強度變化、遮蔽效應

## 📊 視覺化圖表說明

系統會自動生成詳細的視覺化分析圖表，幫助您深入了解模型性能：

### 圖表類型

#### 🔹 基礎比較圖表
- **分類準確度對比圖**：橫向比較各模型的建築物和樓層分類準確率
- **位置誤差對比圖**：顯示各模型的平均位置預測誤差

#### 🔹 誤差分布分析
- **箱型圖**：展示誤差的統計分布，包括中位數、四分位數和異常值
- **小提琴圖**：結合密度分布，顯示誤差分布的詳細形態
- **累積分布函數（CDF）**：顯示達到特定誤差閾值的樣本百分比
- **詳細分布直方圖**：每個模型的誤差頻率分布和統計摘要

#### 🔹 穩健性測試圖表
- **跨情境比較圖**：展示模型在不同干擾條件下的性能變化
- **穩健性評分圖**：綜合評估各模型的抗干擾能力

### 如何解讀圖表

#### 📦 箱型圖解讀
```
     異常值 ○
        |
     ┌──┴──┐  ← 上觸鬚（Q3 + 1.5×IQR）
     │  ×  │  ← 平均值（×標記）
     ├──■──┤  ← 中位數（中線）
     │     │  ← 箱體（Q1到Q3）
     └──┬──┘  ← 下觸鬚（Q1 - 1.5×IQR）
        |
     異常值 ○
```
- **箱體越窄**：誤差越集中，模型越穩定
- **異常值越少**：預測結果越可靠
- **中位數接近平均值**：分布越對稱

#### 📉 CDF 圖解讀
- **曲線越陡峭**：誤差分布越集中
- **左上角的曲線**：表示低誤差樣本比例高
- **關鍵閾值**：
  - 1 公尺：室內導航精度要求
  - 2 公尺：一般定位應用需求  
  - 3 公尺：基本定位服務標準

### 圖表查看建議

1. **快速評估**：先看分類準確度和位置誤差對比圖
2. **穩定性分析**：查看箱型圖了解模型穩定性
3. **精度分析**：通過 CDF 圖評估在不同精度要求下的表現
4. **穩健性評估**：查看跨情境圖表了解抗干擾能力
5. **最終決策**：結合穩健性評分做出模型選擇

### 異常情況識別

⚠️ **需要注意的情況**：
- CDF 曲線過於平緩 → 誤差分布過於分散
- 箱型圖異常值過多 → 預測不穩定  
- 穩健性評分急劇下降 → 對干擾敏感
- 多峰分布 → 可能存在系統性偏差

## 🔧 故障排除

### 常見問題

#### 1. 資料預處理錯誤
```
錯誤：找不到掃描數據檔案
解決：確保 points/scan13/ 目錄存在且包含 JSON 檔案
```

#### 2. 模型訓練記憶體不足
```
錯誤：OutOfMemoryError
解決：減少 batch_size 或使用 GPU 訓練
```

#### 3. 模型載入失敗
```
錯誤：找不到 AttentionLayer
解決：確保從正確的模組導入自定義層
```

#### 4. 評估結果異常
```
錯誤：準確率過低
解決：檢查資料預處理、模型參數、訓練輪數
```

### 除錯技巧

#### 檢查資料形狀
```python
print(f"訓練資料形狀: {train_x.shape}")
print(f"標籤形狀: {train_b.shape}, {train_y.shape}, {train_c.shape}")
```

#### 監控訓練過程
```python
# 添加詳細的回調函數
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5')
]
```

## 🚀 未來改進方向

### 短期目標
- [ ] 增加更多建築物和樓層的支援
- [ ] 優化注意力機制的可解釋性
- [ ] 實現即時定位功能

### 長期目標
- [ ] 支援多校區的跨域定位
- [ ] 整合其他感測器資料 (藍牙、UWB)
- [ ] 開發移動端應用程式

## 🤝 貢獻指南

歡迎提交 Pull Request 或 Issue！

### 開發流程
1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

本專案使用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 📚 參考資料

1. Deep Learning for Indoor Localization: A Survey
2. Attention Mechanism in Deep Learning
3. WiFi-based Indoor Positioning Systems
4. TensorFlow Documentation
5. Keras API Reference

---

## 🔗 相關連結

- [TensorFlow 官方文檔](https://tensorflow.org)
- [Keras 指南](https://keras.io)
- [專案 GitHub](https://github.com/ScaratP/WifiIndoorDataLearning)

---

**最後更新日期**：2025年7月

**版本**：v1.0.0