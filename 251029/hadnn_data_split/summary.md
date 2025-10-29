# 資料預處理摘要 (2025-10-29)

## 1. 原始資料提取 (步驟 1/5)
- **來源**: `C:\Users\nttucsie\Desktop\ap_data\WifiIndoorDataLearning\251029\preprocess\../points/scan13`
- **處理檔案數**: 47 個
- **提取有效參考點數**: 2588 個

## 2. 向量格式轉換 (步驟 2/5)
- **BSSID 數量 (RSSI 維度)**: 526 個

## 3. 標籤映射建立 (步驟 3/5)
- **建築物**: ['sea', 'seb', 'sec']
- **樓層**: [1, 2, 3, 4, 5]
- **獨特點位 (原始)**: 201 個

## 4. CSV 儲存 (步驟 4/5)
- **輸出位置**: `C:\Users\nttucsie\Desktop\ap_data\WifiIndoorDataLearning\251029\preprocess\../processed_data`
- **總參考點數**: 2588 個
- **BSSID 數量**: 526 個
- **建築物 (CSV 中)**: ['sec' 'seb' 'sea']
- **樓層 (CSV 中)**: [1, 2, 3, 4, 5] (整數型態)
- **點位數量 (CSV 中)**: 201
- **原始座標範圍**: x=[0.00, 98.32], y=[0.00, 95.93]
- **標籤映射檔**: `../processed_data/label_mappings.json`

## 5. 1+N 模型資料準備 (步驟 5/5)
- **輸出位置**: `C:\Users\nttucsie\Desktop\ap_data\WifiIndoorDataLearning\251029\preprocess\../hadnn_data_split`
- **群組定義規則**: 合併所有建築的 1-3 樓為 `se1`, `se2`, `se3`；`sea4`, `sea5`, `seb4`, `sec4`, `sec5` 保持獨立。
- **最終群組分布**:
    - se3: 806
    - se2: 564
    - se1: 512
    - sec4: 301
    - seb4: 151
    - sec5: 140
    - sea5: 61
    - sea4: 53
- **保留用於訓練總筆數**: 2588

### 分類器 (模型 1)
- **有效群組**: 8 個 (`['se1', 'se2', 'se3', 'sea4', 'sea5', 'seb4', 'sec4', 'sec5']`)
- **RSSI 維度**: (2588, 526)
- **RSSI 標準化**: Clipping [-100, 0] -> Scaling [0, 1]
- **訓練/測試分割**: 2070 / 518 (分層抽樣)
- **標籤 ID 映射**: `{'se1': 0, 'se2': 1, 'se3': 2, 'sea4': 3, 'sea5': 4, 'seb4': 5, 'sec4': 6, 'sec5': 7}`
- **輸出檔案**:
    - `train_x_classifier.npy`, `train_y_classifier.npy`
    - `test_x_classifier.npy`, `test_y_classifier.npy`
    - `classifier_config.json`

### 回歸器 (模型 2-9)
- **座標標準化方法**: `StandardScaler` (對每個群組獨立 fit_transform, mean=0, std=1)
- **輸出設定檔**: `coord_scaler_config.json` (包含各群組的 mean_x, std_x, mean_y, std_y)

- **群組: se1**
    - 總筆數: 512
    - 座標標準化 Mean/Std (近似): [0./1.], [0./1.]
    - 訓練/測試分割: 409 / 103
    - 輸出檔案: `train_x_se1.npy`, `train_c_se1.npy`, `test_x_se1.npy`, `test_c_se1.npy`, `test_c_original_se1.npy`
- **群組: se2**
    - 總筆數: 564
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 451 / 113
    - 輸出檔案: `train_x_se2.npy`, ..., `test_c_original_se2.npy`
- **群組: se3**
    - 總筆數: 806
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 644 / 162
    - 輸出檔案: `train_x_se3.npy`, ..., `test_c_original_se3.npy`
- **群組: sea4**
    - 總筆數: 53
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 42 / 11
    - 輸出檔案: `train_x_sea4.npy`, ..., `test_c_original_sea4.npy`
- **群組: sea5**
    - 總筆數: 61
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 48 / 13
    - 輸出檔案: `train_x_sea5.npy`, ..., `test_c_original_sea5.npy`
- **群組: seb4**
    - 總筆數: 151
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 120 / 31
    - 輸出檔案: `train_x_seb4.npy`, ..., `test_c_original_seb4.npy`
- **群組: sec4**
    - 總筆數: 301
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 240 / 61
    - 輸出檔案: `train_x_sec4.npy`, ..., `test_c_original_sec4.npy`
- **群組: sec5**
    - 總筆數: 140
    - 座標標準化 Mean/Std (近似): [~0/~1], [~0/~1]
    - 訓練/測試分割: 112 / 28
    - 輸出檔案: `train_x_sec5.npy`, ..., `test_c_original_sec5.npy`