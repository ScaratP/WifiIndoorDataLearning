# 專案修改日誌與 TFLite 部署指南

**日期：** 2025年10月28日

**目標：** 將 Python (`.h5`) 訓練的模型，成功轉換為 `.tflite` 格式，並解決其在 Android 平台上無法正確預測位置的問題。

---

## 1. 總結：核心問題與最終架構

我們在過程中發現了導致 Android 部署失敗的三個主要問題：

1.  **TFLite 相容性問題：** 原始模型 (`original_hadnn.py`, `mlp.py`) 使用了 TFLite 轉換器無法辨識的自定義 `AttentionLayer` 類別。
2.  **資料前處理不一致：** Python (`hadnn_adapter.py`) 使用的 RSSI 訊號縮放方式 (StandardScaler, `(v-mean)/std`) 與 Android 端實作的 (Normalization, `[0, 1]` 範圍) **完全不同**。
3.  **【最關鍵的邏輯錯誤】座標系統假設錯誤：**
    * 我們最後發現，座標 (x, y) 並非「全域」座標，而是基於**「不同地圖」的「像素百分比」**。
    * 這導致我們之前訓練的「單一模型」(`original_hadnn.tflite`, `mlp.tflite`)，其座標輸出 (`coord_output`) 是**無效**的，因為它試圖在同一個模型裡學習不同地圖上相同百分比 (x, y) 點的*不同* WiFi 指紋。
    * 這也解釋了為什麼您的 `.h5` 測試看似正常（因為分類準確率很高），但在 Android 上座標是混亂的。

---

## 2. 已執行的關鍵更動

為了修正上述問題，對 `251008` 資料夾 進行了以下重大修改：

### A. 針對 TFLite 相容性的模型修改

* **目標：** 讓 TFLite 轉換器能看懂模型架構。
* **檔案：** `original_hadnn.py`, `mlp.py`
* **更動：**
    * **移除 `AttentionLayer`**：完全刪除了 `class AttentionLayer(layers.Layer)` 的自定義程式碼。
    * **替換為 `MultiHeadAttention`**：在 `build_..._model` 函數中，我們改用 TensorFlow 官方支援、TFLite 相容的 `tf.keras.layers.MultiHeadAttention` 來取代它。
    * **啟用 TF Ops**：在 `TFLiteConverter` 中加入了 `converter.target_spec.supported_ops = [..., tf.lite.OpsSet.SELECT_TF_OPS]`，以確保 TFLite 能使用 `MultiHeadAttention` 需要的操作。

### B. 針對資料一致性的預處理修改

* **目標：** 確保 Python 訓練時的資料格式，與 Android App 輸入的格式 100% 相同。
* **檔案：** `preprocess/hadnn_adapter.py`
* **更動：**
    * **統一 RSSI 縮放邏輯**：修改了 `normalize_rss` 函數。
    * **捨棄 `StandardScaler`**：不再使用 `(v-mean)/std` 的標準化。
    * **採用 `[0, 1]` 縮放**：改為使用與您 Android App 一致的 `np.clip(rss, -100.0, 0.0)`，接著 `(clipped + 100.0) / 100.0`，將所有 RSSI 值縮放到 `[0, 1]` 區間。

### C. 針對訓練效能的環境與參數修改

* **目標：** 解決模型過於複雜導致 CPU 訓練過慢，以及 GPU 記憶體不足 (OOM) 的問題。
* **檔案：** `original_hadnn.py`, `mlp.py`
* **環境設定 (本地 Windows)：**
    * **安裝 Python 3.10.11**：因為 `tensorflow==2.10` 不支援 Python 3.11+。
    * **安裝 CUDA 11.2**：安裝 `tensorflow==2.10` 指定的 CUDA 版本。
    * **安裝 cuDNN 8.1**：手動複製 `cuDNN 8.1` 相關的 `bin`, `include`, `lib` 檔案至 CUDA 11.2 資料夾。
    * **降級 NumPy**：解決 `tensorflow==2.10` 與 `NumPy 2.x` 的相容性衝突，降級為 `pip install "numpy<2.0"`。
* **訓練參數 (程式碼內)：**
    * **降低 `batch_size`**：在 `model.fit(...)` 中，將 `batch_size=128` 降低為 `batch_size=32`，以解決 T400 4GB 顯卡 的 `ResourceExhaustedError` (OOM) 問題。
