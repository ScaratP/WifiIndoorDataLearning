# NTTU WiFi 室內定位模型改進報告

## 執行摘要

本報告分析了四種神經網路模型在 NTTU WiFi 室內定位任務上的表現，並提出改進建議。目前最佳模型 CFNN2 的最終分數為 11.501，仍有顯著的改進空間。

## 1. 現況分析

### 1.1 模型表現對比

| 模型 | 測試 MSE | 建築準確率 (%) | 樓層準確率 (%) | 最終分數 |
|------|----------|----------------|----------------|----------|
| Baseline | 9.084 | 83.87 | 70.97 | **21.400** |
| CFNN2 | 9.744 | 100.0 | 96.77 | **11.501** ⭐ |
| HADNN2 | 9.516 | 100.0 | 96.77 | 12.457 |
| HADNNh2 | **8.609** ⭐ | 100.0 | 93.55 | 12.086 |

### 1.2 關鍵發現

**優勢：**
- 所有改進模型都實現了完美的建築物分類 (100%)
- CFNN2 和 HADNN2 在樓層分類上表現優異 (96.77%)
- HADNNh2 在位置回歸上表現最佳 (MSE: 8.609)

**問題點：**
- 最終分數仍偏高（理想值應 < 5）
- 位置回歸精度有改進空間 (MSE > 8)
- 模型間表現差異明顯，缺乏一致性

## 2. 問題分析

### 2.1 資料層面問題

1. **資料品質**
   - RSS 訊號噪音影響
   - 時間變異性未考慮
   - 環境因素變化

2. **資料分布**
   - 訓練/測試集分布可能不平衡
   - 空間採樣密度不均
   - 邊界區域資料稀少

3. **特徵工程**
   - RSS 標準化方法可能不適當
   - 缺乏時序特徵
   - 空間相關特徵未充分利用

### 2.2 模型結構問題

1. **架構設計**
   - 層次結構可能過於簡單
   - 注意力機制缺失
   - 殘差連接不夠充分

2. **超參數設定**
   - 學習率調度可能不適當
   - 批次大小可能過小
   - 正則化強度需要調整

## 3. 改進建議

### 3.1 短期改進 (立即可執行)

#### A. 資料預處理優化

```python
# 新的 RSS 預處理方法
def advanced_rss_preprocessing(rss_data):
    # 1. 異常值檢測和處理
    rss_data = np.clip(rss_data, -100, -30)
    
    # 2. 信號強度加權標準化
    weights = np.exp(rss_data / 20)  # 強信號權重更高
    rss_data = rss_data * weights
    
    # 3. 移動平均平滑
    from scipy import ndimage
    rss_data = ndimage.uniform_filter1d(rss_data, size=3)
    
    return rss_data
```

#### B. 超參數調優

```python
# 建議的超參數配置
IMPROVED_CONFIG = {
    'batch_size': 64,  # 增加批次大小
    'max_lr': 0.005,   # 降低最大學習率
    'warm_epochs': 10, # 增加熱身期
    'epochs': 150,     # 增加訓練輪次
    'dropout_rate': 0.4,  # 增加 dropout
    'weight_decay': 1e-3  # 增加權重衰減
}
```

#### C. 損失函數改進

```python
def improved_loss_function(y_true, y_pred, hierarchy_weights=[1.0, 2.0, 1.5]):
    """改進的階層損失函數"""
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true[0], y_pred[0])
    
    # 加權分類損失
    total_loss = mse_loss
    for i in range(len(y_true) - 1):
        class_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
            y_true[i+1], y_pred[i+1]
        )
        total_loss += hierarchy_weights[i] * class_loss
    
    return total_loss
```

### 3.2 中期改進 (需要重新設計)

#### A. 注意力機制

```python
class AttentionHADNN(tf.keras.Model):
    def __init__(self, n_classes, n_buildings, n_floors, input_dim):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=64
        )
        # ...其他層定義
    
    def call(self, inputs):
        # RSS 注意力機制
        attended_features = self.attention(inputs, inputs)
        # ...後續處理
```

#### B. 集成學習方法

```python
class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, x):
        predictions = [model.predict(x) for model in self.models]
        # 加權平均預測結果
        weights = [0.3, 0.3, 0.2, 0.2]  # 基於驗證集表現調整
        return np.average(predictions, axis=0, weights=weights)
```

#### C. 時空特徵工程

```python
def extract_spatial_features(coordinates, k=5):
    """提取空間特徵"""
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    
    return {
        'neighbor_distances': distances,
        'spatial_density': 1.0 / (distances.mean(axis=1) + 1e-6)
    }
```

### 3.3 長期改進 (研究導向)

#### A. 深度生成模型

```python
class VariationalAutoencoder(tf.keras.Model):
    """用於 RSS 資料去噪和增強"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
    
    def build_encoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.latent_dim * 2)  # mean + log_var
        ])
```

#### B. 圖神經網路

```python
class GraphNeuralNetwork(tf.keras.Model):
    """利用空間拓撲關係的圖神經網路"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gnn_layers = [
            GraphConvolution(hidden_dim) for _ in range(3)
        ]
    
    def call(self, node_features, adjacency_matrix):
        # 圖卷積處理
        pass
```

## 4. 實施計畫

### 4.1 第一階段 (1-2 週)
- [ ] 實施資料預處理優化
- [ ] 調整超參數配置
- [ ] 改進損失函數
- [ ] 增加資料增強技術

### 4.2 第二階段 (2-4 週)
- [ ] 開發注意力機制模型
- [ ] 實施集成學習方法
- [ ] 添加空間特徵工程
- [ ] 建立交叉驗證框架

### 4.3 第三階段 (1-2 個月)
- [ ] 研究深度生成模型
- [ ] 探索圖神經網路應用
- [ ] 開發在線學習機制
- [ ] 建立模型解釋性工具

## 5. 預期改進效果

### 5.1 短期目標
- 最終分數降至 < 8.0
- 位置 MSE 降至 < 6.0
- 保持分類準確率 > 95%

### 5.2 中期目標
- 最終分數降至 < 5.0
- 位置 MSE 降至 < 4.0
- 實現實時推理 < 100ms

### 5.3 長期目標
- 最終分數降至 < 3.0
- 支援動態環境適應
- 達到商用部署標準

## 6. 風險評估

### 6.1 技術風險
- **高風險**: 深度模型可能過擬合
- **中風險**: 新架構可能不穩定
- **低風險**: 超參數調優失敗

### 6.2 資源風險
- 計算資源需求增加 2-3 倍
- 開發時間可能延長 50%
- 需要額外的驗證資料集

## 7. 結論

目前的 HADNN 模型系列已經展現了良好的基礎性能，特別是在分類任務上。通過系統性的改進，包括資料預處理優化、模型架構增強和訓練策略改進，預計可以將模型性能提升 30-50%。

建議優先實施短期改進方案，並逐步推進中長期研究目標，以實現更加精確和穩健的室內定位系統。

---

**報告作成日期**: 2025年6月
