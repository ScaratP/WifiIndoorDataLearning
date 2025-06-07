import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 將 AttentionLayer 類別定義移至檔案頂部，以便其他文件可以直接導入
class AttentionLayer(layers.Layer):
    """注意力機制層，用於識別重要的 Wi-Fi AP"""
    def __init__(self, units, **kwargs):  # 增加 **kwargs 來接受額外的參數
        super(AttentionLayer, self).__init__(**kwargs)  # 將 kwargs 傳遞給父類別
        self.units = units  # 儲存 units 作為實例變數
        self.W = None  # 初始化為 None，在 build 方法中創建
        self.V = None  # 初始化為 None，在 build 方法中創建
    
    def build(self, input_shape):
        # 在 build 方法中創建權重，而不是在 __init__ 中
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_W"
        )
        self.V = self.add_weight(
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_V"
        )
        super(AttentionLayer, self).build(input_shape)
    
    def get_config(self):
        # 為序列化和反序列化提供配置方法
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config
        
    def call(self, inputs):
        # 計算注意力權重
        score = tf.nn.tanh(tf.matmul(inputs, self.W))  # 使用矩陣乘法代替 Dense 層
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
        # 應用注意力權重
        context_vector = attention_weights * inputs
        return context_vector, attention_weights

def enhanced_hadnn_model(input_dim, n_buildings, n_floors):
    """改進的 HADNN 模型，加強建築物分類能力"""
    
    # 輸入層
    inputs = layers.Input(shape=(input_dim,))
    
    # 共享特徵提取層 (更深層網路)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # 注意力機制
    attention_output, attention_weights = AttentionLayer(128)(x)
    
    # 建築物分類分支 (加強此分支)
    building_features = layers.Dense(128, activation='relu')(attention_output)
    building_features = layers.BatchNormalization()(building_features)
    building_features = layers.Dense(64, activation='relu')(building_features)
    building_output = layers.Dense(n_buildings, activation='softmax', name='building_output')(building_features)
    
    # 樓層分類分支 (以建築物分類結果為條件)
    floor_input = layers.Concatenate()([x, building_features])
    floor_features = layers.Dense(128, activation='relu')(floor_input)
    floor_features = layers.BatchNormalization()(floor_features)
    floor_features = layers.Dense(64, activation='relu')(floor_features)
    floor_output = layers.Dense(n_floors, activation='softmax', name='floor_output')(floor_features)
    
    # 位置回歸分支 (同時以建築物和樓層為條件)
    position_input = layers.Concatenate()([x, building_features, floor_features])
    position_features = layers.Dense(128, activation='relu')(position_input)
    position_features = layers.BatchNormalization()(position_features)
    position_features = layers.Dense(64, activation='relu')(position_features)
    position_output = layers.Dense(2, name='position_output')(position_features)
    
    model = Model(inputs=inputs, outputs=[building_output, floor_output, position_output])
    
    # 使用自定義損失權重，加重建築物分類權重
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'building_output': 'sparse_categorical_crossentropy',
            'floor_output': 'sparse_categorical_crossentropy',
            'position_output': 'mse'
        },
        loss_weights={
            'building_output': 2.0,  # 加大建築物分類的權重
            'floor_output': 1.0,
            'position_output': 1.0
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'position_output': 'mse'
        }
    )
    
    return model

def advanced_data_preprocessing(rss_data):
    """進階資料預處理，包括特徵工程和異常檢測"""
    
    # 1. 替換極端值 (-100)，使用更合理的缺失值表示
    missing_mask = (rss_data == -100)
    mean_values = np.mean(rss_data[~missing_mask])
    # 使用略低於均值的數值來代替缺失值
    replacement_value = mean_values - 20  # RSS 一般是負值
    processed_data = np.copy(rss_data)
    processed_data[missing_mask] = replacement_value
    
    # 2. 計算每個 AP 的信號穩定性 (可用作權重或特徵)
    std_per_ap = np.std(processed_data, axis=0)
    stability_score = 1 / (1 + std_per_ap)  # 越低標準差越穩定
    
    # 3. 特徵歸一化 (更穩健的方法)
    # 使用穩健的最小-最大縮放，基於分位數而非極值
    q_min, q_max = np.percentile(processed_data, [5, 95], axis=0)
    # 避免除以零
    q_range = q_max - q_min
    q_range[q_range == 0] = 1
    normalized_data = (processed_data - q_min) / q_range
    
    # 4. 特徵擴展 (添加信號穩定性信息)
    # 此部分可選，看模型是否需要這額外信息
    # extended_data = np.column_stack([normalized_data, stability_score])
    
    return normalized_data, stability_score

def train_enhanced_model(hadnn_data_dir, output_model_dir):
    """訓練增強型 HADNN 模型"""
    # 載入數據
    train_x = np.load(os.path.join(hadnn_data_dir, 'train_x.npy'))
    train_b = np.load(os.path.join(hadnn_data_dir, 'train_b.npy'))
    train_c = np.load(os.path.join(hadnn_data_dir, 'train_c.npy'))
    train_y = np.load(os.path.join(hadnn_data_dir, 'train_y.npy'))
    
    test_x = np.load(os.path.join(hadnn_data_dir, 'test_x.npy'))
    test_b = np.load(os.path.join(hadnn_data_dir, 'test_b.npy'))
    test_c = np.load(os.path.join(hadnn_data_dir, 'test_c.npy'))
    test_y = np.load(os.path.join(hadnn_data_dir, 'test_y.npy'))
    
    # 檢查和修正資料形狀不一致問題
    print("檢查資料形狀...")
    
    # 修正訓練集形狀
    if train_c.shape[0] != train_x.shape[0]:
        print(f"訓練集形狀不匹配 - train_x: {train_x.shape}, train_c: {train_c.shape}")
        if train_c.shape[0] == 2 * train_x.shape[0]:
            # 可能是座標被重複存儲
            train_c = train_c[:train_x.shape[0]]
            print(f"已修正訓練集位置座標形狀: {train_c.shape}")
    
    # 修正測試集形狀
    if test_c.shape[0] != test_x.shape[0]:
        print(f"測試集形狀不匹配 - test_x: {test_x.shape}, test_c: {test_c.shape}")
        if test_c.shape[0] == 2 * test_x.shape[0]:
            # 可能是座標被重複存儲
            test_c = test_c[:test_x.shape[0]]
            print(f"已修正測試集位置座標形狀: {test_c.shape}")
    
    # 確保標籤是正確的形狀
    train_b = train_b.reshape(-1)
    train_c = train_c.reshape(-1, 2) if len(train_c.shape) == 1 and train_c.shape[0] % 2 == 0 else train_c
    train_y = train_y.reshape(-1)
    
    test_b = test_b.reshape(-1)
    test_c = test_c.reshape(-1, 2) if len(test_c.shape) == 1 and test_c.shape[0] % 2 == 0 else test_c
    test_y = test_y.reshape(-1)
    
    # 增強數據預處理
    train_x_enhanced, train_stability = advanced_data_preprocessing(train_x)
    test_x_enhanced, test_stability = advanced_data_preprocessing(test_x)
    
    # 載入配置檔案，以確保使用正確的分類數目
    import json
    config_path = os.path.join(hadnn_data_dir, 'dataset_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            n_buildings = config.get('n_buildings', len(np.unique(train_b)))
            n_floors = config.get('n_floors', len(np.unique(train_y)))  # 改用 train_y 代替 train_c
    else:
        n_buildings = len(np.unique(train_b))
        n_floors = len(np.unique(train_y))  # 改用 train_y 代替 train_c
    
    print(f"建築物類別數: {n_buildings}")
    print(f"樓層類別數: {n_floors}")
    print(f"輸入特徵維度: {train_x.shape[1]}")
    print(f"訓練資料形狀: {train_x.shape}")
    print(f"測試資料形狀: {test_x.shape}")
    
    # 確認標籤形狀
    print(f"建築物標籤形狀: {train_b.shape}")
    print(f"位置標籤形狀: {train_c.shape}")  # train_c 是位置座標
    print(f"樓層標籤形狀: {train_y.shape}")  # train_y 是樓層標籤
    
    # 創建增強模型
    model = enhanced_hadnn_model(train_x.shape[1], n_buildings, n_floors)
    
    # 檢查模型摘要
    model.summary()
    
    # 設置回調函數
    callbacks = [
        EarlyStopping(
            monitor='val_building_output_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1  # 添加 verbose=1 以輸出早停資訊
        ),
        ReduceLROnPlateau(
            monitor='val_building_output_accuracy',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # 確保標籤是正確的形狀
    train_b = train_b.reshape(-1)
    train_y = train_y.reshape(-1)  # 使用 train_y 作為樓層標籤
    test_b = test_b.reshape(-1)
    test_y = test_y.reshape(-1)  # 使用 test_y 作為樓層標籤
    
    # 訓練模型 (使用更多 epochs)
    history = model.fit(
        train_x_enhanced,
        {
            'building_output': train_b,
            'floor_output': train_y,  # 使用 train_y 作為樓層標籤
            'position_output': train_c  # train_c 是位置座標
        },
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 儲存訓練歷史以便後續分析
    import pickle
    with open(os.path.join(output_model_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    # 顯示最終訓練情況
    if 'val_building_output_accuracy' in history.history:
        max_acc = max(history.history['val_building_output_accuracy'])
        print(f"最高建築物分類驗證準確度: {max_acc:.4f}")
        if max_acc >= 0.99:
            print("建築物分類驗證準確度接近或達到100%，這可能是訓練提前結束的原因")
    
    # 評估模型
    results = model.evaluate(
        test_x_enhanced,
        {
            'building_output': test_b,
            'floor_output': test_y,  # 使用 test_y 作為樓層標籤
            'position_output': test_c  # test_c 是位置座標
        }
    )
    
    # 儲存模型
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    model.save(os.path.join(output_model_dir, 'enhanced_hadnn_model.h5'))
    
    # 儲存模型評估結果
    with open(os.path.join(output_model_dir, 'model_results.txt'), 'w') as f:
        f.write("模型評估結果:\n")
        f.write(f"總損失: {results[0]:.4f}\n")
        f.write(f"建築物分類損失: {results[1]:.4f}\n")
        f.write(f"建築物分類準確率: {results[4]:.2%}\n")
        f.write(f"樓層分類損失: {results[2]:.4f}\n")
        f.write(f"樓層分類準確率: {results[5]:.2%}\n")
        f.write(f"位置預測 MSE: {results[3]:.4f}\n")
    
    print("模型訓練和評估完成")
    return model, history

def analyze_building_misclassifications(model, test_x, test_b, test_names=None):
    """分析建築物分類錯誤的情況"""
    # 預測建築物分類
    predictions = model.predict(test_x)
    building_preds = np.argmax(predictions[0], axis=1)
    
    # 找出錯誤分類的實例
    misclassified_indices = np.where(building_preds != test_b)[0]
    
    print(f"建築物分類錯誤的樣本數: {len(misclassified_indices)}/{len(test_b)} ({len(misclassified_indices)/len(test_b):.2%})")
    
    # 如果有測試點名稱，則輸出這些點的信息
    if test_names is not None:
        for idx in misclassified_indices:
            print(f"錯誤樣本 {idx}: {test_names[idx]}")
            print(f"  真實建築物: {test_b[idx]}, 預測建築物: {building_preds[idx]}")
    
    return misclassified_indices, building_preds

if __name__ == "__main__":
    print("=== 開始增強型 HADNN 模型訓練 ===")
    hadnn_data_dir = "./hadnn_data"  # 修正路徑
    output_model_dir = "./enhanced_models"  # 修正路徑
    
    model, history = train_enhanced_model(hadnn_data_dir, output_model_dir)
    
    print("模型訓練完成。模型保存在:", output_model_dir)
    print("請使用 evaluate_model.py 進行詳細評估")
