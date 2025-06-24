import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json

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

def hybrid_model(input_dim, n_buildings, n_floors):
    """整合式深度學習模型架構，使用小型MLP取代隨機森林
    同時處理建築物分類、樓層分類和位置預測"""
    
    # 輸入層
    inputs = layers.Input(shape=(input_dim,))
    
    # 簡化共享特徵提取層 (更精簡的網路)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # 注意力機制
    attention_output, attention_weights = AttentionLayer(64)(x)
    
    # 建築物分類分支
    building_features = layers.Dense(64, activation='relu')(attention_output)
    building_features = layers.Dropout(0.2)(building_features)
    building_output = layers.Dense(n_buildings, activation='softmax', name='building_output')(building_features)
    
    # 樓層分類分支 (新增，取代隨機森林)
    floor_features = layers.Dense(64, activation='relu')(attention_output)
    floor_features = layers.Dropout(0.2)(floor_features)
    floor_output = layers.Dense(n_floors, activation='softmax', name='floor_output')(floor_features)
    
    # 位置預測分支 (新增，取代隨機森林)
    position_features = layers.Dense(64, activation='relu')(attention_output)
    position_features = layers.Dropout(0.2)(position_features)
    position_output = layers.Dense(2, name='position_output')(position_features)  # 輸出x,y座標
    
    # 創建整合模型
    model = Model(inputs=inputs, outputs=[building_output, floor_output, position_output])
    
    # 編譯模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'building_output': 'sparse_categorical_crossentropy',
            'floor_output': 'sparse_categorical_crossentropy',
            'position_output': 'mse'
        },
        loss_weights={
            'building_output': 1.0,
            'floor_output': 1.0,
            'position_output': 0.5
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy'
        }
    )
    
    return model

# 自訂回調函數，監控建築判斷準確度
class BuildingAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=1.0, patience=5):
        super(BuildingAccuracyCallback, self).__init__()
        self.threshold = threshold
        self.patience = patience
        self.consecutive_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        accuracy = logs.get('val_building_output_accuracy', 0)
        
        if accuracy >= self.threshold:
            self.consecutive_count += 1
            print(f"\n建築判斷準確度達到 {accuracy:.4f}，已連續 {self.consecutive_count} 次")
            
            if self.consecutive_count >= self.patience:
                print(f"\n建築判斷準確度連續 {self.patience} 次達到 {self.threshold}，提前停止訓練")
                self.model.stop_training = True
        else:
            self.consecutive_count = 0

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
    
    return normalized_data, stability_score

def train_enhanced_model(hadnn_data_dir, output_model_dir="./models"):
    """訓練整合式深度學習模型，取代原先的混合模型（深度學習 + 隨機森林）"""
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
    config_path = os.path.join(hadnn_data_dir, 'dataset_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            n_buildings = config.get('n_buildings', len(np.unique(train_b)))
            n_floors = config.get('n_floors', len(np.unique(train_y)))
    else:
        n_buildings = len(np.unique(train_b))
        n_floors = len(np.unique(train_y))
    
    print(f"建築物類別數: {n_buildings}")
    print(f"樓層類別數: {n_floors}")
    print(f"輸入特徵維度: {train_x.shape[1]}")
    print(f"訓練資料形狀: {train_x.shape}")
    print(f"測試資料形狀: {test_x.shape}")
    
    # 確認標籤形狀
    print(f"建築物標籤形狀: {train_b.shape}")
    print(f"位置標籤形狀: {train_c.shape}")
    print(f"樓層標籤形狀: {train_y.shape}")
    
    # 創建整合式深度學習模型
    nn_model = hybrid_model(train_x.shape[1], n_buildings, n_floors)
    nn_model.summary()
    
    # 設置回調函數
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # 加入建築判斷準確度的監控回調
        BuildingAccuracyCallback(threshold=0.95, patience=3)
    ]
    
    # 確保標籤是正確的形狀
    train_b = train_b.reshape(-1)
    train_y = train_y.reshape(-1)
    
    # 訓練整合式深度學習模型
    history = nn_model.fit(
        train_x_enhanced,
        {
            'building_output': train_b,
            'floor_output': train_y,
            'position_output': train_c
        },
        epochs=50,  # 減少訓練輪數避免過度擬合
        batch_size=64,  # 增加批次大小提高穩定性
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 儲存模型
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir, exist_ok=True)
    
    # 使用Python檔案名稱作為模型檔名
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 儲存模型為 .h5 格式
    h5_path = os.path.join(output_model_dir, f'{model_name}.h5')
    nn_model.save(h5_path)
    print(f"模型已保存為 .h5 格式: {h5_path}")
    
    # 儲存模型為 .tflite 格式
    converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(output_model_dir, f'{model_name}.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"模型已保存為 .tflite 格式: {tflite_path}")
    
    # 儲存訓練歷史
    with open(os.path.join(output_model_dir, 'training_history.pkl'), 'wb') as f:
        import pickle
        pickle.dump(history.history, f)
    
    # 評估模型
    print("評估整合模型性能...")
    results = nn_model.evaluate(
        test_x_enhanced,
        {
            'building_output': test_b,
            'floor_output': test_y,
            'position_output': test_c
        }
    )
    
    # 使用模型進行預測
    predictions = nn_model.predict(test_x_enhanced)
    building_pred = np.argmax(predictions[0], axis=1)
    floor_pred = np.argmax(predictions[1], axis=1)
    position_pred = predictions[2]
    
    # 計算各項指標
    building_accuracy = np.mean(building_pred == test_b) * 100
    floor_accuracy = np.mean(floor_pred == test_y) * 100
    
    # 計算位置預測誤差
    position_errors = np.sqrt(np.sum((test_c - position_pred) ** 2, axis=1))
    mean_error = np.mean(position_errors)
    median_error = np.median(position_errors)
    
    # 儲存評估結果
    with open(os.path.join(output_model_dir, 'model_results.txt'), 'w') as f:
        f.write("整合式深度學習模型評估結果:\n")
        f.write(f"  建築物分類準確率: {building_accuracy:.2f}%\n")
        f.write(f"  樓層分類準確率: {floor_accuracy:.2f}%\n")
        f.write(f"  位置預測平均誤差: {mean_error:.4f}\n")
        f.write(f"  位置預測中位數誤差: {median_error:.4f}\n")
    
    print(f"整合式深度學習模型訓練和評估完成。模型已保存在 {output_model_dir}")
    return nn_model, history

class IntegratedPositionModel:
    """整合式深度學習定位模型類別，取代原混合模型"""
    
    def __init__(self, model):
        """
        初始化整合式定位模型
        
        參數:
            model: 整合式深度學習模型
        """
        self.model = model
    
    @staticmethod
    def load(model_dir):
        """
        從磁碟載入整合式定位模型
        
        參數:
            model_dir: 模型儲存目錄
        
        返回:
            IntegratedPositionModel 實例
        """
        # 加載自定義層
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        # 載入H5模型
        model_path = os.path.join(model_dir, 'integrated_model.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到整合模型: {model_path}")
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
        return IntegratedPositionModel(model)
    
    def predict(self, rss_data):
        """
        使用整合式定位模型進行預測
        
        參數:
            rss_data: RSS 信號強度數據，形狀為 (batch_size, n_features)
            
        返回:
            tuple: (建築物分類結果, 樓層分類結果, 位置座標預測)
        """
        # 使用深度學習模型進行預測
        predictions = self.model.predict(rss_data)
        
        # 提取各部分的輸出
        building_output = predictions[0]  # 建築物分類概率
        floor_output = predictions[1]     # 樓層分類概率
        position_pred = predictions[2]    # 位置座標
        
        return building_output, floor_output, position_pred
    
    def save(self, model_dir="./models"):
        """
        保存整合式定位模型到磁碟
        
        參數:
            model_dir: 模型儲存目錄
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # 使用Python檔案名稱作為模型檔名
        model_name = os.path.splitext(os.path.basename(__file__))[0]
            
        # 保存為 .h5 格式
        h5_path = os.path.join(model_dir, f'{model_name}.h5')
        self.model.save(h5_path)
        
        # 保存為 .tflite 格式
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(model_dir, f'{model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"整合式定位模型已保存至: {model_dir}")
        print(f"- H5格式: {h5_path}")
        print(f"- TFLite格式: {tflite_path}")

def create_integrated_positioning_model(hadnn_data_dir, output_model_dir):
    """建立並訓練整合式定位模型，並返回IntegratedPositionModel實例"""
    model, _ = train_enhanced_model(hadnn_data_dir, output_model_dir)
    
    # 創建整合式定位模型實例
    integrated_model = IntegratedPositionModel(model)
    return integrated_model
    
if __name__ == "__main__":
    print("=== 開始整合式深度學習模型訓練 ===")
    hadnn_data_dir = "./hadnn_data"
    output_model_dir = "./models"
    
    # 訓練模型
    model, history = train_enhanced_model(hadnn_data_dir, output_model_dir)
    
    # 創建整合式定位模型實例
    integrated_model = create_integrated_positioning_model(hadnn_data_dir, output_model_dir)
    
    print("整合式定位模型創建完成")
