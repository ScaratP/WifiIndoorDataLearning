import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import pickle
import pandas as pd
import time


# 添加 TFLite 驗證函數的導入
try:
    from tflite_validator import verify_tflite_accuracy, compare_model_sizes
except ImportError:
    print("警告: 無法導入 tflite_validator，TFLite 模型驗證將被跳過")
    verify_tflite_accuracy = None
    compare_model_sizes = None

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

# 創建自定義的JSON編碼器處理NumPy數據類型
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def mlp_model(input_dim, n_buildings, n_floors):
    """簡化的MLP模型架構，取代隨機森林
    同時處理建築物分類、樓層分類和位置預測"""
    
    # 確保輸入參數為純 Python 整數
    input_dim = int(input_dim)
    n_buildings = int(n_buildings)
    n_floors = int(n_floors)
    
    print(f"創建模型 - 輸入維度: {input_dim}, 建築物類別數: {n_buildings}, 樓層類別數: {n_floors}")
    
    # 輸入層 - 使用明確的形狀定義
    inputs = tf.keras.layers.Input(shape=(input_dim,), dtype=tf.float32, name='wifi_input')
    
    # 簡化共享特徵提取層 (更精簡的網路)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # 使用共享特徵層
    shared_features = x
    
    # 建築物分類分支
    building_features = layers.Dense(64, activation='relu')(shared_features)
    building_features = layers.Dropout(0.2)(building_features)
    building_output = layers.Dense(n_buildings, activation='softmax', name='building_output')(building_features)
    
    # 樓層分類分支
    floor_features = layers.Dense(64, activation='relu')(shared_features)
    floor_features = layers.Dropout(0.2)(floor_features)
    floor_output = layers.Dense(n_floors, activation='softmax', name='floor_output')(floor_features)
    
    # 位置預測分支
    position_features = layers.Dense(64, activation='relu')(shared_features)
    position_features = layers.Dropout(0.2)(position_features)
    position_output = layers.Dense(2, name='position_output')(position_features)
    
    # 創建整合模型 - 使用明確的輸入和輸出
    model = Model(inputs=[inputs], outputs=[building_output, floor_output, position_output])
    
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

def convert_numpy_types(obj):
    """將NumPy類型轉換為Python標準類型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# ...existing code...
def train_mlp_model(hadnn_data_dir, output_dir="./models"):
    """訓練MLP深度學習模型，取代隨機森林"""
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
    print(f"原始資料形狀 - train_x: {train_x.shape}, train_b: {train_b.shape}, train_c: {train_c.shape}, train_y: {train_y.shape}")
    
    # 處理標籤數據多於特徵數據的情況
    if train_b.shape[0] > train_x.shape[0]:
        print(f"警告: 標籤數據({train_b.shape[0]})多於特徵數據({train_x.shape[0]})，將對標籤數據進行截斷")
        train_b = train_b[:train_x.shape[0]]
        if train_c.shape[0] > train_x.shape[0]:
            train_c = train_c[:train_x.shape[0]]
        if train_y.shape[0] > train_x.shape[0]:
            train_y = train_y[:train_x.shape[0]]
    
    # 修正訓練集形狀
    if train_c.shape[0] != train_x.shape[0]:
        print(f"訓練集形狀不匹配 - train_x: {train_x.shape}, train_c: {train_c.shape}")
        if train_c.shape[0] == 2 * train_x.shape[0]:
            # 可能是座標被重複存儲
            train_c = train_c[:train_x.shape[0]]
            print(f"已修正訓練集位置座標形狀: {train_c.shape}")
        elif train_c.shape[0] > train_x.shape[0]:
            # 如果形狀不匹配但不是倍數關係，直接截斷
            train_c = train_c[:train_x.shape[0]]
            print(f"已截斷訓練集位置座標形狀: {train_c.shape}")
    
    # 修正測試集形狀
    if test_b.shape[0] > test_x.shape[0]:
        print(f"警告: 測試標籤數據({test_b.shape[0]})多於特徵數據({test_x.shape[0]})，將對標籤數據進行截斷")
        test_b = test_b[:test_x.shape[0]]
        if test_c.shape[0] > test_x.shape[0]:
            test_c = test_c[:test_x.shape[0]]
        if test_y.shape[0] > test_x.shape[0]:
            test_y = test_y[:test_x.shape[0]]
            
    if test_c.shape[0] != test_x.shape[0]:
        print(f"測試集形狀不匹配 - test_x: {test_x.shape}, test_c: {test_c.shape}")
        if test_c.shape[0] == 2 * test_x.shape[0]:
            # 可能是座標被重複存儲
            test_c = test_c[:test_x.shape[0]]
            print(f"已修正測試集位置座標形狀: {test_c.shape}")
        elif test_c.shape[0] > test_x.shape[0]:
            # 如果形狀不匹配但不是倍數關係，直接截斷
            test_c = test_c[:test_x.shape[0]]
            print(f"已截斷測試集位置座標形狀: {test_c.shape}")
    
    # 確保標籤是正確的形狀
    train_b = train_b.reshape(-1)
    train_c = train_c.reshape(-1, 2) if len(train_c.shape) == 1 and train_c.shape[0] % 2 == 0 else train_c
    train_y = train_y.reshape(-1)
    
    test_b = test_b.reshape(-1)
    test_c = test_c.reshape(-1, 2) if len(test_c.shape) == 1 and test_c.shape[0] % 2 == 0 else test_c
    test_y = test_y.reshape(-1)
    
    # 最終確認所有數據形狀一致
    print(f"處理後資料形狀 - train_x: {train_x.shape}, train_b: {train_b.shape}, train_c: {train_c.shape}, train_y: {train_y.shape}")
    
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
    
    # 創建MLP模型
    try:
        print("創建MLP模型...")
        input_dim = int(train_x.shape[1])
        n_buildings_int = int(n_buildings)
        n_floors_int = int(n_floors)
        print(f"Debug資訊 - 特徵維度: {input_dim} (型別: {type(input_dim)}), 建築物類別: {n_buildings_int} (型別: {type(n_buildings_int)}), 樓層類別: {n_floors_int} (型別: {type(n_floors_int)})")
        mlp = mlp_model(input_dim, n_buildings_int, n_floors_int)
        mlp.summary()
        dummy_input = np.random.random((1, input_dim))
        test_output = mlp.predict(dummy_input)
        print("模型測試成功，輸出形狀：", [out.shape for out in test_output])
    except Exception as e:
        print(f"創建模型失敗: {e}")
        print(f"Debug資訊 - 特徵維度: {train_x.shape[1]}, 建築物類別: {n_buildings}, 樓層類別: {n_floors}")
        raise
    
    # 設置回調函數
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        BuildingAccuracyCallback(threshold=0.95, patience=3)
    ]

    # 加入訓練計時與可觀測性
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.epoch_times = []
            self.train_start = time.time()
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()
        def on_epoch_end(self, epoch, logs=None):
            dt = time.time() - self._t0
            self.epoch_times.append(dt)
            print(f"[Timing] Epoch {epoch+1} 用時: {dt:.3f}s")
        def on_train_end(self, logs=None):
            total = time.time() - self.train_start
            if self.epoch_times:
                print(f"[Timing] 平均每輪: {np.mean(self.epoch_times):.3f}s, 中位數: {np.median(self.epoch_times):.3f}s, 總訓練時間: {total:.2f}s")
            else:
                print(f"[Timing] 總訓練時間: {total:.2f}s")

    callbacks.append(TimeHistory())

    # 訓練設定與可觀測性輸出
    epochs = 50
    batch_size = 64
    val_split = 0.2
    n_train = train_x_enhanced.shape[0]
    val_samples = int(n_train * val_split)
    n_train_eff = n_train - val_samples
    steps_per_epoch = int(np.ceil(n_train_eff / batch_size))
    gpus = tf.config.list_physical_devices('GPU')
    print(f"訓練樣本數: {n_train} (有效訓練: {n_train_eff}, 驗證: {val_samples}), 批次大小: {batch_size}, 每輪步數: {steps_per_epoch}")
    print(f"可用裝置 - GPU: {len(gpus)} 台, CPU: {len(tf.config.list_physical_devices('CPU'))} 台")

    # 確保標籤是正確的形狀
    train_b = train_b.reshape(-1)
    train_y = train_y.reshape(-1)

    # 訓練MLP模型
    fit_start = time.time()
    history = mlp.fit(
        train_x_enhanced,
        {
            'building_output': train_b,
            'floor_output': train_y,
            'position_output': train_c
        },
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1
    )
    fit_total = time.time() - fit_start
    trained_epochs = len(history.history.get('loss', []))
    last_loss = history.history.get('loss', [None])[-1]
    last_val_loss = history.history.get('val_loss', [None])[-1]
    print(f"實際訓練的 epoch 數: {trained_epochs}/{epochs}, 最後一輪 loss: {last_loss}, val_loss: {last_val_loss}, 總訓練時間: {fit_total:.2f}s")
    
    # 儲存模型
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model_name = 'mlp'
    
    # 儲存模型為 .h5 格式
    h5_path = os.path.join(output_dir, f'{model_name}.h5')
    mlp.save(h5_path)
    print(f"模型已保存為 .h5 格式: {h5_path}")
    
    # 儲存模型為 .tflite 格式
    print("正在轉換為 TensorFlow Lite 格式...")
    import warnings
    import logging
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    # 清理模型，移除訓練時的狀態
    model_for_conversion = tf.keras.models.clone_model(mlp)
    model_for_conversion.set_weights(mlp.get_weights())
    model_for_conversion.compile(
        optimizer='adam',
        loss={
            'building_output': 'sparse_categorical_crossentropy',
            'floor_output': 'sparse_categorical_crossentropy',
            'position_output': 'mse'
        }
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_conversion)
    converter.optimizations = []
    converter.target_spec.supported_types = [tf.float32]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    converter.allow_custom_ops = True
    
    try:
        print("正在執行 TFLite 轉換（可能需要一些時間）...")
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, f'{model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ 模型已保存為 .tflite 格式: {tflite_path}")
        print("驗證 TFLite 模型...")
        interpreter_test = tf.lite.Interpreter(model_path=tflite_path)
        interpreter_test.allocate_tensors()
        print(f"✅ TFLite 模型載入成功，輸入形狀: {interpreter_test.get_input_details()[0]['shape']}")
        
        # 驗證 TFLite 模型的準確性（若提供）
        try:
            from tflite_validator import verify_tflite_accuracy, compare_model_sizes
            print("驗證 TFLite 模型準確性...")
            verify_success = verify_tflite_accuracy(
                mlp,
                tflite_path,
                test_x_enhanced[:50],
                test_b[:50],
                test_y[:50],
                test_c[:50],
                tolerance=1e-4
            )
            if verify_success:
                print("✅ TFLite 轉換驗證成功")
            else:
                print("⚠️  TFLite 轉換存在精度差異，但模型已保存")
            compare_model_sizes(h5_path, tflite_path)
        except Exception:
            print("⚠️  跳過 TFLite 模型驗證（驗證模組不可用）")
    except Exception as e:
        print(f"❌ TFLite 轉換失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        warnings.resetwarnings()
        logging.getLogger("tensorflow").setLevel(logging.INFO)
    
    # 儲存訓練歷史
    with open(os.path.join(output_dir, f'{model_name}_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    # 評估模型
    print("評估整合模型性能...")
    results = mlp.evaluate(
        test_x_enhanced,
        {
            'building_output': test_b,
            'floor_output': test_y,
            'position_output': test_c
        }
    )
    
    # 使用模型進行預測
    predictions = mlp.predict(test_x_enhanced)
    building_pred = np.argmax(predictions[0], axis=1)
    floor_pred = np.argmax(predictions[1], axis=1)
    position_pred = predictions[2]
    
    # 指標
    building_accuracy = np.mean(building_pred == test_b) * 100
    floor_accuracy = np.mean(floor_pred == test_y) * 100
    position_errors = np.sqrt(np.sum((test_c - position_pred) ** 2, axis=1))
    mean_error = np.mean(position_errors)
    median_error = np.median(position_errors)
    
    # 生成與儲存摘要
    evaluation_summary = generate_evaluation_summary(
        'MLP深度學習模型',
        building_accuracy,
        floor_accuracy,
        mean_error,
        median_error,
        position_errors,
        test_b,
        test_y,
        building_pred,
        floor_pred,
        results
    )
    summary_path = os.path.join(output_dir, 'mlp.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(evaluation_summary)
    print(f"評估摘要已儲存至: {summary_path}")
    
    # JSON 結果
    results_dict = {
        'building_accuracy': float(building_accuracy),
        'floor_accuracy': float(floor_accuracy),
        'mean_position_error': float(mean_error),
        'median_position_error': float(median_error),
        'model_results': [float(r) for r in results]
    }
    with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2, cls=NpEncoder)
    
    print(f"MLP深度學習模型訓練和評估完成。模型已保存在 {output_dir}")
    return mlp, results_dict
# ...existing code...

def generate_evaluation_summary(model_name, building_accuracy, floor_accuracy, 
                               mean_error, median_error, position_errors, 
                               test_b, test_y, building_pred, floor_pred, 
                               model_results):
    """生成評估摘要"""
    summary = f"""
{'='*60}
{model_name} 評估摘要
{'='*60}
生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

一、模型性能指標
{'='*40}
建築物分類準確率: {building_accuracy:.2f}%
樓層分類準確率: {floor_accuracy:.2f}%
位置預測平均誤差: {mean_error:.4f} 米
位置預測中位數誤差: {median_error:.4f} 米
位置預測標準差: {np.std(position_errors):.4f} 米

二、分類詳細分析
{'='*40}
建築物分類統計:
- 測試樣本數: {len(test_b)}
- 正確分類數: {np.sum(building_pred == test_b)}
- 錯誤分類數: {np.sum(building_pred != test_b)}
- 分類準確率: {building_accuracy:.2f}%

樓層分類統計:
- 測試樣本數: {len(test_y)}
- 正確分類數: {np.sum(floor_pred == test_y)}
- 錯誤分類數: {np.sum(floor_pred != test_y)}
- 分類準確率: {floor_accuracy:.2f}%

三、位置預測分析
{'='*40}
位置誤差統計:
- 最小誤差: {np.min(position_errors):.4f} 米
- 最大誤差: {np.max(position_errors):.4f} 米
- 平均誤差: {mean_error:.4f} 米
- 中位數誤差: {median_error:.4f} 米
- 標準差: {np.std(position_errors):.4f} 米

誤差分佈:
- 誤差 < 1.0 米: {np.sum(position_errors < 1.0)} 個 ({np.sum(position_errors < 1.0)/len(position_errors)*100:.1f}%)
- 誤差 < 2.0 米: {np.sum(position_errors < 2.0)} 個 ({np.sum(position_errors < 2.0)/len(position_errors)*100:.1f}%)
- 誤差 < 3.0 米: {np.sum(position_errors < 3.0)} 個 ({np.sum(position_errors < 3.0)/len(position_errors)*100:.1f}%)
- 誤差 >= 3.0 米: {np.sum(position_errors >= 3.0)} 個 ({np.sum(position_errors >= 3.0)/len(position_errors)*100:.1f}%)

四、模型訓練結果
{'='*40}
模型訓練損失值:
- 總損失: {model_results[0]:.4f}
- 建築物分類損失: {model_results[1]:.4f}
- 樓層分類損失: {model_results[2]:.4f}
- 位置預測損失: {model_results[3]:.4f}

五、綜合評估
{'='*40}
綜合得分 (越低越好): {mean_error / (building_accuracy * floor_accuracy / 10000):.4f}

模型表現評級:
- 建築物分類: {'優秀' if building_accuracy >= 90 else '良好' if building_accuracy >= 80 else '一般' if building_accuracy >= 70 else '需改進'}
- 樓層分類: {'優秀' if floor_accuracy >= 85 else '良好' if floor_accuracy >= 75 else '一般' if floor_accuracy >= 65 else '需改進'}
- 位置預測: {'優秀' if mean_error <= 2.0 else '良好' if mean_error <= 3.0 else '一般' if mean_error <= 4.0 else '需改進'}

六、建議與改進方向
{'='*40}
"""
    
    # 根據結果添加具體建議
    if building_accuracy < 85:
        summary += "- 建築物分類準確率偏低，建議增加更多建築物相關的特徵或調整網路結構\n"
    if floor_accuracy < 80:
        summary += "- 樓層分類準確率偏低，建議改進樓層特徵提取或增加訓練資料\n"
    if mean_error > 3.0:
        summary += "- 位置預測誤差較大，建議調整位置回歸分支的網路結構或損失函數權重\n"
    
    summary += f"\n{'='*60}\n報告結束\n{'='*60}"
    
    return summary

class MLPPositionModel:
    """MLP深度學習定位模型類別"""
    
    def __init__(self, model):
        """初始化MLP定位模型"""
        self.model = model
    
    @staticmethod
    def load(model_dir):
        """從磁碟載入MLP模型"""
        # 加載自定義層
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        # 載入H5模型
        model_path = os.path.join(model_dir, 'mlp.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到MLP模型: {model_path}")
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
        return MLPPositionModel(model)
    
    def predict(self, rss_data):
        """使用MLP模型進行預測"""
        # 使用深度學習模型進行預測
        predictions = self.model.predict(rss_data)
        
        # 提取各部分的輸出
        building_output = predictions[0]  # 建築物分類概率
        floor_output = predictions[1]     # 樓層分類概率
        position_pred = predictions[2]    # 位置座標
        
        return building_output, floor_output, position_pred
    
    def save(self, model_dir="./models"):
        """保存MLP模型到磁碟"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # 保存為 .h5 格式
        h5_path = os.path.join(model_dir, 'mlp.h5')
        self.model.save(h5_path)
        
        # 保存為 .tflite 格式
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(model_dir, 'mlp.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"MLP定位模型已保存至: {model_dir}")
        print(f"- H5格式: {h5_path}")
        print(f"- TFLite格式: {tflite_path}")

def create_mlp_positioning_model(hadnn_data_dir, output_dir):
    """建立並訓練MLP定位模型，並返回MLPPositionModel實例"""
    try:
        # 先嘗試載入已訓練的模型
        mlp_position_model = MLPPositionModel.load(output_dir)
        print("成功載入已存在的模型")
        return mlp_position_model
    except FileNotFoundError:
        # 若模型不存在，則重新訓練
        print("找不到已訓練的模型，開始訓練新模型...")
        model, _ = train_mlp_model(hadnn_data_dir, output_dir)
        mlp_position_model = MLPPositionModel(model)
        return mlp_position_model

if __name__ == "__main__":
    print("=== 開始MLP深度學習模型訓練 ===")
    hadnn_data_dir = "./hadnn_data"
    output_dir = "./models"
    
    # 訓練模型 (只訓練一次)
    mlp_model, mlp_results = train_mlp_model(hadnn_data_dir, output_dir)
    
    # 使用已訓練好的模型創建MLPPositionModel實例
    mlp_position_model = MLPPositionModel(mlp_model)
    
    print("MLP定位模型創建完成")