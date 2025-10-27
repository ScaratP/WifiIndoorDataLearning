import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tflite_validator import verify_tflite_accuracy, compare_model_sizes
import sys
import logging
import warnings
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 關鍵修改 1：移除自定義 AttentionLayer ---
# 舊的 class AttentionLayer(layers.Layer) ... 已被完全移除

# 載入資料集的輔助函數 (與 original_hadnn.py 相同)
def load_dataset():
    """
    動態載入資料集，假設 hadnn_adapter.py 在 preprocess 資料夾中
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_path = os.path.join(current_dir, 'preprocess')
    
    if preprocess_path not in sys.path:
        sys.path.append(preprocess_path)
        
    try:
        from hadnn_adapter import NTTUDataset
    except ImportError:
        print(f"❌ 無法從 {preprocess_path} 導入 NTTUDataset")
        print("請確保 hadnn_adapter.py 位於 'preprocess' 資料夾中")
        return None
    
    hadnn_data_path = os.path.join(current_dir, 'hadnn_data')
    processed_data_path = os.path.join(current_dir, 'processed_data')
    config_path = os.path.join(hadnn_data_path, 'dataset_config.json')
    
    if not os.path.exists(config_path):
        print(f"❌ 在 {config_path} 中找不到 dataset_config.json")
        print("請先執行 preprocess/run_preprocessing.py")
        return None
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    dataset = NTTUDataset(processed_data_path)
    
    print("✅ 資料集成功載入")
    return dataset, config

def build_mlp_multi_output_model(num_rss_features, num_buildings, num_floors, attention_units=128):
    """
    建立 MLP 多輸出模型 (平行預測)
    
    --- 已修改為使用 TFLite 相容的 MultiHeadAttention ---
    """
    
    # 輸入層
    input_rss = layers.Input(shape=(num_rss_features,), name='rss_input')
    
    # --- 關鍵修改 2：用標準層替換自定義層 ---
    # 1. 將輸入 (None, num_rss) 轉換為 (None, num_rss, 1)
    input_reshaped = layers.Reshape((num_rss_features, 1))(input_rss)
    
    # 2. 應用 MultiHeadAttention (與 HADNN 一致)
    num_heads = 4 
    key_dim = max(1, attention_units // num_heads) 
        
    print(f"   使用 TFLite 相容的 MultiHeadAttention (heads={num_heads}, key_dim={key_dim})")
    
    mha_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        name="multi_head_attention"
    )(query=input_reshaped, value=input_reshaped, key=input_reshaped) 
    
    # 3. 將 MHA 的 3D 輸出 (None, num_rss, 1) 壓平回 2D (None, num_rss)
    attention_layer = layers.Flatten(name="attention_flatten")(mha_output)
    # --- 修改結束 ---

    # 將注意力和原始輸入結合
    concatenated_input = layers.Concatenate()([input_rss, attention_layer])
    
    # 共享的 Dense 層 (主幹)
    shared_dense_1 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concatenated_input)
    shared_dense_1 = layers.Dropout(0.5)(shared_dense_1)
    shared_dense_2 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared_dense_1)
    shared_dense_2 = layers.Dropout(0.5)(shared_dense_2)

    # --- 關鍵修改 3：建立三個「平行」的分支 ---
    # 所有分支都從 shared_dense_2 開始，互不依賴

    # 1. 建築物預測分支 (Building Branch)
    b_branch = layers.Dense(256, activation='relu', name='b_branch_dense')(shared_dense_2)
    b_branch = layers.Dropout(0.5)(b_branch)
    b_output = layers.Dense(num_buildings, activation='softmax', name='building_output')(b_branch)

    # 2. 樓層預測分支 (Floor Branch)
    f_branch = layers.Dense(256, activation='relu', name='f_branch_dense')(shared_dense_2)
    f_branch = layers.Dropout(0.5)(f_branch)
    f_output = layers.Dense(num_floors, activation='softmax', name='floor_output')(f_branch)

    # 3. 座標預測分支 (Coordinate Branch)
    c_branch = layers.Dense(256, activation='relu', name='c_branch_dense')(shared_dense_2)
    c_branch = layers.Dropout(0.5)(c_branch)
    c_output = layers.Dense(2, activation='linear', name='coord_output')(c_branch)
    # --- 修改結束 ---

    # 建立多輸出模型
    model = Model(inputs=input_rss, outputs=[b_output, f_output, c_output])
    
    return model

def train_and_evaluate_mlp(dataset, config):
    """
    訓練、評估、儲存和轉換 MLP 模型
    """
    print("\n--- 開始訓練 MLP 多輸出模型 (已修改 TFLite 相容) ---")
    
    # 從 config 獲取模型參數
    num_rss_features = config['n_rss']
    num_buildings = config['n_buildings']
    num_floors = config['n_floors']
    
    # 從 dataset 獲取資料
    train_x = dataset.train_x
    test_x = dataset.test_x
    
    # 準備多個輸出的標籤
    train_labels = {
        'building_output': dataset.train_b,
        'floor_output': dataset.train_y,
        'coord_output': dataset.train_c
    }
    test_labels = {
        'building_output': dataset.test_b,
        'floor_output': dataset.test_y,
        'coord_output': dataset.test_c
    }
    
    # 建立模型
    model = build_mlp_multi_output_model(num_rss_features, num_buildings, num_floors)
    
    # 編譯模型 (注意 loss_weights 可以調整)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss={
            'building_output': 'sparse_categorical_crossentropy', 
            'floor_output': 'sparse_categorical_crossentropy',   
            'coord_output': 'mean_squared_error'               
        },
        loss_weights={
            'building_output': 1.0,  
            'floor_output': 1.0,
            'coord_output': 1.0     
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'coord_output': 'mean_absolute_error'
        }
    )
    
    model.summary()
    
    # 設定回調函數
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # 訓練模型
    print("\n--- 開始訓練 ---")
    history = model.fit(
        train_x,
        train_labels,
        epochs=150,  
        batch_size=32, # <-- *** 已幫您改成 32，避免 OOM ***
        validation_data=(test_x, test_labels),
        callbacks=callbacks
    )
    
    # 評估模型
    print("\n--- 評估最佳模型 ---")
    evaluation = model.evaluate(test_x, test_labels, verbose=1)
    
    print(f"總損失 (Total Loss): {evaluation[0]:.4f}")
    print(f"建築物 損失: {evaluation[1]:.4f}, 準確率: {evaluation[4]:.4f}")
    print(f"樓層 損失: {evaluation[2]:.4f}, 準確率: {evaluation[5]:.4f}")
    print(f"座標 損失 (MSE): {evaluation[3]:.4f}, MAE: {evaluation[6]:.4f}")
    
    # 修正儲存路徑
    model_dir = os.path.join(os.path.dirname(__file__), 'models') 
    os.makedirs(model_dir, exist_ok=True)
    model_h5_path = os.path.join(model_dir, 'mlp.h5')
    model.save(model_h5_path)
    print(f"✅ .h5 模型已儲存至: {model_h5_path}")
    
    # 儲存模型結構
    summary_path = os.path.join(model_dir, 'mlp.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 轉換為 TensorFlow Lite 模型
    print("\n--- 轉換為 TFLite (Float32) ---")
    
    warnings.simplefilter("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    tflite_path = os.path.join(model_dir, 'mlp.tflite')
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS    
        ]
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite 模型已儲存至: {tflite_path}")
        
        # 驗證 TFLite 模型準確性
        print("\n--- 驗證 TFLite 模型準確性 ---")
        try:
            # 驗證座標 (輸出索引為 2)
            is_accurate_coord = verify_tflite_accuracy(
                tflite_path=tflite_path,
                x_test=test_x,
                y_test=test_labels['coord_output'],
                is_multi_output=True,
                output_index=2, # 座標是第 2 個索引
                tolerance=1e-2
            )
            if is_accurate_coord:
                print("✅ TFLite 模型(座標)準確率驗證通過")
            else:
                print("⚠️  TFLite 轉換(座標)存在精度差異")
        
        except TypeError as e_val:
            print(f"   ⚠️ 無法自動驗證 TFLite 模型 (可能 tflite_validator.py 不支援多輸出): {e_val}")
            print("   跳過 TFLite 驗證，但 .tflite 檔案已成功生成。")
        
        compare_model_sizes(model_h5_path, tflite_path)
        
    except Exception as e:
        print(f"❌ TFLite 轉換失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        warnings.resetwarnings()
        logging.getLogger("tensorflow").setLevel(logging.INFO)
    
    print("模型訓練和評估完成")
    return model, history


# 主執行函數
if __name__ == "__main__":
    
    # 載入資料
    dataset, config = load_dataset()
    
    if dataset:
        # 訓練模型
        model, history = train_and_evaluate_mlp(dataset, config)
        
        # (可選) 執行與 original_hadnn.py 相同的錯誤分析
        print("\n--- 分析建築物分類錯誤 (MLP) ---")
        data_path = os.path.join(os.path.dirname(__file__), 'processed_data', 'nttu_wifi_data.csv')
        df = pd.read_csv(data_path)
        
        indices = np.arange(len(df))
        _, test_idx = train_test_split(
            indices, 
            test_size=0.2, 
            random_state=42, # 確保 random_state 一致
            stratify=df['building_id'].values
        )
        test_names = df.iloc[test_idx]['name'].values
        
        # 導入 original_hadnn.py 的分析函數
        try:
            from original_hadnn import analyze_building_misclassifications
            analyze_building_misclassifications(model, dataset.test_x, dataset.test_b, test_names)
        except ImportError:
            print("無法導入 analyze_building_misclassifications 函數，跳過錯誤分析。")
        except Exception as e:
            print(f"分析時發生錯誤: {e}")