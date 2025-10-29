import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import time
import logging
import warnings
import sys

# --- 設定 ---
DATA_DIR = "hadnn_data_split" # 預處理輸出的資料夾
MODEL_DIR = "models"         # 儲存 H5 和 TFLite 模型的資料夾
CONFIG_DIR = "hadnn_data_split" # 設定檔所在的資料夾

# 模型超參數 (可以根據需要調整)
ATTENTION_UNITS = 128
NUM_HEADS = 4
DENSE_UNITS_1 = 512 # 第一個主要 Dense 層
DENSE_UNITS_2 = 256 # 第二個主要 Dense 層 (注意力後)
DROPOUT_RATE = 0.4
L2_REG = 0.005
LEARNING_RATE = 0.0001
EPOCHS = 150 # 減少 Epochs 以加快範例執行，實際可設 100-150
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE = 5

# --- 模型建立函數 (基於 MLP + Attention) ---
def build_coord_regressor_model(num_rss_features, attention_units=ATTENTION_UNITS):
    """
    建立座標回歸模型 (MLP + MultiHeadAttention)
    輸入 RSSI，輸出 2D 座標。
    """
    input_rss = layers.Input(shape=(num_rss_features,), name='rss_input')

    # 加入高斯噪聲以提高魯棒性 (可選)
    # noisy_input = layers.GaussianNoise(stddev=0.02)(input_rss)
    # input_reshaped = layers.Reshape((num_rss_features, 1))(noisy_input)

    # 不加噪聲的版本
    input_reshaped = layers.Reshape((num_rss_features, 1))(input_rss)

    key_dim = max(1, attention_units // NUM_HEADS)

    # TFLite 相容的 MultiHeadAttention
    mha_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=key_dim,
        name="multi_head_attention"
    )(query=input_reshaped, value=input_reshaped, key=input_reshaped)

    attention_layer = layers.Flatten(name="attention_flatten")(mha_output)

    # 將注意力和原始輸入結合 (或只用注意力輸出)
    # concatenated_input = layers.Concatenate()([input_rss, attention_layer]) # 結合
    concatenated_input = attention_layer # 只用 Attention 特徵

    # 主要 Dense 層
    shared_dense_1 = layers.Dense(
        DENSE_UNITS_1, activation='relu',
        kernel_regularizer=regularizers.l2(L2_REG)
    )(concatenated_input)
    shared_dense_1 = layers.Dropout(DROPOUT_RATE)(shared_dense_1)

    shared_dense_2 = layers.Dense(
        DENSE_UNITS_2, activation='relu',
        kernel_regularizer=regularizers.l2(L2_REG)
    )(shared_dense_1)
    shared_dense_2 = layers.Dropout(DROPOUT_RATE)(shared_dense_2)

    # 座標輸出層 (2 個單元，線性激活)
    coord_output = layers.Dense(2, activation='linear', name='coord_output')(shared_dense_2)

    model = Model(inputs=input_rss, outputs=coord_output)
    return model

# --- 資料載入函數 ---
def load_group_data(data_dir, group_name):
    """載入特定群組的訓練和測試資料"""
    print(f"      載入 {group_name} 的資料...")
    try:
        train_x = np.load(os.path.join(data_dir, f'train_x_{group_name}.npy'))
        train_c = np.load(os.path.join(data_dir, f'train_c_{group_name}.npy'))
        test_x = np.load(os.path.join(data_dir, f'test_x_{group_name}.npy'))
        test_c = np.load(os.path.join(data_dir, f'test_c_{group_name}.npy'))
        test_c_original = np.load(os.path.join(data_dir, f'test_c_original_{group_name}.npy'))
        print(f"      成功載入 {group_name} 資料。")
        return train_x, train_c, test_x, test_c, test_c_original
    except FileNotFoundError as e:
        print(f"      ❌ 錯誤：找不到 {group_name} 的 npy 檔案: {e}")
        return None, None, None, None, None

# --- 評估函數：計算平均距離誤差 ---
def calculate_mean_distance_error(y_true_original, y_pred_original):
    """計算預測座標與真實座標之間的平均歐幾里得距離 (公尺)"""
    if y_true_original.shape != y_pred_original.shape or y_true_original.shape[1] != 2:
        print("      ❌ 錯誤：計算誤差時座標維度不匹配！")
        return float('inf') # 返回無窮大表示錯誤

    # 計算每個點的歐幾里得距離
    distances = np.sqrt(np.sum(np.square(y_true_original - y_pred_original), axis=1))
    # 計算平均距離
    mean_distance = np.mean(distances)
    return mean_distance

# --- 主訓練迴圈 ---
def main():
    print("--- 開始訓練 8 個獨立座標回歸模型 ---")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 載入設定檔
    print(f"\n[步驟 1/3] 載入設定檔...")
    classifier_config_path = os.path.join(CONFIG_DIR, 'classifier_config.json')
    coord_scaler_config_path = os.path.join(CONFIG_DIR, 'coord_scaler_config.json')

    try:
        with open(classifier_config_path, 'r', encoding='utf-8') as f:
            classifier_config = json.load(f)
        group_names = list(classifier_config['group_mapping'].values()) # 從 ID->Name 映射獲取名稱列表
        group_names.sort() # 確保順序
        print(f"   找到 {len(group_names)} 個群組: {group_names}")
    except Exception as e:
        print(f"❌ 錯誤：無法載入或解析 {classifier_config_path}: {e}")
        return

    try:
        with open(coord_scaler_config_path, 'r', encoding='utf-8') as f:
            coord_scaler_config = json.load(f)
        print(f"   成功載入座標標準化參數。")
    except Exception as e:
        print(f"❌ 錯誤：無法載入或解析 {coord_scaler_config_path}: {e}")
        return

    # 2. 迴圈訓練每個群組的模型
    print(f"\n[步驟 2/3] 開始迴圈訓練 {len(group_names)} 個模型...")
    results = {} # 儲存每個群組的最終誤差

    for group_name in group_names:
        print(f"\n--- 正在處理群組: {group_name} ---")
        start_time_group = time.time()

        # A. 載入資料
        train_x, train_c, test_x, test_c, test_c_original = load_group_data(DATA_DIR, group_name)
        if train_x is None:
            print(f"   跳過群組 {group_name} 因資料載入失敗。")
            results[group_name] = {'mean_distance_error_m': 'N/A - Data Load Error'}
            continue

        num_rss_features = train_x.shape[1]
        print(f"   資料維度: RSSI={num_rss_features}, 座標=2")
        print(f"   訓練集大小: {len(train_x)}, 測試集大小: {len(test_x)}")

        # B. 建立新模型實例
        print("   建立模型架構...")
        # 清除之前的 Keras session (避免潛在的命名衝突或記憶體洩漏)
        tf.keras.backend.clear_session()
        model = build_coord_regressor_model(num_rss_features)
        # model.summary() # (可選) 印出模型結構

        # C. 編譯模型
        print("   編譯模型...")
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mean_squared_error', # 回歸任務常用 MSE
            metrics=['mean_absolute_error'] # MAE 更直觀 (單位與座標相同)
        )

        # D. 設定回調函數
        callbacks = [
            EarlyStopping(
                monitor='val_loss', # 監控驗證集損失
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, # 完成後恢復到最佳權重
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, # 學習率乘以 0.5
                patience=LR_REDUCE_PATIENCE,
                min_lr=1e-6, # 最小學習率
                verbose=1
            )
        ]

        # E. 訓練模型
        print("   開始訓練...")
        history = model.fit(
            train_x, train_c, # 使用標準化後的座標作為目標
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1, # 從訓練集中分出 10% 作為驗證集
            callbacks=callbacks,
            verbose=2 # 顯示每個 epoch 的結果
        )
        print("   訓練完成。")

        # F. 評估模型 (在標準化座標上)
        print("   評估模型 (標準化座標)...")
        eval_loss, eval_mae = model.evaluate(test_x, test_c, verbose=0)
        print(f"   測試集損失 (MSE): {eval_loss:.4f}")
        print(f"   測試集平均絕對誤差 (MAE - Scaled): {eval_mae:.4f}")

        # G. 預測並計算真實世界誤差 (公尺)
        print("   預測測試集座標並計算真實誤差 (公尺)...")
        pred_c_scaled = model.predict(test_x)

        # 獲取該群組的標準化參數
        scaler_params = coord_scaler_config.get(group_name)
        if not scaler_params:
            print(f"   ❌ 錯誤：在設定檔中找不到 {group_name} 的標準化參數！無法計算真實誤差。")
            mean_distance_error_m = float('inf')
        else:
            mean_x = scaler_params['mean_x']
            std_x = scaler_params['std_x']
            mean_y = scaler_params['mean_y']
            std_y = scaler_params['std_y']

            # 反標準化預測結果
            pred_c_original = pred_c_scaled.copy()
            pred_c_original[:, 0] = (pred_c_scaled[:, 0] * std_x) + mean_x
            pred_c_original[:, 1] = (pred_c_scaled[:, 1] * std_y) + mean_y

            # 計算平均距離誤差
            mean_distance_error_m = calculate_mean_distance_error(test_c_original, pred_c_original)
            print(f"   ✅ 平均定位誤差: {mean_distance_error_m:.2f} 公尺")

        results[group_name] = {
            'test_loss_mse': eval_loss,
            'test_mae_scaled': eval_mae,
            'mean_distance_error_m': mean_distance_error_m
        }

        # H. 儲存 H5 模型
        model_h5_path = os.path.join(MODEL_DIR, f'coord_{group_name}.h5')
        print(f"   儲存 H5 模型至: {model_h5_path}...")
        try:
            model.save(model_h5_path)
            print("      H5 模型儲存成功。")
        except Exception as e:
            print(f"      ❌ 錯誤: 無法儲存 H5 模型: {e}")

        # I. 轉換為 TFLite 模型
        tflite_path = os.path.join(MODEL_DIR, f'coord_{group_name}.tflite')
        print(f"   轉換為 TFLite 模型至: {tflite_path}...")

        # 暫時禁用 TensorFlow 的冗餘日誌
        warnings.simplefilter("ignore")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # 確保包含 TF Ops (如果使用了像 MultiHeadAttention 這樣的層)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()

            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"      ✅ TFLite 模型儲存成功。")

            # (可選) 比較模型大小
            # if compare_model_sizes:
            #     compare_model_sizes(model_h5_path, tflite_path)

        except Exception as e:
            print(f"      ❌ 錯誤: TFLite 轉換失敗: {e}")
            # import traceback
            # traceback.print_exc() # 可取消註解以獲得詳細錯誤
        finally:
            # 恢復日誌設定
            warnings.resetwarnings()
            logging.getLogger("tensorflow").setLevel(logging.INFO)

        end_time_group = time.time()
        print(f"   --- 群組 {group_name} 處理完成，耗時: {end_time_group - start_time_group:.2f} 秒 ---")


    # (在檔案開頭確保導入了 numpy: import numpy as np)

    # 3. 輸出總結報告
    print(f"\n[步驟 3/3] 所有群組訓練完成。結果摘要：")
    print("-" * 50)
    print("| {:<10} | {:<25} |".format("群組", "平均定位誤差 (公尺)"))
    print("-" * 50)
    total_error = 0.0 # 初始化為浮點數
    valid_groups = 0

    for group_name, res in results.items():
        error = res['mean_distance_error_m']

        # --- 修改後的判斷式 ---
        # 使用 np.issubdtype 檢查是否為數字型態 (包含 NumPy 數字)
        # 使用 np.isfinite 檢查是否為有效的有限數值 (排除 inf 和 NaN)
        if np.issubdtype(type(error), np.number) and np.isfinite(error):
            print("| {:<10} | {:<25.2f} |".format(group_name, error))
            total_error += error
            valid_groups += 1
        else:
            # 如果不是有效數字 (例如 'N/A' 或 inf)，則照常印出
            print("| {:<10} | {:<25} |".format(group_name, str(error)))
        # --- 修改結束 ---

    print("-" * 50)
    if valid_groups > 0:
        average_error = total_error / valid_groups
        print(f"   所有 {valid_groups} 個有效群組的平均誤差: {average_error:.2f} 公尺") # <-- 這裡現在應該能正確計算了
    else:
        print("   未能成功計算任何群組的誤差。")

    print("\n--- ✅ 座標回歸模型訓練流程結束 ---")

if __name__ == "__main__":
    main()