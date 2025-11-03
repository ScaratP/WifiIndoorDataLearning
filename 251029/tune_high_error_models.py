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
import datetime

# --- 設定 ---
DATA_DIR = "hadnn_data_split" 
# --- ★ 修改：建立一個新的模型資料夾 ---
MODEL_DIR_BASE = "models_tuned_high" # 避免覆蓋舊結果
CONFIG_DIR = "hadnn_data_split"

# --- ★ 修改：專為「高誤差」模型特調的超參數 ★ ---
ATTENTION_UNITS = 256       # <--- 加倍
NUM_HEADS = 8               # <--- 加倍
DENSE_UNITS_1 = 1024        # <--- 加倍 (從 512 變 1024)
DENSE_UNITS_2 = 512         # <--- 加倍 (從 256 變 512)
DROPOUT_RATE = 0.5          # <--- 提高 (從 0.4 變 0.5)
L2_REG = 0.005              # (保持不變或略微提高)
LEARNING_RATE = 0.0001      # (保持不變)
EPOCHS = 300                # <--- 延長 (從 150 變 300)，讓模型有更多時間學習
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 20 # <--- 延長 (給它更多耐心)
LR_REDUCE_PATIENCE = 8     # <--- 延長

# --- (模型建立函數 build_coord_regressor_model 保持不變) ---
def build_coord_regressor_model(num_rss_features, attention_units=ATTENTION_UNITS):
    """
    建立座標回歸模型 (MLP + MultiHeadAttention)
    輸入 RSSI，輸出 2D 座標。
    """
    input_rss = layers.Input(shape=(num_rss_features,), name='rss_input')
    input_reshaped = layers.Reshape((num_rss_features, 1))(input_rss)
    key_dim = max(1, attention_units // NUM_HEADS)

    mha_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=key_dim,
        name="multi_head_attention"
    )(query=input_reshaped, value=input_reshaped, key=input_reshaped)

    attention_layer = layers.Flatten(name="attention_flatten")(mha_output)
    concatenated_input = attention_layer 

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

    coord_output = layers.Dense(2, activation='linear', name='coord_output')(shared_dense_2)
    model = Model(inputs=input_rss, outputs=coord_output)
    return model

# --- (資料載入函數 load_group_data 保持不變) ---
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

# --- (誤差計算函數 calculate_mean_distance_error 保持不變) ---
def calculate_mean_distance_error(y_true_original, y_pred_original):
    """計算預測座標與真實座標之間的平均歐幾里得距離 (公尺)"""
    if y_true_original.shape != y_pred_original.shape or y_true_original.shape[1] != 2:
        print("      ❌ 錯誤：計算誤差時座標維度不匹配！")
        return float('inf') 
    distances = np.sqrt(np.sum(np.square(y_true_original - y_pred_original), axis=1))
    mean_distance = np.mean(distances)
    return mean_distance

# --- ★ 修改：主訓練迴圈 ★ ---
def main():
    print("--- 開始訓練「高誤差」群組 (se1, se2, se3) ---")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_model_dir = os.path.join(MODEL_DIR_BASE, f"coord_run_high_error_{timestamp}")
    try:
        os.makedirs(run_model_dir, exist_ok=True)
        print(f"本次執行的模型和記錄將儲存於: {run_model_dir}")
    except OSError as e:
        print(f"❌ 錯誤：無法創建資料夾 {run_model_dir}: {e}")
        sys.exit(1) 

    hparams = {
        "timestamp": timestamp,
        "script_name": os.path.basename(__file__),
        "data_dir": DATA_DIR,
        "config_dir": CONFIG_DIR,
        "attention_units": ATTENTION_UNITS,
        "num_heads": NUM_HEADS,
        "dense_units_1": DENSE_UNITS_1,
        "dense_units_2": DENSE_UNITS_2,
        "dropout_rate": DROPOUT_RATE,
        "l2_reg": L2_REG,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_reduce_patience": LR_REDUCE_PATIENCE,
    }
    params_filepath = os.path.join(run_model_dir, "training_params_and_results.txt")
    try:
        with open(params_filepath, 'w', encoding='utf-8') as f:
            f.write("--- Hyperparameters (High Error Models) ---\n")
            for key, value in hparams.items():
                f.write(f"{key}: {value}\n")
        print(f"超參數已記錄至: {params_filepath}")
    except IOError as e:
        print(f"❌ 錯誤：無法寫入參數檔案 {params_filepath}: {e}")

    # 1. 載入設定檔
    print(f"\n[步驟 1/3] 載入設定檔...")
    coord_scaler_config_path = os.path.join(CONFIG_DIR, 'coord_scaler_config.json')
    
    # --- ★ 修改：只鎖定我們要的群組 ★ ---
    group_names = ["se1", "se2", "se3"]
    print(f"   將專注訓練高誤差群組: {group_names}")
    
    try:
        with open(coord_scaler_config_path, 'r', encoding='utf-8') as f:
            coord_scaler_config = json.load(f)
        print(f"   成功載入座標標準化參數。")
    except Exception as e:
        print(f"❌ 錯誤：無法載入或解析 {coord_scaler_config_path}: {e}")
        return

    # 2. 迴圈訓練每個群組的模型
    print(f"\n[步驟 2/3] 開始迴圈訓練 {len(group_names)} 個模型...")
    results = {} 

    for group_name in group_names:
        print(f"\n--- 正在處理群組: {group_name} ---")
        start_time_group = time.time()

        # (A, B, C, D, E, F, G 步驟... 都和之前一樣, 保持不變)
        # A. 載入資料
        train_x, train_c, test_x, test_c, test_c_original = load_group_data(DATA_DIR, group_name)
        if train_x is None:
            results[group_name] = {'mean_distance_error_m': 'N/A - Data Load Error'}
            continue
        num_rss_features = train_x.shape[1]
        print(f"   資料維度: RSSI={num_rss_features}, 座標=2")
        print(f"   訓練集大小: {len(train_x)}, 測試集大小: {len(test_x)}")

        # B. 建立新模型實例
        print("   建立模型架構...")
        tf.keras.backend.clear_session()
        model = build_coord_regressor_model(num_rss_features)

        # C. 編譯模型
        print("   編譯模型...")
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mean_squared_error', 
            metrics=['mean_absolute_error'] 
        )

        # D. 設定回調函數
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=LR_REDUCE_PATIENCE,
                min_lr=1e-6, 
                verbose=1
            )
        ]

        # E. 訓練模型
        print("   開始訓練...")
        history = model.fit(
            train_x, train_c, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1, 
            callbacks=callbacks,
            verbose=2 
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
        scaler_params = coord_scaler_config.get(group_name)
        if not scaler_params:
            print(f"   ❌ 錯誤：在設定檔中找不到 {group_name} 的標準化參數！")
            mean_distance_error_m = float('inf')
        else:
            mean_x = scaler_params['mean_x']
            std_x = scaler_params['std_x']
            mean_y = scaler_params['mean_y']
            std_y = scaler_params['std_y']
            pred_c_original = pred_c_scaled.copy()
            pred_c_original[:, 0] = (pred_c_scaled[:, 0] * std_x) + mean_x
            pred_c_original[:, 1] = (pred_c_scaled[:, 1] * std_y) + mean_y
            mean_distance_error_m = calculate_mean_distance_error(test_c_original, pred_c_original)
            print(f"   ✅ 平均定位誤差: {mean_distance_error_m:.2f} 公尺")

        results[group_name] = {
            'test_loss_mse': eval_loss,
            'test_mae_scaled': eval_mae,
            'mean_distance_error_m': mean_distance_error_m,
            'epochs_run': len(history.history['loss'])
        }

        # (H, I 步驟... 都和之前一樣, 保持不變)
        # H. 儲存 H5 模型
        model_h5_path = os.path.join(run_model_dir, f'coord_{group_name}.h5')
        print(f"   儲存 H5 模型至: {model_h5_path}...")
        try:
            model.save(model_h5_path)
        except Exception as e:
            print(f"      ❌ 錯誤: 無法儲存 H5 模型: {e}")

        # I. 轉換為 TFLite 模型
        tflite_path = os.path.join(run_model_dir, f'coord_{group_name}.tflite')
        print(f"   轉換為 TFLite 模型至: {tflite_path}...")
        warnings.simplefilter("ignore")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"      ✅ TFLite 模型儲存成功。")
        except Exception as e:
            print(f"      ❌ 錯誤: TFLite 轉換失敗: {e}")
        finally:
            warnings.resetwarnings()
            logging.getLogger("tensorflow").setLevel(logging.INFO)

        end_time_group = time.time()
        print(f"   --- 群組 {group_name} 處理完成，耗時: {end_time_group - start_time_group:.2f} 秒 ---")

    # (3. 輸出總結報告 ... 和之前一樣, 保持不變)
    print(f"\n[步驟 3/3] 所有群組訓練完成。結果摘要：")
    try:
        with open(params_filepath, 'a', encoding='utf-8') as f:
            f.write("\n\n--- Training Results (High Error Groups) ---\n")
            header_str = "| {:<10} | {:<25} | {:<15} |\n".format("群組", "平均定位誤差 (公尺)", "Epochs 運行")
            separator = "-" * (len(header_str) - 8) + "\n" # 粗估
            print(separator, end="")
            f.write(separator)
            print(header_str, end="")
            f.write(header_str)
            print(separator, end="")
            f.write(separator)
            total_error = 0.0
            valid_groups = 0
            for group_name, res in results.items():
                error = res['mean_distance_error_m']
                epochs_run = res.get('epochs_run', 'N/A')
                if np.issubdtype(type(error), np.number) and np.isfinite(error):
                    line = "| {:<10} | {:<25.2f} | {:<15} |\n".format(group_name, error, epochs_run)
                    print(line, end="")
                    f.write(line)
                    total_error += error
                    valid_groups += 1
                else:
                    line = "| {:<10} | {:<25} | {:<15} |\n".format(group_name, str(error), epochs_run)
                    print(line, end="")
                    f.write(line)
            print(separator, end="")
            f.write(separator)
            if valid_groups > 0:
                average_error = total_error / valid_groups
                avg_line = f"   所有 {valid_groups} 個有效群組的平均誤差: {average_error:.2f} 公尺\n"
                print(avg_line, end="")
                f.write(avg_line)
            else:
                avg_line = "   未能成功計算任何群組的誤差。\n"
                print(avg_line, end="")
                f.write(avg_line)
        print(f"\n   ✅ 摘要已儲存至: {params_filepath}")
    except Exception as e:
        print(f"\n   ❌ 錯誤：無法將摘要寫入 {params_filepath}: {e}")
    print(f"\n--- ✅ 高誤差模型訓練流程結束 (結果儲存於 {run_model_dir}) ---")

if __name__ == "__main__":
    main()