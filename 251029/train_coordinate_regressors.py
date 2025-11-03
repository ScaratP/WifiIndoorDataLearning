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
import datetime # <--- 新增：導入 datetime 模組

# --- 設定 ---
DATA_DIR = "hadnn_data_split" # 預處理輸出的資料夾
MODEL_DIR_BASE = "models"    # 基礎模型資料夾名稱
CONFIG_DIR = "hadnn_data_split" # 設定檔所在的資料夾

# --- ★★★ 關鍵修改：我們定義兩種模型參數 ★★★ ---

# 1. 給「小地圖」 (sea4, sea5...) 用的「基礎」設定
BASE_HPARAMS = {
    "ATTENTION_UNITS": 256,
    "NUM_HEADS": 8,
    "DENSE_UNITS_1": 1024,
    "DENSE_UNITS_2": 512,
    "DROPOUT_RATE": 0.3,
    "L2_REG": 0.0005,
    "LEARNING_RATE": 0.0001, # 保持 0.0001
    "EPOCHS": 500,           # <--- 增加到 500
    "BATCH_SIZE": 32,
    "EARLY_STOPPING_PATIENCE": 50, # <--- 大幅增加耐心
    "LR_REDUCE_PATIENCE": 15       # <--- 增加耐心
    # "ADD_NOISE": True,     # (先關掉噪聲，專心處理複雜度)
    # "NOISE_STDDEV": 0.02,
}

# 2. 給「大地圖」 (se1, se2, se3) 用的「加大」設定
LARGE_HPARAMS = BASE_HPARAMS.copy() # 繼承基礎設定
LARGE_HPARAMS.update({
    "ATTENTION_UNITS": 512,        # <--- 加倍
    "NUM_HEADS": 8,
    "DENSE_UNITS_1": 2048,       # <--- 加倍
    "DENSE_UNITS_2": 1024,       # <--- 加倍
    "DROPOUT_RATE": 0.4,           # <--- Dropout 提高一點，因為模型變超大
    "L2_REG": 0.0001,              # <--- L2 再降一點
    "EPOCHS": 800,               # <--- 給它更多時間
    "EARLY_STOPPING_PATIENCE": 75, # <--- 給它更多耐心
})

# 3. ★★★ 新增：給「棘手地圖」 (se1, se3) 用的「精調」設定 ★★★
PROBLEMATIC_HPARAMS = LARGE_HPARAMS.copy() # 繼承「加大」設定
PROBLEMATIC_HPARAMS.update({
    "LEARNING_RATE": 0.00001,      # <--- 學習率降一半！
    "EPOCHS": 1000,                # <--- 給它更多時間
    "EARLY_STOPPING_PATIENCE": 100,# <--- 更有耐心
    "LR_REDUCE_PATIENCE": 20,      # <--- 更有耐心
    "ADD_NOISE": True,             # <--- ★ 把噪聲加回來
    "NOISE_STDDEV": 0.02,
})

# 4. 定義哪些群組要用「加大」設定
PROBLEMATIC_GROUPS = {'se1', 'se3'}
LARGE_GROUPS = {'se2'}

# --- 模型建立函數 (基於 MLP + Attention) ---
def build_coord_regressor_model(num_rss_features, hparams):
    """
    建立座標回歸模型 (MLP + MultiHeadAttention)
    輸入 RSSI，輸出 2D 座標。
    """
    # --- ★★★ 從 hparams 解包我們需要的參數 ★★★ ---
    attention_units = hparams['ATTENTION_UNITS']
    num_heads = hparams['NUM_HEADS']
    dense_units_1 = hparams['DENSE_UNITS_1']
    dense_units_2 = hparams['DENSE_UNITS_2']
    dropout_rate = hparams['DROPOUT_RATE']
    l2_reg = hparams['L2_REG']

    input_rss = layers.Input(shape=(num_rss_features,), name='rss_input')

    # 加入高斯噪聲以提高穩健性 (Robustness) (可選)
    # --- ★★★ 關鍵修改：我們要檢查 hparams 是否要啟用噪聲 ★★★ ---
    if hparams.get("ADD_NOISE", False):
        print("      啟用高斯噪聲 (GaussianNoise) 提高穩健性。")
        noisy_input = layers.GaussianNoise(stddev=hparams["NOISE_STDDEV"])(input_rss)
        input_reshaped = layers.Reshape((num_rss_features, 1))(noisy_input)
    else:
        # 不加噪聲的版本
        input_reshaped = layers.Reshape((num_rss_features, 1))(input_rss)

    key_dim = max(1, attention_units // num_heads)

    # TFLite 相容的 MultiHeadAttention
    mha_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="multi_head_attention"
    )(query=input_reshaped, value=input_reshaped, key=input_reshaped)

    attention_layer = layers.Flatten(name="attention_flatten")(mha_output)

    # 將注意力和原始輸入結合 (或只用注意力輸出)
    # concatenated_input = layers.Concatenate()([input_rss, attention_layer]) # 結合
    concatenated_input = attention_layer # 只用 Attention 特徵

    # 主要 Dense 層
    shared_dense_1 = layers.Dense(
        dense_units_1, activation='relu', # <--- 它會自動使用新傳入的
        kernel_regularizer=regularizers.l2(l2_reg) # <--- 它會自動使用新傳入的
    )(concatenated_input)
    shared_dense_1 = layers.Dropout(dropout_rate)(shared_dense_1) # <--- 它會自動使用新傳入的

    shared_dense_2 = layers.Dense(
        dense_units_2, activation='relu', # <--- 它會自動使用新傳入的
        kernel_regularizer=regularizers.l2(l2_reg) # <--- 它會自動使用新傳入的
    )(shared_dense_1)
    shared_dense_2 = layers.Dropout(dropout_rate)(shared_dense_2) # <--- 它會自動使用新傳入的

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
    print("--- 開始訓練 N 個獨立座標回歸模型 ---")
    
    # --- 新增：生成時間戳和創建資料夾 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_model_dir = os.path.join(MODEL_DIR_BASE, f"coord_run_{timestamp}")
    try:
        os.makedirs(run_model_dir, exist_ok=True)
        print(f"本次執行的模型和記錄將儲存於: {run_model_dir}")
    except OSError as e:
        print(f"❌ 錯誤：無法創建資料夾 {run_model_dir}: {e}")
        sys.exit(1) # 停止執行

    params_filepath = os.path.join(run_model_dir, "training_params_and_results.txt")
    try:
        with open(params_filepath, 'w', encoding='utf-8') as f:
            f.write("--- Hyperparameters (BASE) ---\n")
            for key, value in BASE_HPARAMS.items():
                f.write(f"{key}: {value}\n")

            f.write("\n--- Hyperparameters (LARGE) ---\n")
            for key, value in LARGE_HPARAMS.items():
                f.write(f"{key}: {value}\n")

        print(f"超參數已記錄至: {params_filepath}")
    except IOError as e:
        print(f"❌ 錯誤：無法寫入參數檔案 {params_filepath}: {e}")
        # (可以選擇是否要停止執行)

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

        # --- ★★★ 關鍵修改：使用三層邏輯 ★★★ ---
        if group_name in PROBLEMATIC_GROUPS:
            hparams = PROBLEMATIC_HPARAMS
            print(f"   偵測到棘手地圖群組，使用「精調」參數 (LR={hparams['LEARNING_RATE']})")
        elif group_name in LARGE_GROUPS:
            hparams = LARGE_HPARAMS
            print(f"   偵測到大地圖群組，使用「加大」參數 (Dense={hparams['DENSE_UNITS_1']})")
        else:
            hparams = BASE_HPARAMS
            print(f"   使用「基礎」參數 (Dense={hparams['DENSE_UNITS_1']})")
        # --- ★★★ 修改完畢 ★★★ ---
        
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
        tf.keras.backend.clear_session()
        model = build_coord_regressor_model(num_rss_features, hparams)

        # C. 編譯模型
        print("   編譯模型...")
        model.compile(
            optimizer=Adam(learning_rate=hparams['LEARNING_RATE']), # <--- 傳入參數
            loss='mean_squared_error', 
            metrics=['mean_absolute_error'] 
        )

        # D. 設定回調函數
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=hparams['EARLY_STOPPING_PATIENCE'], # <--- 傳入參數
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=hparams['LR_REDUCE_PATIENCE'], # <--- 傳入參數
                min_lr=1e-6, 
                verbose=1
            )
        ]

        # E. 訓練模型
        print("   開始訓練...")
        history = model.fit(
            train_x, train_c, 
            epochs=hparams['EPOCHS'],      # <--- 傳入參數
            batch_size=hparams['BATCH_SIZE'],  # <--- 傳入參數
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
            print(f"   ❌ 錯誤：在設定檔中找不到 {group_name} 的標準化參數！無法計算真實誤差。")
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
            'epochs_run': len(history.history['loss']) # <--- 新增：記錄實際運行的 Epochs
        }

        # H. 儲存 H5 模型
        # --- 修改：儲存到 run_model_dir ---
        model_h5_path = os.path.join(run_model_dir, f'coord_{group_name}.h5')
        print(f"   儲存 H5 模型至: {model_h5_path}...")
        try:
            model.save(model_h5_path)
            print("      H5 模型儲存成功。")
        except Exception as e:
            print(f"      ❌ 錯誤: 無法儲存 H5 模型: {e}")

        # I. 轉換為 TFLite 模型
        # --- 修改：儲存到 run_model_dir ---
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


    # --- 修改：將總結報告同時輸出到終端機和日誌檔案 ---
    print(f"\n[步驟 3/3] 所有群組訓練完成。結果摘要：")
    
    try:
        with open(params_filepath, 'a', encoding='utf-8') as f:
            f.write("\n\n--- Training Results (All Groups) ---\n")
            
            # 建立標題和分隔線
            # (我們稍微調整一下格式以容納更多資訊)
            header_str = "| {:<10} | {:<25} | {:<15} |\n".format("群組", "平均定位誤差 (公尺)", "Epochs 運行")
            separator_len = len(header_str.encode('utf-8')) - 8 # (粗略計算，中文佔位)
            separator = "-" * separator_len + "\n"

            # 同時寫入檔案和終端機
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
                epochs_run = res.get('epochs_run', 'N/A') # 獲取 Epochs

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

    print(f"\n--- ✅ 座標回歸模型訓練流程結束 (結果儲存於 {run_model_dir}) ---")


if __name__ == "__main__":
    main()