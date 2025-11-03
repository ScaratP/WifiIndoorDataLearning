import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import sys
import logging
import warnings
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime # <--- 新增：導入 datetime 模組

# --- (從你修改後的 hadnn_adapter.py 複製過來或 import) ---
# --- 如果你選擇 import，請確保路徑正確 ---
# from preprocess.hadnn_adapter import define_groups
# --- 如果選擇複製，放在這裡 ---
def define_groups(df):
    """
    根據 building 和 floor 欄位，建立新的群組規則：
    - 1樓 (不分建築) -> se1
    - 2樓 (不分建築) -> se2
    - 3樓 (不分建築) -> se3
    - sea4, sea5, seb4, sec4, sec5 維持不變
    回傳 DataFrame 新增 'group' 欄位。
    """
    # ... (省略函數內部程式碼，請從 hadnn_adapter.py 複製完整內容) ...
    print("   正在定義資料群組 (合併 1-3 樓)...")
    df['group'] = 'other'
    try:
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
        df.dropna(subset=['floor'], inplace=True)
        df['floor'] = df['floor'].astype(int)
        df.loc[df['floor'] == 1, 'group'] = 'se1'
        df.loc[df['floor'] == 2, 'group'] = 'se2'
        df.loc[df['floor'] == 3, 'group'] = 'se3'
        df.loc[(df['building'] == 'sea') & (df['floor'] == 4), 'group'] = 'sea4'
        df.loc[(df['building'] == 'sea') & (df['floor'] == 5), 'group'] = 'sea5'
        df.loc[(df['building'] == 'seb') & (df['floor'] == 4), 'group'] = 'seb4'
        df.loc[(df['building'] == 'sec') & (df['floor'] == 4), 'group'] = 'sec4'
        df.loc[(df['building'] == 'sec') & (df['floor'] == 5), 'group'] = 'sec5'
    except Exception as e:
        print(f"   錯誤：在定義群組時發生問題：{e}")
        raise
    print("   ✅ 新群組定義完成。")
    return df
# --- define_groups 函數結束 ---

# --- 設定 ---
DATA_DIR_SPLIT = "hadnn_data_split" # 分割後的資料
DATA_DIR_PROC = "processed_data"    # 包含 CSV 和 label_mappings
MODEL_DIR_BASE = "models" # <--- 修改：基礎模型資料夾名稱
CONFIG_DIR_SPLIT = "hadnn_data_split"

# 模型超參數 (與 train_coordinate_regressors.py 保持一致或微調)
ATTENTION_UNITS = 256
NUM_HEADS = 8
DENSE_UNITS_1 = 1024  # 原始 HADNN 的第一層
DENSE_UNITS_2 = 512   # 原始 HADNN 的第二層 (B 分支前)
BRANCH_DENSE_UNITS = 256 # B 和 F 分支內的 Dense 層
DROPOUT_RATE = 0.3
L2_REG = 0.0005
LEARNING_RATE = 0.0001
EPOCHS = 150 # 可以增加 Epochs
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 25
LR_REDUCE_PATIENCE = 10
ADD_NOISE = False # 是否加入高斯噪聲，可以設為 True 來實驗
NOISE_STDDEV = 0.02

# --- 資料載入函數 (修改) ---
def load_hierarchical_data():
    """
    載入階層式分類器所需的資料：
    - X: train_x_classifier.npy, test_x_classifier.npy
    - y: 重新產生的 train_b, train_f, test_b, test_f
    - num_buildings, num_floors
    - test_y_group_ids (用於最終群組準確率計算)
    - id_to_building, id_to_floor_str (新增返回)
    """
    print("載入階層式分類器資料...")

    # 1. 載入 X 特徵
    try:
        train_x = np.load(os.path.join(DATA_DIR_SPLIT, 'train_x_classifier.npy'))
        test_x = np.load(os.path.join(DATA_DIR_SPLIT, 'test_x_classifier.npy'))
        # 載入真實的群組標籤，用於最後評估
        test_y_group_ids = np.load(os.path.join(DATA_DIR_SPLIT, 'test_y_classifier.npy'))
        num_rss_features = train_x.shape[1]
        print(f"   成功載入 X 特徵 (維度: {num_rss_features})")
    except FileNotFoundError as e:
        print(f"❌ 錯誤：找不到 classifier 的 npy 檔案: {e}")
        print(f"   請先執行 preprocess/run_preprocessing.py")
        return None

    # 2. 載入 label_mappings 以獲取類別數量
    try:
        with open(os.path.join(DATA_DIR_PROC, 'label_mappings.json'), 'r', encoding='utf-8') as f:
            label_mappings = json.load(f)
        num_buildings = len(label_mappings['building_mapping'])
        num_floors = len(label_mappings['floor_mapping'])
        # 新增：預先建立映射
        id_to_building = {v: k for k, v in label_mappings['building_mapping'].items()}
        id_to_floor_str = {v: k for k, v in label_mappings['floor_mapping'].items()}
        print(f"   找到 {num_buildings} 個建築類別, {num_floors} 個樓層類別")
    except Exception as e:
        print(f"❌ 錯誤：無法載入或解析 label_mappings.json: {e}")
        return None

    # 3. 載入 CSV 並重新產生對齊的 B/F 標籤
    try:
        main_csv_path = os.path.join(DATA_DIR_PROC, 'nttu_wifi_data.csv')
        df = pd.read_csv(main_csv_path)
        df = define_groups(df) # 應用相同的群組定義
        df_filtered = df[df['group'] != 'other'].copy() # 過濾

        if len(df_filtered) != (len(train_x) + len(test_x)):
            print("❌ 錯誤：過濾後的 DataFrame 長度與載入的 npy 檔案總長度不匹配！")
            print(f"   DataFrame 長度: {len(df_filtered)}, npy 總長度: {len(train_x) + len(test_x)}")
            return None

        # 提取 B/F/Group ID 標籤
        b_all_labels = df_filtered['building_id'].values.astype(np.int32)
        f_all_labels = df_filtered['floor_id'].values.astype(np.int32)

        # 建立 group_id (必須與 prepare_split_data 中的邏輯一致)
        group_names = sorted(df_filtered['group'].unique())
        group_to_id = {name: i for i, name in enumerate(group_names)}
        df_filtered['group_id'] = df_filtered['group'].map(group_to_id)
        y_all_group_ids = df_filtered['group_id'].values.astype(np.int32)

        # *** 使用完全相同的參數重新分割 B/F 標籤 ***
        indices = np.arange(len(df_filtered))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y_all_group_ids # 使用 group_id 進行分層
        )

        train_b = b_all_labels[train_idx]
        train_f = f_all_labels[train_idx]
        test_b = b_all_labels[test_idx]
        test_f = f_all_labels[test_idx]

        # 驗證長度是否匹配
        if len(train_x) != len(train_b) or len(test_x) != len(test_b):
            print("❌ 錯誤：重新分割後的 B/F 標籤長度與 X 特徵長度不匹配！")
            return None

        print("   成功重新產生並分割 B/F 標籤。")
        return (train_x, train_b, train_f,
                test_x, test_b, test_f, test_y_group_ids,
                num_rss_features, num_buildings, num_floors,
                id_to_building, id_to_floor_str) # 新增返回

    except Exception as e:
        print(f"❌ 錯誤：在處理 CSV 或分割 B/F 標籤時發生問題: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- 模型建立函數 (修改) ---
def build_hierarchical_classifier(num_rss_features, num_buildings, num_floors,
                                attention_units=ATTENTION_UNITS, add_noise=ADD_NOISE):
    """
    建立階層式分類器模型 (B->F)
    移除座標分支。
    """
    input_rss = layers.Input(shape=(num_rss_features,), name='rss_input')

    current_input = input_rss
    if add_noise:
        print(f"   模型中加入高斯噪聲 (stddev={NOISE_STDDEV})")
        current_input = layers.GaussianNoise(stddev=NOISE_STDDEV)(current_input)

    input_reshaped = layers.Reshape((num_rss_features, 1))(current_input)

    key_dim = max(1, attention_units // NUM_HEADS)
    print(f"   使用 TFLite 相容 MultiHeadAttention (heads={NUM_HEADS}, key_dim={key_dim})")
    mha_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=key_dim,
        name="multi_head_attention"
    )(query=input_reshaped, value=input_reshaped, key=input_reshaped)

    attention_layer = layers.Flatten(name="attention_flatten")(mha_output)
    concatenated_input = layers.Concatenate()([current_input, attention_layer]) # 保持與原始一致

    # 共享 Dense 層
    shared_dense_1 = layers.Dense(DENSE_UNITS_1, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(concatenated_input)
    shared_dense_1 = layers.Dropout(DROPOUT_RATE)(shared_dense_1)
    shared_dense_2 = layers.Dense(DENSE_UNITS_2, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(shared_dense_1)
    shared_dense_2 = layers.Dropout(DROPOUT_RATE)(shared_dense_2)

    # 1. 建築物預測分支 (Building Branch) - 保持不變
    b_branch = layers.Dense(BRANCH_DENSE_UNITS, activation='relu')(shared_dense_2)
    b_branch = layers.Dropout(DROPOUT_RATE)(b_branch)
    b_output = layers.Dense(num_buildings, activation='softmax', name='building_output')(b_branch)

    # 2. 樓層預測分支 (Floor Branch) - 保持不變
    f_input_concat = layers.Concatenate()([shared_dense_2, b_output])
    f_branch = layers.Dense(BRANCH_DENSE_UNITS, activation='relu')(f_input_concat)
    f_branch = layers.Dropout(DROPOUT_RATE)(f_branch) # (原始代碼是 f_branch_f，修正一下)
    f_output = layers.Dense(num_floors, activation='softmax', name='floor_output')(f_branch)

    # --- 移除座標預測分支 ---

    # 建立模型，輸出 B 和 F
    model = Model(inputs=input_rss, outputs=[b_output, f_output])
    print("   ✅ Model built successfully.")
    return model

# --- 訓練與評估函數 (修改) ---
def train_and_evaluate_hierarchical(data, run_model_dir, params_filepath, hparams):
    """
    訓練、評估階層式分類器 (B->F)，並記錄參數與結果。
    新增 run_model_dir, params_filepath, hparams 參數。
    """
    print("\n--- 開始訓練階層式分類器 (B->F) ---")
    print(f"結果將儲存於: {run_model_dir}")

    (train_x, train_b, train_f,
    test_x, test_b, test_f, test_y_group_ids,
    num_rss_features, num_buildings, num_floors,
    id_to_building, id_to_floor_str) = data

    # 準備標籤字典
    train_labels = {
        'building_output': train_b,
        'floor_output': train_f
    }
    test_labels = {
        'building_output': test_b,
        'floor_output': test_f
    }

    # 建立模型
    tf.keras.backend.clear_session() # 清理 session
    model = build_hierarchical_classifier(num_rss_features, num_buildings, num_floors,
                                        attention_units=hparams['attention_units'],
                                        add_noise=hparams['add_noise'])

    # 編譯模型 (多輸出)
    model.compile(
        optimizer=Adam(learning_rate=hparams['learning_rate']),
        loss={
            'building_output': 'sparse_categorical_crossentropy',
            'floor_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={ # 可以調整權重，例如更重視樓層
            'building_output': 1.0,
            'floor_output': 1.0
        },
        metrics={ # 分別監控 B 和 F 的準確率
            'building_output': 'accuracy',
            'floor_output': 'accuracy'
        }
    )

    # (可選) 將模型摘要也存檔
    summary_filepath = os.path.join(run_model_dir, "model_summary.txt")
    # --- 修改這裡：加入 encoding='utf-8' ---
    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f: # <--- ★★★ 明確指定 utf-8 編碼
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"   模型摘要已儲存至: {summary_filepath}")
    except IOError as e:
        print(f"   ❌ 錯誤：無法寫入模型摘要檔案 {summary_filepath}: {e}")

    # 回調函數
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=hparams['early_stopping_patience'], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=hparams['lr_reduce_patience'], min_lr=1e-6, verbose=1)
    ]

    # 訓練模型
    print("\n--- 開始訓練 ---")
    history = model.fit(
        train_x,
        train_labels,
        epochs=hparams['epochs'],
        batch_size=hparams['batch_size'],
        validation_split=0.1, # 從訓練集中分割驗證集
        callbacks=callbacks,
        verbose=2
    )
    print("--- 訓練完成 ---")

    # 評估模型 (B 和 F 各自的指標)
    print("\n--- 評估最佳模型 (B/F 指標) ---")
    # 評估結果的順序：[總損失, B損失, F損失, B準確率, F準確率]
    evaluation = model.evaluate(test_x, test_labels, verbose=1)
    print(f"   總損失 (Total Loss): {evaluation[0]:.4f}")
    print(f"   建築物 準確率: {evaluation[3]:.4f}")
    print(f"   樓層   準確率: {evaluation[4]:.4f}")

    # --- 新增：計算最終的群組分類準確率 ---
    print("\n--- 計算最終群組分類準確率 ---")
    group_accuracy = 0.0
    try:
        # 1. 進行預測
        predictions = model.predict(test_x)
        pred_b_ids = np.argmax(predictions[0], axis=1)
        pred_f_ids = np.argmax(predictions[1], axis=1)

        # 2. 載入 Group Name 到 Group ID 的映射
        with open(os.path.join(CONFIG_DIR_SPLIT, 'classifier_config.json'), 'r', encoding='utf-8') as f:
            classifier_config = json.load(f)
        id_to_group_name = classifier_config['group_mapping']
        group_name_to_id = {name: int(id_str) for id_str, name in id_to_group_name.items()}

        # 3. 組合預測的 B/F ID -> 預測的 Group ID
        predicted_group_ids = []
        for i in range(len(pred_b_ids)):
            b_id = pred_b_ids[i]
            f_id = pred_f_ids[i]

            b_name = id_to_building.get(b_id)
            f_num_str = id_to_floor_str.get(f_id)
            f_num = None
            if f_num_str is not None:
                try: f_num = int(f_num_str)
                except ValueError: pass

            pred_group_name = 'other'
            if f_num is not None:
                if f_num == 1: pred_group_name = 'se1'
                elif f_num == 2: pred_group_name = 'se2'
                elif f_num == 3: pred_group_name = 'se3'
                elif b_name == 'sea' and f_num == 4: pred_group_name = 'sea4'
                elif b_name == 'sea' and f_num == 5: pred_group_name = 'sea5'
                elif b_name == 'seb' and f_num == 4: pred_group_name = 'seb4'
                elif b_name == 'sec' and f_num == 4: pred_group_name = 'sec4'
                elif b_name == 'sec' and f_num == 5: pred_group_name = 'sec5'

            pred_group_id = group_name_to_id.get(pred_group_name, -1)
            predicted_group_ids.append(pred_group_id)

        predicted_group_ids = np.array(predicted_group_ids)

        # 4. 與真實的 test_y_group_ids 比較
        correct_predictions = np.sum(predicted_group_ids == test_y_group_ids)
        total_test_samples = len(test_y_group_ids)
        if total_test_samples > 0:
            group_accuracy = correct_predictions / total_test_samples
        print(f"   ✅ 最終群組分類準確率: {group_accuracy:.4f}")

    except Exception as e:
        print(f"❌ Error calculating final group accuracy: {e}")

    # --- 新增：記錄結果到 txt 檔案 ---
    print(f"\n--- 將結果追加到 {params_filepath} ---")
    try:
        actual_epochs = len(history.history['loss']) # 實際運行的 Epochs 數
        results_to_save = {
            "epochs_run": actual_epochs,
            "final_total_loss": evaluation[0],
            "final_building_loss": evaluation[1],
            "final_floor_loss": evaluation[2],
            "final_building_accuracy": evaluation[3],
            "final_floor_accuracy": evaluation[4],
            "final_group_accuracy": group_accuracy,
        }
        with open(params_filepath, 'a', encoding='utf-8') as f:
            f.write("\n--- Training Results ---\n") # 寫入結果標題
            for key, value in results_to_save.items():
                if isinstance(value, (float, np.floating)): # 檢查是否為浮點數
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        print("   ✅ 結果記錄成功。")
    except Exception as e:
        print(f"   ❌ 錯誤：無法記錄結果到檔案: {e}")

    # 儲存模型
    model_h5_path = os.path.join(run_model_dir, 'hierarchical_classifier.h5')
    print(f"\n儲存 H5 模型至: {model_h5_path}...")
    try:
        model.save(model_h5_path)
        print("   H5 模型儲存成功。")
    except Exception as e:
        print(f"   ❌ 錯誤: 無法儲存 H5 模型: {e}")

    # 轉換為 TFLite 模型
    tflite_path = os.path.join(run_model_dir, 'hierarchical_classifier.tflite')
    print(f"\n轉換為 TFLite 模型至: {tflite_path}...")
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
        print(f"   ✅ TFLite 模型儲存成功 (注意：此模型有 2 個輸出)。")
        # TFLite 驗證比較複雜，暫時跳過
    except Exception as e:
        print(f"   ❌ 錯誤: TFLite 轉換失敗: {e}")
    finally:
        warnings.resetwarnings()
        logging.getLogger("tensorflow").setLevel(logging.INFO)

    print("\n模型訓練和評估完成")
    # 返回歷史紀錄和最終的群組準確率，方便比較
    return model, history, group_accuracy


# 主執行函數
if __name__ == "__main__":
    # --- 新增：生成時間戳和創建資料夾 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_model_dir = os.path.join(MODEL_DIR_BASE, f"hierarchical_run_{timestamp}")
    try:
        os.makedirs(run_model_dir, exist_ok=True)
        print(f"本次執行的模型和記錄將儲存於: {run_model_dir}")
    except OSError as e:
        print(f"❌ 錯誤：無法創建資料夾 {run_model_dir}: {e}")
        sys.exit(1) # 停止執行

    # --- 新增：收集並儲存超參數 ---
    hparams = {
        "timestamp": timestamp,
        "script_name": os.path.basename(__file__),
        "data_dir_split": DATA_DIR_SPLIT,
        "data_dir_proc": DATA_DIR_PROC,
        "attention_units": ATTENTION_UNITS,
        "num_heads": NUM_HEADS,
        "dense_units_1": DENSE_UNITS_1,
        "dense_units_2": DENSE_UNITS_2,
        "branch_dense_units": BRANCH_DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "l2_reg": L2_REG,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_reduce_patience": LR_REDUCE_PATIENCE,
        "add_noise": ADD_NOISE,
        "noise_stddev": NOISE_STDDEV if ADD_NOISE else "N/A",
    }
    params_filepath = os.path.join(run_model_dir, "training_params_and_results.txt")
    try:
        with open(params_filepath, 'w', encoding='utf-8') as f:
            f.write("--- Hyperparameters ---\n")
            for key, value in hparams.items():
                f.write(f"{key}: {value}\n")
        print(f"超參數已記錄至: {params_filepath}")
    except IOError as e:
        print(f"❌ 錯誤：無法寫入參數檔案 {params_filepath}: {e}")
        # 可以選擇是否要停止執行

    # 載入資料
    loaded_data = load_hierarchical_data()

    if loaded_data:
        # 訓練模型，並傳入儲存路徑和參數
        model, history, final_group_accuracy = train_and_evaluate_hierarchical(
            loaded_data, run_model_dir, params_filepath, hparams
        )
        print(f"\n========================================================")
        print(f"Final Group Classification Accuracy (Hierarchical B->F): {final_group_accuracy:.4f}")
        print(f"Results saved in: {run_model_dir}")
        print(f"========================================================")
    else:
        print("\n❌ Training aborted due to data loading errors.")