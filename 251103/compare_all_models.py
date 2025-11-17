#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_all_models_v4.1 (for NTTU 'code' project)

此腳本專為評估 'code' 目錄下 (使用 call_data.py 和 train.py)
訓練流程所產生的模型而設計。

V4.1 更新:
1.  (V4.1) [★★★ 最終修正 ★★★] 
    TFLiteConverter 持續忽略 'signature_keys' 且不儲存輸出名稱
    (V3/V4 方案均失敗，log 顯示輸出為 StatefulPartitionedCall:0)。

    我們放棄讀取「名稱」，改為 100% 穩健的「形狀 (Shape)」反查法。
    - 輸出 shape [?, 2] -> 座標
    - 輸出 shape [?, 3] -> 建築 (self.n_buildings)
    - 輸出 shape [?, 5] -> 樓層 (self.n_floors)
    
    此方法可繞過 TFLiteConverter 的 bug，並正確讀取 TFLite 輸出。

2.  (V2.2) [FIX] 在 Keras (.h5) 區塊中正確定義 has_building 和 has_floor 變數。
3.  (V2.2) [FIX] TFLite 區塊改用 sample-by-sample 迴圈進行預測。
4.  (V2.2) [FIX] 將 haversine_vectorized 函式內置。
"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score
import importlib
from datetime import datetime

# --- (V2.2 修正) ---
# 不再嘗試從外部導入，直接在這裡定義 haversine_vectorized
def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    使用 Numpy 向量化計算兩點之間的 Haversine 距離 (公尺)
    """
    print("  > 正在使用內置 Haversine 函式計算公尺誤差...")
    # 將經緯度從十進位度數轉換為弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 地球半徑 (公尺)
    r_meters = 6371000
    return c * r_meters
# ---------------------------------------------------------


# 檢查是否安裝了 tabulate 套件
TABULATE_AVAILABLE = importlib.util.find_spec("tabulate") is not None

# --- 解決 matplotlib 中文字體問題的修改 ---
try:
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Microsoft JhengHei' not in available_fonts:
        print("警告: 未找到 Microsoft JhengHei 字體，使用預設字體")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
except Exception as e:
    print(f"警告: 字體設定失敗: {e}，使用預設字體")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
# ---------------------------------------------


# 自定義格式化表格函數
def format_table(df):
    cols = df.columns
    header = " | ".join(str(c) for c in cols)
    separator = "-" * len(header)
    rows = []
    
    for _, row in df.iterrows():
        formatted_values = []
        for col in cols:
            val = row[col]
            if isinstance(val, (int, np.integer)):
                formatted_values.append(f"{val}")
            elif isinstance(val, (float, np.floating)):
                formatted_values.append(f"{val:.4f}")
            else:
                formatted_values.append(str(val))
        rows.append(" | ".join(formatted_values))
    
    return header + "\n" + separator + "\n" + "\n".join(rows)

class ModelComparison:
    """WiFi 室內定位模型比較類 (V4.1 - 適配 NTTU 'code' 項目)"""
    
    def __init__(self, data_path_dir='sample', models_to_compare=None, output_dir='./model_comparison_report'):
        """
        初始化模型比較器
        """
        print("初始化模型比較器 (V4.1 - for NTTU 'code' project)...")
        self.data_path_dir = data_path_dir
        self.output_dir = output_dir
        
        try:
            self.data_folder_dir = os.path.join(self.data_path_dir, "data", "NTTU_WIFI_DATA")
            train_csv_path = os.path.join(self.data_folder_dir, "trainingData_nttu.csv")
            self.test_csv_path = os.path.join(self.data_folder_dir, "validationData_nttu.csv")
            
            print(f"從 {train_csv_path} 讀取訓練集以獲取統計數據...")
            train_features = pd.read_csv(train_csv_path)
            
            # (★ V4.1 關鍵 ★) 
            # 我們需要這些值來反查 TFLite 的形狀
            self.n_rss = 531         
            self.n_buildings = 3     
            self.n_floors = 5         
            self.n_coords = 2 # 座標維度
            
            self.lo_mean = train_features.LONGITUDE.mean()
            self.lo_std = train_features.LONGITUDE.std()
            self.la_mean = train_features.LATITUDE.mean()
            self.la_std = train_features.LATITUDE.std()
            
            print(f"  經度(Lon) mean={self.lo_mean:.6f}, std={self.lo_std:.6f}")
            print(f"  緯度(Lat) mean={self.la_mean:.6f}, std={self.la_std:.6f}")
            print(f"  BSSIDs (n_rss)={self.n_rss}, 建築={self.n_buildings}, 樓層={self.n_floors}")
            
        except FileNotFoundError:
            print(f"錯誤：找不到資料檔案。請確保路徑 '{self.data_path_dir}' 正確，")
            print(f"且包含 'data/NTTU_WIFI_DATA/trainingData_nttu.csv'")
            self.test_features_df = None 
            return
        except Exception as e:
            print(f"載入資料統計數據時出錯: {e}")
            self.test_features_df = None 
            return

        # --- (V4.1 修正：路徑指向你最新的 'NTTUse_20251115_021534' 資料夾) ---
        if models_to_compare is None:
            # (★ 請確保這裡指向你 *剛才* 訓練完的資料夾 ★)
            base_dir = os.path.join(self.data_path_dir, 'results/NTTUse_20251115_021534')
            print(f"\n*** 警告：正在讀取 {base_dir} 路徑下的模型 ***")
            print("*** 請確保這是你最新訓練的資料夾 ***\n")
            
            models_to_compare = [
                {
                    'name': 'Baseline (h5)',
                    'path': os.path.join(base_dir, 'Baseline/best_model.h5'), 
                    'type': 'keras'
                },
                {
                    'name': 'CFNN2 (h5)',
                    'path': os.path.join(base_dir, 'CFNN2/best_model.h5'),
                    'type': 'keras'
                },
                {
                    'name': 'HADNN2 (h5)',
                    'path': os.path.join(base_dir, 'HADNN2/best_model.h5'),
                    'type': 'keras'
                },
                {
                    'name': 'HADNNh2 (h5)',
                    'path': os.path.join(base_dir, 'HADNNh2/best_model.h5'),
                    'type': 'keras'
                },
                # (★ 我們這次一定要讓 tflite 成功 ★)
                {
                    'name': 'CFNN2 (tflite)',
                    'path': os.path.join(base_dir, 'CFNN2/model_signed.tflite'),
                    'type': 'tflite'
                },
                {
                    'name': 'HADNN2 (tflite)',
                    'path': os.path.join(base_dir, 'HADNN2/model_signed.tflite'),
                    'type': 'tflite'
                },
                {
                    'name': 'HADNNh2 (tflite)',
                    'path': os.path.join(base_dir, 'HADNNh2/model_signed.tflite'),
                    'type': 'tflite'
                },
            ]
        # -----------------------------------------
        
        self.models_to_compare = models_to_compare
        self.results = {}
        
        print(f"從 {self.test_csv_path} 讀取原始測試資料...")
        self.test_features_df = pd.read_csv(self.test_csv_path)
        self.raw_test_x, self.raw_test_b, self.raw_test_f, self.raw_test_c = self.divide_features(self.test_features_df)
        print(f"原始測試資料載入完畢: x={self.raw_test_x.shape}, b={self.raw_test_b.shape}, f={self.raw_test_f.shape}, c={self.raw_test_c.shape}")

    
    def divide_features(self, features_df):
        """
        從 'code/call_data.py' 的 NTTUse.Divide_features 改編而來。
        """
        n_rss = self.n_rss
        all_values = features_df.values
        
        raw_x = all_values[:, :n_rss].astype(np.float32)
        raw_c_lon_lat = all_values[:, n_rss:n_rss+2].astype(np.float32) 
        raw_f = all_values[:, n_rss+2].astype(np.int32) 
        raw_b = all_values[:, n_rss+3].astype(np.int32) 
        
        raw_f = raw_f - 1
        
        return raw_x, raw_b, raw_f, raw_c_lon_lat

    def normalize_data(self, data_x):
        """
        從 'code/call_data.py' 的 NTTUse.Normalize_data 複製而來。
        """
        mean = np.mean(data_x, axis=1, keepdims=True)
        std = np.std(data_x, axis=1, keepdims=True)
        
        std[std == 0] = 1.0 
        
        normalized_x = (data_x - mean) / std
        
        if np.isnan(normalized_x).any():
            print("警告：Normalize_data 後仍然發現 NaN！正在用 0 填充...")
            normalized_x = np.nan_to_num(normalized_x)
            
        return normalized_x.astype(np.float32)

    def simulate_data_corruption(self, raw_data_x, noise_level=0, missing_rate=0, random_seed=42):
        """
        在 *正規化之前* 模擬資料損壞。
        """
        np.random.seed(random_seed)
        
        corrupted_test_x = raw_data_x.copy()
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, corrupted_test_x.shape)
            corrupted_test_x = corrupted_test_x + noise
            corrupted_test_x = np.clip(corrupted_test_x, -105, 0)
                  
        if missing_rate > 0:
            num_missing = int(np.prod(corrupted_test_x.shape) * missing_rate)
            missing_indices = np.random.choice(corrupted_test_x.size, num_missing, replace=False)
            corrupted_test_x.flat[missing_indices] = -105 
            
        return corrupted_test_x

    
    def load_and_evaluate_model(self, model_info, input_data_normalized, true_b, true_f, true_c_unnormalized):
        """
        載入並評估單個模型 (V4.1 版本 - 基於形狀反查)
        """
        name = model_info['name']
        path = model_info['path']
        model_type = model_info['type']
        
        # --- (V2.2 修正) ---
        # 在 try 區塊的開頭初始化這些變數
        has_building = False
        has_floor = False
        # ---------------------
        
        print(f"\n評估模型: {name} ({path})")
        
        if not os.path.exists(path):
            print(f"錯誤: 找不到模型檔案 {path}")
            return None
        
        try:
            if model_type == 'keras':
                model = tf.keras.models.load_model(path, compile=False)
                
                predictions = model.predict(input_data_normalized)
                
                # --- (V2.2 修正：在此處定義 has_building/has_floor) ---
                if isinstance(predictions, list) and len(predictions) == 3:
                    # HADNN 或 CFNN (順序：座標, 建築, 樓層)
                    pos_pred_normalized = predictions[0]
                    build_pred_logits = predictions[1]
                    floor_pred_logits = predictions[2]
                    has_building = True
                    has_floor = True
                elif isinstance(predictions, np.ndarray) or (isinstance(predictions, list) and len(predictions) == 1):
                    # Baseline (只有座標)
                    if isinstance(predictions, list):
                        pos_pred_normalized = predictions[0]
                    else:
                        pos_pred_normalized = predictions
                    
                    # 創建假的 (全零) 預測，這樣後續程式碼才不會出錯
                    build_pred_logits = np.zeros((len(true_b), self.n_buildings))
                    floor_pred_logits = np.zeros((len(true_f), self.n_floors))
                    has_building = False
                    has_floor = False
                else:
                    print(f"錯誤：無法識別的 Keras 輸出格式。")
                    return None
                # --------------------------------------------------

                building_preds = np.argmax(build_pred_logits, axis=1)
                floor_preds = np.argmax(floor_pred_logits, axis=1)
            
            elif model_type == 'tflite':
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print(f"  TFLite 輸出數量: {len(output_details)}")
                
                # --- (V4.1 關鍵修正：從 "名稱" 改為 "形狀" 反查) ---
                #
                # 我們放棄讀取名稱 (coordinates)，因為 TFLiteConverter 忽略了它。
                # 我們改用 "形狀" 來反查哪一個索引才是正確的輸出。
                #
                print(f"  正在動態反查 TFLite 輸出索引 (V4.1 - 基於形狀)...")
                pos_index = -1
                build_index = -1
                floor_index = -1
                
                has_building = False
                has_floor = False

                # 從 self 獲取我們在 __init__ 中定義的維度
                n_coord = self.n_coords     # e.g., 2
                n_build = self.n_buildings  # e.g., 3
                n_floor = self.n_floors     # e.g., 5

                for detail in output_details:
                    # TFLite 輸出的 shape 可能是 [1, 5] or [1, 3] or [1, 2]
                    output_shape = detail['shape']
                    
                    # 檢查最後一個維度 (特徵數量)
                    if output_shape[-1] == n_coord:
                        pos_index = detail['index']
                        print(f"    > 找到 座標 (shape={output_shape}) @ 索引 {pos_index}")
                    elif output_shape[-1] == n_build:
                        build_index = detail['index']
                        has_building = True
                        print(f"    > 找到 建築 (shape={output_shape}) @ 索引 {build_index}")
                    elif output_shape[-1] == n_floor:
                        floor_index = detail['index']
                        has_floor = True
                        print(f"    > 找到 樓層 (shape={output_shape}) @ 索引 {floor_index}")

                # 檢查是否缺少關鍵輸出
                if pos_index == -1:
                    print(f"錯誤：在 TFLite 輸出中找不到形狀為 [?, {n_coord}] 的 '座標' 輸出！")
                    print("此模型的所有輸出形狀：")
                    for detail in output_details: print(f"  - Index {detail['index']}: {detail['shape']}")
                    return None
                
                # Baseline 模型的特殊處理 (它只有座標)
                if "baseline" in name.lower():
                    if has_building or has_floor:
                        print("警告：Baseline 模型有多餘的輸出，將忽略。")
                    has_building = False # 強制設為 False
                    has_floor = False    # 強制設為 False
                
                # 非 Baseline 模型的檢查
                elif not has_building or not has_floor:
                    print(f"警告：非 Baseline 模型，但找不到 '建築' (shape [?,{n_build}]) 或 '樓層' (shape [?,{n_floor}]) 輸出。")
                
                # --- (V4.1 修正結束) ---
                
                print(f"  TFLite 索引映射: 座標={pos_index}, 建築={build_index if has_building else 'N/A'}, 樓層={floor_index if has_floor else 'N/A'}")
                
                # --- (V2.2 修正：使用 sample-by-sample 迴圈) ---
                total_samples = input_data_normalized.shape[0]
                
                # 準備空的列表來收集單筆預測
                pos_pred_normalized_list = []
                build_pred_logits_list = []
                floor_pred_logits_list = []

                # 檢查 TFLite 模型的輸入形狀
                expected_input_shape = input_details[0]['shape']
                if expected_input_shape[0] != 1:
                    print(f"警告: TFLite 模型的預期批次大小不是 1 (而是 {expected_input_shape[0]})，")
                    print("這可能會導致 'Dimension mismatch' 錯誤。")

                print(f"  開始 TFLite 迴圈預測 (共 {total_samples} 筆資料)...")
                
                for i in range(total_samples):
                    # 獲取第 i 筆資料，並確保它是 [1, n_rss] 的形狀
                    sample_input = input_data_normalized[i:i+1] 
                    
                    interpreter.set_tensor(input_details[0]['index'], sample_input)
                    interpreter.invoke()

                    # (★ 關鍵 ★) 
                    # 使用我們在 V4.1 中動態反查到的 "正確" 索引
                    #
                    # 獲取這 *一筆* 預測結果
                    pos_pred_normalized_list.append(interpreter.get_tensor(pos_index)[0]) # [0] 移除批次維度
                    
                    if has_building:
                        build_pred_logits_list.append(interpreter.get_tensor(build_index)[0])
                    
                    if has_floor:
                        floor_pred_logits_list.append(interpreter.get_tensor(floor_index)[0])
                
                print(f"  TFLite 迴圈預測完成。")

                # 將收集到的結果轉回 numpy 陣列
                pos_pred_normalized = np.array(pos_pred_normalized_list)
                
                if has_building:
                    build_pred_logits = np.array(build_pred_logits_list)
                    building_preds = np.argmax(build_pred_logits, axis=1)
                else:
                    building_preds = np.zeros_like(true_b) # 假的
                
                if has_floor:
                    floor_pred_logits = np.array(floor_pred_logits_list)
                    floor_preds = np.argmax(floor_pred_logits, axis=1)
                else:
                    floor_preds = np.zeros_like(true_f) # 假的
                # ---------------------------------------------

            else:
                print(f"不支援的模型類型: {model_type}")
                return None
            
            # --- (V2 評估邏輯) ---
            
            # 1. 分類準確率
            # (V2.2 修正) 只有在模型真的有預測建築時才計算
            if has_building:
                building_accuracy = accuracy_score(true_b, building_preds)
            else:
                building_accuracy = 0.0 # Baseline 沒有預測建築
            
            correct_building_mask = (building_preds == true_b)
            
            # (V2.2 修正) 只有在模型真的有預測樓層 *且* 建築預測正確時才計算
            if np.any(correct_building_mask) and has_floor:
                # (V4.1 修正) Baseline (has_building=False) 會跳過這裡，
                # 但 HADNN (has_building=True) 會在這裡計算
                if has_building:
                    floor_accuracy = accuracy_score(true_f[correct_building_mask], floor_preds[correct_building_mask])
                else:
                    # 如果模型只有樓層沒有建築 (未來可能)，這裡可以改
                    floor_accuracy = accuracy_score(true_f, floor_preds)
            else:
                floor_accuracy = 0.0
            
            # 2. 位置誤差 (使用 Haversine)
            pos_pred_unnormalized = pos_pred_normalized.copy()
            pos_pred_unnormalized[:, 0] = pos_pred_unnormalized[:, 0] * self.lo_std + self.lo_mean
            pos_pred_unnormalized[:, 1] = pos_pred_unnormalized[:, 1] * self.la_std + self.la_mean
            
            # (V2.2 修正) 只在 haversine 函式第一次被呼叫時印出訊息
            if 'haversine_called' not in globals():
                globals()['haversine_called'] = True
                euclidean_distances = haversine_vectorized(
                    pos_pred_unnormalized[:, 0], pos_pred_unnormalized[:, 1],
                    true_c_unnormalized[:, 0], true_c_unnormalized[:, 1]
                )
            else:
                # 之後不再印出訊息，避免洗版
                lon1, lat1, lon2, lat2 = map(np.radians, [pos_pred_unnormalized[:, 0], pos_pred_unnormalized[:, 1], true_c_unnormalized[:, 0], true_c_unnormalized[:, 1]])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                r_meters = 6371000
                euclidean_distances = c * r_meters

            
            mean_error = np.mean(euclidean_distances)
            median_error = np.median(euclidean_distances)
            std_error = np.std(euclidean_distances)
            
            # 3. 條件位置誤差
            correct_floor = (floor_preds == true_f)
            correct_both = correct_building_mask & correct_floor
            
            # (V2.2 修正) 只有在模型 *同時* 預測 B/F 且 *至少有一筆* 全對時才計算
            if np.any(correct_both) and has_building and has_floor:
                conditional_distances = euclidean_distances[correct_both]
                conditional_mean_error = np.mean(conditional_distances)
                conditional_median_error = np.median(conditional_distances)
                conditional_count = np.sum(correct_both)
            else:
                conditional_mean_error = mean_error 
                conditional_median_error = median_error
                conditional_count = 0
            
            result = {
                'building_accuracy': building_accuracy,
                'floor_accuracy': floor_accuracy,
                'position_mean_error': mean_error,
                'position_median_error': median_error,
                'position_std_error': std_error,
                'conditional_position_mean_error': conditional_mean_error,
                'conditional_position_median_error': conditional_median_error,
                'conditional_correct_count': conditional_count,
                # (V3.0) 儲存 flag 以便穩健性評分
                'has_building': has_building,
                'has_floor': has_floor,
                'predictions': {
                    'building': building_preds.tolist(),
                    'floor': floor_preds.tolist(),
                    'position_lonlat': pos_pred_unnormalized.tolist() 
                }
            }
            
            print("模型評估完成:")
            print(f"  建築物分類準確率: {result['building_accuracy'] * 100:.4f}%")
            print(f"  樓層分類準確率(建築物正確時): {result['floor_accuracy'] * 100:.4f}%")
            print(f"  整體位置預測平均誤差: {result['position_mean_error']:.4f} 公尺")
            print(f"  條件位置預測平均誤差: {result['conditional_position_mean_error']:.4f} 公尺")
            
            return result
        except Exception as e:
            print(f"評估模型 {name} 失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_models(self):
        """
        執行所有模型的比較，並加入穩健性測試 (V2 流程)
        """
        plt.close('all')
        
        if self.test_features_df is None:
            print("無法執行比較，因為測試資料載入失敗。")
            return
            
        robustness_scenarios = {
            '原始資料': {'noise': 0, 'missing_rate': 0},
            '高斯雜訊 3dB': {'noise': 3, 'missing_rate': 0},
            '高斯雜訊 5dB': {'noise': 5, 'missing_rate': 0},
            '高斯雜訊 10dB': {'noise': 10, 'missing_rate': 0},
            '設備故障 10%': {'noise': 0, 'missing_rate': 0.1},
            '設備故障 20%': {'noise': 0, 'missing_rate': 0.2},
            '設備故障 35%': {'noise': 0, 'missing_rate': 0.35},
            '雜訊 5dB + 故障 10%': {'noise': 5, 'missing_rate': 0.1},
            '雜訊 10dB + 故障 20%': {'noise': 10, 'missing_rate': 0.2},
        }
        
        num_trials = 5 
        
        full_results = {}
        
        raw_x = self.raw_test_x
        raw_b = self.raw_test_b
        raw_f = self.raw_test_f
        raw_c_lonlat = self.raw_test_c
        
        # (V2.2 修正) 重置 haversine 的列印標記
        if 'haversine_called' in globals():
            del globals()['haversine_called']
        
        for scenario_name, params in robustness_scenarios.items():
            print(f"\n--- 執行情境: {scenario_name} ({num_trials}次測試) ---")
            print(f"參數: 雜訊={params['noise']}dB, 故障率={params['missing_rate']:.1%}")
            
            scenario_results = {} 
            
            for trial in range(num_trials):
                print(f"  執行第 {trial + 1}/{num_trials} 次測試...")
                
                seed = 42 + trial * 100
                
                corrupted_raw_x = self.simulate_data_corruption(
                    raw_x,
                    noise_level=params['noise'],
                    missing_rate=params['missing_rate'],
                    random_seed=seed
                )
                
                corrupted_normalized_x = self.normalize_data(corrupted_raw_x)
                
                if trial == 0 and (params['noise'] > 0 or params['missing_rate'] > 0):
                    orig_normalized_x = self.normalize_data(raw_x)
                    diff = np.abs(orig_normalized_x - corrupted_normalized_x)
                    print(f"  數據差異驗證 (正規化後):")
                    print(f"    平均差異: {np.mean(diff):.4f}")
                    print(f"    最大差異: {np.max(diff):.4f}")
                    print(f"    差異比例: {np.sum(diff > 1e-5) / np.prod(diff.shape):.2%}")

                for model_info in self.models_to_compare:
                    result = self.load_and_evaluate_model(
                        model_info, 
                        input_data_normalized=corrupted_normalized_x,
                        true_b=raw_b, 
                        true_f=raw_f, 
                        true_c_unnormalized=raw_c_lonlat
                    )
                    
                    if result is not None:
                        if model_info['name'] not in scenario_results:
                            scenario_results[model_info['name']] = []
                        scenario_results[model_info['name']].append(result)

            # 計算平均
            averaged_results = {}
            for model_name, trial_results_list in scenario_results.items():
                if not trial_results_list:
                    continue
                metrics = [
                    'building_accuracy', 'floor_accuracy',
                    'position_mean_error', 'position_median_error',
                    'position_std_error',
                    'conditional_position_mean_error', 'conditional_position_median_error'
                ]
                averaged_result = {}
                for metric in metrics:
                    values = []
                    for result in trial_results_list:
                        val = result.get(metric, 0)
                        if val != float('inf') and not np.isnan(val):
                            values.append(val)
                    if values:
                        averaged_result[metric] = np.mean(values)
                        averaged_result[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0
                    else:
                        averaged_result[metric] = 0 if 'error' not in metric else float('inf')
                        averaged_result[f'{metric}_std'] = 0

                correct_counts = [result.get('conditional_correct_count', 0) for result in trial_results_list]
                averaged_result['conditional_correct_count'] = int(np.mean(correct_counts))
                
                # (V3.0) 傳遞 has_building/has_floor flags
                averaged_result['has_building'] = trial_results_list[0].get('has_building', True)
                averaged_result['has_floor'] = trial_results_list[0].get('has_floor', True)
                
                averaged_result['predictions'] = trial_results_list[-1]['predictions']
                averaged_result['trials'] = trial_results_list
                averaged_result['num_trials'] = len(trial_results_list)
                averaged_results[model_name] = averaged_result

            full_results[scenario_name] = averaged_results
            
            print(f"  情境 {scenario_name} 完成，測試了 {num_trials} 次")
            for model_name, result in averaged_results.items():
                building_acc = result['building_accuracy'] * 100
                building_std = result.get('building_accuracy_std', 0) * 100
                pos_error = result['position_mean_error']
                pos_std = result.get('position_mean_error_std', 0)
                
                print(f"    {model_name}: 建築物準確率 {building_acc:.2f}±{building_std:.2f}%, "
                      f"位置誤差 {pos_error:.4f}±{pos_std:.4f} 公尺")
        
        self.display_comparison(full_results)
        os.makedirs(self.output_dir, exist_ok=True)
        self.generate_comparison_report(full_results, self.output_dir)
        self.generate_comparison_charts(full_results, self.output_dir)
        self.generate_robustness_summary(full_results, self.output_dir)
        
        print(f"比較報告已生成至: {self.output_dir}")

    # ==================================================================
    #
    # 以下所有函式 (display, generate_report, generate_charts, ...)
    # 均保持不變 (從 V2.1 / V2.2 複製)
    # V3.0 / V4.1 的 compute_robustness_scores 函式有微調
    #
    # ==================================================================

    def display_comparison(self, full_results):
        """顯示所有情境下的比較表格"""
        if not full_results:
            print("沒有可比較的結果。")
            return
            
        for scenario_name, results in full_results.items():
            print(f"\n=== 情境: {scenario_name} 的模型比較結果 ===")
            if not results:
                print("此情境沒有成功評估的模型。")
                continue
                
            names = list(results.keys())
            
            def format_with_std(mean, std, decimals=4):
                if std > 0:
                    return f"{mean:.{decimals}f}±{std:.{decimals}f}"
                else:
                    return f"{mean:.{decimals}f}"
            
            data = []
            for name in names:
                res = results[name]
                data.append({
                    '模型名稱': name,
                    '建築物準確率 (%)': format_with_std(res['building_accuracy'] * 100, res.get('building_accuracy_std', 0) * 100, 4),
                    '樓層準確率 (%)': format_with_std(res['floor_accuracy'] * 100, res.get('floor_accuracy_std', 0) * 100, 4),
                    '條件位置平均誤差 (公尺)': format_with_std(res['conditional_position_mean_error'], res.get('conditional_position_mean_error_std', 0), 4),
                    '正確分類樣本數': res['conditional_correct_count'],
                    '位置中位數誤差 (公尺)': f"{res['position_median_error']:.4f}",
                    '位置標準差 (公尺)': f"{res['position_std_error']:.4f}",
                    '測試次數': res['num_trials']
                })
            
            df = pd.DataFrame(data)
            
            if TABULATE_AVAILABLE:
                from tabulate import tabulate
                print(tabulate(df, headers='keys', tablefmt='psql'))
            else:
                print(format_table(df))

    def generate_comparison_report(self, full_results, output_dir):
        """生成詳細的 Markdown 格式報告"""
        report_path = os.path.join(output_dir, 'model_comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 模型評估與穩健性比較報告 (V4.1)\n\n")
            f.write(f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("本報告基於 'code' 專案架構 (NTTUse 資料集, Haversine 誤差) 進行評估。\n\n")
            f.write("**說明**：\n")
            f.write("- **(V4.1 註)**: TFLite 輸出索引是基於「形狀」 (e.g., [?,2], [?,3], [?,5]) 動態反查，以繞過 TFLiteConverter 儲存簽名失敗的 bug。\n")
            f.write("- **位置誤差**：使用 Haversine 公式計算的經緯度實際公尺誤差。\n")
            f.write("- **條件位置誤差**：只針對建築物和樓層都預測正確的樣本計算的位置誤差。\n")
            f.write("- **樓層準確率**：只針對建築物預測正確的樣本計算的樓層分類準確率。\n")
            f.write("- **多次測試**：每個情境進行 5 次獨立測試並報告平均值±標準差。\n\n")
            
            for scenario_name, results in full_results.items():
                if not results:
                    continue
                
                f.write(f"## 情境: {scenario_name}\n\n")
                
                def format_with_std(mean, std, decimals=4):
                    if std > 0:
                        return f"{mean:.{decimals}f}±{std:.{decimals}f}"
                    else:
                        return f"{mean:.{decimals}f}"
                
                data = []
                for name in results.keys():
                    res = results[name]
                    data.append({
                        '模型名稱': name,
                        '建築物準確率 (%)': format_with_std(res['building_accuracy'] * 100, res.get('building_accuracy_std', 0) * 100, 4),
                        '樓層準確率 (%)': format_with_std(res['floor_accuracy'] * 100, res.get('floor_accuracy_std', 0) * 100, 4),
                        '條件位置平均誤差 (公尺)': format_with_std(res['conditional_position_mean_error'], res.get('conditional_position_mean_error_std', 0), 4),
                        '正確分類樣本數': res['conditional_correct_count'],
                        '位置中位數誤差 (公尺)': f"{res['position_median_error']:.4f}",
                        '測試次數': res['num_trials']
                    })
                
                df = pd.DataFrame(data)
                
                if TABULATE_AVAILABLE:
                    from tabulate import tabulate
                    f.write(tabulate(df, headers='keys', tablefmt='github'))
                else:
                    f.write(format_table(df))
                
                f.write("\n\n")
            
            f.write("---\n")
            
        print(f"比較報告已保存至: {report_path}")

    def generate_comparison_charts(self, full_results, output_dir):
        """生成並保存圖表"""
        plt.close('all')
        
        for scenario_name, results in full_results.items():
            if not results:
                continue

            names = list(results.keys())
            building_accuracies = [results[name]['building_accuracy'] * 100 for name in names]
            floor_accuracies = [results[name]['floor_accuracy'] * 100 for name in names]
            mean_errors = [results[name]['position_mean_error'] for name in names]

            # 分類準確度圖表
            df_acc = pd.DataFrame({
                '建築物準確率': building_accuracies,
                '樓層準確率': floor_accuracies
            }, index=names)

            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                df_acc.plot(kind='bar', width=0.4, align='center', ax=ax)
                ax.set_xlabel('模型')
                ax.set_ylabel('準確率 (%)')
                ax.set_title(f'不同模型的分類準確度對比 - {scenario_name}')
                plt.xticks(rotation=15)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'classification_accuracy_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
            finally:
                plt.close(fig)

            # 位置誤差圖表
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.bar(names, mean_errors, color='skyblue')
                ax.set_xlabel('模型')
                ax.set_ylabel('平均誤差 (公尺)')
                ax.set_title(f'不同模型的位置預測平均誤差對比 - {scenario_name}')
                plt.xticks(rotation=15)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'position_errors_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
            finally:
                plt.close(fig)

            # 誤差分布圖表
            self.generate_enhanced_error_distribution_charts(scenario_name, names, results, output_dir)

        # 生成跨情境比較圖表
        self.generate_cross_scenario_charts(full_results, output_dir)

    def generate_enhanced_error_distribution_charts(self, scenario_name, names, results, output_dir):
        """生成增強版的位置誤差分布圖表 (V2 - 使用 Haversine)"""
        plt.close('all')
        
        all_errors = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        true_c_unnormalized = self.raw_test_c
        
        # (V2.2 修正) 重置 haversine 的列印標記
        if 'haversine_called' in globals():
            del globals()['haversine_called']
        
        for i, name in enumerate(names):
            if 'predictions' not in results[name]:
                print(f"警告: {name} 沒有 'predictions' 鍵，跳過 CDF/Boxplot 繪圖。")
                continue
            pred_c_unnormalized = np.array(results[name]['predictions']['position_lonlat'])
            
            # (V2.2 修正) 只在 haversine 函式第一次被呼叫時印出訊息
            if 'haversine_called' not in globals():
                globals()['haversine_called'] = True
                errors = haversine_vectorized(
                    pred_c_unnormalized[:, 0], pred_c_unnormalized[:, 1],
                    true_c_unnormalized[:, 0], true_c_unnormalized[:, 1]
                )
            else:
                # 之後不再印出訊息，避免洗版
                lon1, lat1, lon2, lat2 = map(np.radians, [pred_c_unnormalized[:, 0], pred_c_unnormalized[:, 1], true_c_unnormalized[:, 0], true_c_unnormalized[:, 1]])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                r_meters = 6371000
                errors = c * r_meters
                
            all_errors[name] = errors
        
        # 再次檢查 all_errors 是否為空
        if not all_errors:
            print(f"情境 {scenario_name} 沒有可繪製的誤差數據。")
            return
            
        # 過濾掉 names 中沒有 error 數據的模型
        names_with_errors = [name for name in names if name in all_errors]
        if not names_with_errors:
            print(f"情境 {scenario_name} all_errors 為空，無法繪圖。")
            return

        # 1. 箱型圖 (Box Plot)
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            box_data = [all_errors[name] for name in names_with_errors]
            
            box_plot = ax.boxplot(box_data, labels=names_with_errors, patch_artist=True, 
                                  showmeans=True, meanline=True,
                                  flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
            for patch, color in zip(box_plot['boxes'], colors[:len(names_with_errors)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('模型')
            ax.set_ylabel('位置預測誤差 (公尺)')
            ax.set_title(f'位置預測誤差箱型圖 - {scenario_name}')
            plt.xticks(rotation=15)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'error_boxplot_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"箱型圖生成失敗: {e}")
        finally:
            plt.close(fig)

        # 2. 累積分布函數 (CDF)
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            max_error_limit = 0
            
            for i, name in enumerate(names_with_errors):
                errors = all_errors[name]
                if len(errors) == 0: continue
                
                sorted_errors = np.sort(errors)
                cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                
                max_error_limit = max(max_error_limit, np.max(sorted_errors))
                
                ax.plot(sorted_errors, cumulative_prob * 100, 
                        label=name, color=colors[i % len(colors)], linewidth=2)
                
                for threshold in [1.0, 2.0, 3.0, 5.0, 10.0]:
                    if threshold <= np.max(sorted_errors):
                        percentage = np.sum(errors <= threshold) / len(errors) * 100
                        ax.scatter(threshold, percentage, color=colors[i % len(colors)], s=50, zorder=5, alpha=0.8)
            
            for threshold in [1.0, 2.0, 3.0, 5.0, 10.0]:
                ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
                ax.text(threshold, 5, f'{threshold}m', rotation=90, va='bottom', ha='right', fontsize=9)
            
            ax.set_xlabel('位置預測誤差 (公尺)')
            ax.set_ylabel('累積百分比 (%)')
            ax.set_title(f'位置預測誤差累積分布函數 (CDF) - {scenario_name}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0, right=max(20, max_error_limit)) # 動態 X 軸
            ax.set_ylim(0, 100)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'error_cdf_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"CDF圖生成失敗: {e}")
        finally:
            plt.close(fig)

    def generate_cross_scenario_charts(self, full_results, output_dir):
        """生成跨情境的比較圖表"""
        plt.close('all')
        
        scenarios = list(full_results.keys())
        all_models = set()
        for results in full_results.values():
            all_models.update(results.keys())
        all_models = sorted(list(all_models))
        
        if not all_models:
            print("沒有模型可供跨情境比較。")
            return

        x = np.arange(len(scenarios))
        
        # (V4.1) 增加顏色以容納更多模型 (h5 + tflite)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        width = 0.8 / len(all_models)
            
        # 穩健性測試 - 建築物準確率
        try:
            fig, ax = plt.subplots(figsize=(15, 10))
            for i, model in enumerate(all_models):
                accuracies = []
                for scenario in scenarios:
                    accuracies.append(full_results.get(scenario, {}).get(model, {}).get('building_accuracy', 0) * 100)
                ax.bar(x + (i - len(all_models)/2 + 0.5) * width, accuracies, width, label=model, color=colors[i % len(colors)])

            ax.set_xlabel('測試情境')
            ax.set_ylabel('建築物分類準確率 (%)')
            ax.set_title('不同情境下的模型穩健性測試 - 建築物分類')
            plt.xticks(x, scenarios, rotation=15)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_building_accuracy.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"建築物穩健性圖生成失敗: {e}")
        finally:
            plt.close(fig)

        # 穩健性測試 - 位置誤差
        try:
            fig, ax = plt.subplots(figsize=(15, 10))
            for i, model in enumerate(all_models):
                position_errors = []
                for scenario in scenarios:
                    position_errors.append(full_results.get(scenario, {}).get(model, {}).get('position_mean_error', float('inf')))
                ax.bar(x + (i - len(all_models)/2 + 0.5) * width, position_errors, width, label=model, color=colors[i % len(colors)])

            ax.set_xlabel('測試情境')
            ax.set_ylabel('位置預測平均誤差 (公尺)')
            ax.set_title('不同情境下的模型穩健性測試 - 位置預測')
            plt.xticks(x, scenarios, rotation=15)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_position_error.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"位置穩健性圖生成失敗: {e}")
        finally:
            plt.close(fig)

        # 穩健性評分圖表
        self.generate_robustness_score_chart(full_results, output_dir, scenarios, all_models, colors, x, width)


    def compute_robustness_scores(self, full_results):
        scenarios = list(full_results.keys())
        if not scenarios:
            return None, [], [], {}

        baseline_scenario = '原始資料' if '原始資料' in scenarios else scenarios[0]

        all_models = set()
        for results in full_results.values():
            all_models.update(results.keys())
        all_models = sorted(list(all_models))

        robustness_scores = {}
        for model in all_models:
            scores = []
            if model not in full_results.get(baseline_scenario, {}):
                scores = [0.0 for _ in scenarios]
            else:
                baseline_res = full_results[baseline_scenario][model]
                baseline_building_acc = baseline_res.get('building_accuracy', 0)
                baseline_floor_acc = baseline_res.get('floor_accuracy', 0)
                baseline_position_error = baseline_res.get('position_mean_error', 1e-6) # 避免除以零
                
                # (V3.0/V4.1 修正) 讀取模型是否真的有 B/F 輸出
                baseline_has_building = baseline_res.get('has_building', True)
                baseline_has_floor = baseline_res.get('has_floor', True)
                
                # 權重: 座標 50%, 建築 30%, 樓層 20%
                w_pos, w_bld, w_flr = 0.5, 0.3, 0.2

                for scenario in scenarios:
                    if model in full_results.get(scenario, {}):
                        res = full_results[scenario][model]
                        building_acc = res.get('building_accuracy', 0)
                        floor_acc = res.get('floor_accuracy', 0)
                        position_error = res.get('position_mean_error', float('inf'))
                        
                        score_pos = (baseline_position_error / position_error if position_error > 0 else 0)
                        score_bld = (building_acc / baseline_building_acc if baseline_building_acc > 0 else 0)
                        score_flr = (floor_acc / baseline_floor_acc if baseline_floor_acc > 0 else 0)
                        
                        # (V3.0/V4.1 修正) 處理 Baseline (它不預測 B/F)
                        if not baseline_has_building and not baseline_has_floor:
                            # 如果是 Baseline (例如 Baseline (h5))，評分只看座標
                            score = score_pos
                        else:
                            # 否則，使用加權
                            score = (w_pos * score_pos + w_bld * score_bld + w_flr * score_flr)
                        
                        score = min(score, 1.0)
                        scores.append(score)
                    else:
                        scores.append(0.0)
            robustness_scores[model] = scores

        return baseline_scenario, scenarios, all_models, robustness_scores


    def generate_robustness_score_chart(self, full_results, output_dir, scenarios, all_models, colors, x, width):
        baseline_scenario, scenarios, all_models, robustness_scores = self.compute_robustness_scores(full_results)
        
        try:
            fig, ax = plt.subplots(figsize=(15, 10))
            for i, (model, scores) in enumerate(robustness_scores.items()):
                ax.bar(x + (i - len(all_models)/2 + 0.5) * width, scores, width, label=model, color=colors[i % len(colors)])

            ax.set_xlabel('測試情境')
            ax.set_ylabel('穩健性評分 (1.0=基準)')
            ax.set_title('不同情境下的模型穩健性評分')
            plt.xticks(x, scenarios, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_scores.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"穩健性評分圖生成失敗: {e}")
        finally:
            plt.close(fig)

    def generate_robustness_summary(self, full_results, output_dir):
        baseline_scenario, scenarios, all_models, robustness_scores = self.compute_robustness_scores(full_results)
        if not scenarios:
            print("沒有情境可生成穩健性摘要。")
            return

        try:
            any_result = next(iter(next(iter(full_results.values())).values()))
            num_trials = any_result.get('num_trials', 1)
        except Exception:
            num_trials = 1

        try:
            baseline_idx = scenarios.index(baseline_scenario)
        except ValueError:
            baseline_idx = 0

        model_avg = {}
        for model in all_models:
            scores = robustness_scores.get(model, [])
            if not scores:
                model_avg[model] = 0.0
                continue
            scores_ex_baseline = [s for i, s in enumerate(scores) if i != baseline_idx]
            avg = float(np.mean(scores_ex_baseline)) if scores_ex_baseline else float(np.mean(scores))
            model_avg[model] = avg

        ranking = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)

        preferred = ['原始資料', '高斯雜訊 5dB', '設備故障 10%', '設備故障 35%', '雜訊 5dB + 故障 10%', '雜訊 10dB + 故障 20%']
        selected_scenarios = [s for s in preferred if s in scenarios]
        if not selected_scenarios:
            selected_scenarios = scenarios[:min(6, len(scenarios))]

        scenario_avg = {}
        for i, sc in enumerate(scenarios):
            vals = [robustness_scores[m][i] for m in all_models if len(robustness_scores[m]) > i]
            scenario_avg[sc] = float(np.mean(vals)) if vals else 0.0
        toughest_scenario = min(
            ((sc, v) for sc, v in scenario_avg.items() if sc != baseline_scenario),
            key=lambda x: x[1],
            default=(scenarios[0], scenario_avg.get(scenarios[0], 0.0))
        )

        out_path = os.path.join(output_dir, 'robustness_summary.md')
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("# 模型穩健性評分摘要 (V4.1)\n\n")
            f.write("本摘要基於不同資料損壞情境下的模型性能保持率計算穩健性評分。\n")
            f.write(f"基準情境: {baseline_scenario}\n\n")
            f.write("**評分標準 (權重: 座標 50%, 建築 30%, 樓層 20%)**：\n")
            f.write("- (註: Baseline 模型只基於座標評分)\n")
            f.write("- 1.0：完美保持基準性能\n")
            f.write("- <0.4：較差的穩健性\n\n")
            f.write("**測試方法**：\n")
            f.write(f"- 每個情境進行 {num_trials} 次獨立測試取平均值\n")
            f.write("- 誤差計算: Haversine (公尺)\n")

            f.write("## 模型穩健性排名 (排除基準)\n\n")
            for i, (m, avg) in enumerate(ranking, start=1):
                f.write(f"{i}. **{m}**：平均穩健性評分 {avg:.3f}\n")
            f.write("\n")

            f.write("## 各情境詳細評分\n\n")
            f.write("| 模型 | " + " | ".join([f"{sc}" for sc in selected_scenarios]) + " |\n")
            f.write("|:---| " + " | ".join([":---" for _ in selected_scenarios]) + " |\n")
            
            for m in all_models:
                f.write(f"| {m} ")
                for sc in selected_scenarios:
                    idx = scenarios.index(sc)
                    val = robustness_scores[m][idx] if len(robustness_scores[m]) > idx else 0.0
                    f.write(f"| {val:.3f} ")
                f.write("|\n")

            f.write("\n## 關鍵發現\n\n")
            if ranking:
                f.write(f"- 最穩健模型：{ranking[0][0]} (平均評分：{ranking[0][1]:.3f})\n")
            f.write(f"- 最具挑戰性情境：{toughest_scenario[0]} (平均評Gv：{toughest_scenario[1]:.3f})\n")
            f.write("---\n")
            f.write(f"*報告生成時間：{now_str}*\n")

        print(f"穩健性摘要已生成至: {out_path}")

        
def main():
    """主函數 (V4.1)"""
    print("=== 開始模型比較和穩健性測試 (V4.1 - for 'code' project) ===")
    
    # (★) 
    # 這裡的 output_dir 要改一個新的，
    # 或是指向你剛才 V3.0 跑失敗的那個 'model_comparison_NTTU_V3_FIXED'
    #
    # 這裡的 data_path_dir='sample' 也要確保 
    # 'sample/results/NTTUse_20251115_021534' 這個路徑是存在的
    #
    comparator = ModelComparison(
        data_path_dir='sample', 
        output_dir='./model_comparison_NTTU_V41_FIXED' # 改個新資料夾名稱
    )
    
    # 確保 __init__ 成功執行
    if comparator.test_features_df is not None:
        comparator.compare_models()
    else:
        print("錯誤：ModelComparison 初始化失敗，無法執行比較。")
    
    print("=== 比較和測試完成 ===")

if __name__ == '__main__':
    main()