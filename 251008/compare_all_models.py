import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error
import importlib
from datetime import datetime  # 新增：時間戳用

# 檢查是否安裝了 tabulate 套件
TABULATE_AVAILABLE = importlib.util.find_spec("tabulate") is not None

# --- 解決 matplotlib 中文字體問題的修改 ---
try:
    # 嘗試使用中文字體
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'SimSun']  # 添加後備字體
    plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號
    
    # 測試字體是否支援特殊符號
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


# 自定義格式化表格函數，在沒有 tabulate 時使用
def format_table(df):
    """
    格式化 DataFrame 為簡單的表格文字
    當系統中沒有 tabulate 套件時，用這個函數代替 to_markdown()
    """
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
    """WiFi 室內定位模型比較類"""
    
    def __init__(self, data_dir='./hadnn_data', models_to_compare=None):
        """
        初始化模型比較器
        
        參數:
            data_dir: 測試資料目錄
            models_to_compare: 要比較的模型列表，每項包含 {name, path, type}
        """
        self.data_dir = data_dir
        
        # 預設模型列表，將 TFLite 模型類型標記為 'tflite'
        if models_to_compare is None:
            models_to_compare = [
                {
                    'name': 'original HADNN',
                    'path': './models/original_hadnn.h5',
                    'type': 'keras'
                },
                {
                    'name': 'mlp',
                    'path': './models/mlp.h5',
                    'type': 'keras'
                },
                {
                    'name': 'hadnn+rf',
                    'path': './models/hadnn_n_random_forest.h5',
                    'type': 'keras'
                },
                {
                    'name': 'original hadnn tflite',
                    'path': './models/original_hadnn.tflite',
                    'type': 'tflite'
                },
                {
                    'name': 'mlp tflite',
                    'path': './models/mlp.tflite',
                    'type': 'tflite'
                },
                {
                    'name': 'hadnn+rf tflite',
                    'path': './models/hadnn_n_random_forest.tflite',
                    'type': 'tflite'
                },
            ]
        
        self.models_to_compare = models_to_compare
        self.results = {}
        self.test_data = None
        
    def load_test_data(self):
        """載入測試資料"""
        print("載入測試資料...")
        try:
            self.test_x = np.load(os.path.join(self.data_dir, 'test_x.npy'))
            self.test_b = np.load(os.path.join(self.data_dir, 'test_b.npy'))
            self.test_c = np.load(os.path.join(self.data_dir, 'test_c.npy'))
            self.test_y = np.load(os.path.join(self.data_dir, 'test_y.npy'))
            
            self.test_b = self.test_b.reshape(-1)
            
            if len(self.test_c.shape) == 1:
                if self.test_c.shape[0] % 2 == 0 and self.test_c.shape[0] // 2 == self.test_x.shape[0]:
                    half_len = self.test_c.shape[0] // 2
                    self.test_c = np.column_stack((self.test_c[:half_len], self.test_c[half_len:]))
                else:
                    from_test_y = False
                    try:
                        self.test_f = np.load(os.path.join(self.data_dir, 'test_f.npy'))
                        self.test_f = self.test_f.reshape(-1)
                    except:
                        self.test_f = self.test_y
                        from_test_y = True
                    
                    if not from_test_y:
                        self.test_c = self.test_y.reshape(-1, 2) if len(self.test_y.shape) > 1 else self.test_c
            
            from original_hadnn import advanced_data_preprocessing
            # 原始測試資料的預處理
            self.test_x_enhanced, _ = advanced_data_preprocessing(self.test_x)
            
            config_path = os.path.join(self.data_dir, 'dataset_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'building_mapping': {},
                    'floor_mapping': {}
                }
                
            print(f"測試資料形狀: x={self.test_x.shape}, b={self.test_b.shape}, c={self.test_c.shape}")
            return True
                
        except Exception as e:
            print(f"載入測試資料失敗: {e}")
            return False
            
    def simulate_data_corruption(self, noise_level=0, missing_rate=0, random_seed=42):
        """
        模擬資料損壞，包括增加高斯雜訊和隨機移除資料點。
        
        參數:
            noise_level (float): 高斯雜訊的標準差 (dB)。
            missing_rate (float): 資料遺失的百分比 (0-1)。
            random_seed (int): 隨機種子，確保結果可重現
            
        回傳:
            np.array: 經模擬損壞後的測試資料。
        """
        # 設置隨機種子，確保每次運行結果一致
        np.random.seed(random_seed)
        
        corrupted_test_x = self.test_x.copy()
        
        # 保存原始數據統計信息用於驗證
        original_mean = np.mean(corrupted_test_x)
        original_std = np.std(corrupted_test_x)
        
        # 1. 增加高斯雜訊
        if noise_level > 0:
            # 使用特定雜訊級別創建干擾
            noise = np.random.normal(0, noise_level, corrupted_test_x.shape)
            
            # 紀錄添加的雜訊特性，用於驗證
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            noise_max = np.max(np.abs(noise))
            
            # 應用雜訊
            corrupted_test_x = corrupted_test_x + noise
            
            # 確保 RSSI 值保持在合理範圍內
            corrupted_test_x = np.clip(corrupted_test_x, -120, -30)
            
            # 驗證雜訊是否有效應用
            after_mean = np.mean(corrupted_test_x)
            after_std = np.std(corrupted_test_x)
            
            # 輸出雜訊影響統計
            print(f"  雜訊驗證 - 級別: {noise_level}dB")
            print(f"    原始數據: 平均={original_mean:.2f}, 標準差={original_std:.2f}")
            print(f"    雜訊特性: 平均={noise_mean:.2f}, 標準差={noise_std:.2f}, 最大值={noise_max:.2f}")
            print(f"    應用後: 平均={after_mean:.2f}, 標準差={after_std:.2f}")
            print(f"    變化量: △平均={after_mean-original_mean:.2f}, △標準差={after_std-original_std:.2f}")
            
            # 數據差異比例，用於確認不同級別的雜訊產生不同影響
            diff_ratio = np.mean(np.abs(corrupted_test_x - self.test_x)) / np.abs(original_mean)
            print(f"    數據差異比例: {diff_ratio:.4f}")
                
        # 2. 模擬資料遺失
        if missing_rate > 0:
            num_missing = int(np.prod(corrupted_test_x.shape) * missing_rate)
            missing_indices = np.random.choice(corrupted_test_x.size, num_missing, replace=False)
            corrupted_test_x.flat[missing_indices] = -120  # 用一個極端值代表遺失
            
            print(f"  遺失率驗證 - 比例: {missing_rate:.2%}")
            print(f"    應替換點數: {num_missing}/{corrupted_test_x.size}")
            print(f"    極端值比例: {np.sum(corrupted_test_x == -120) / corrupted_test_x.size:.2%}")
            
        return corrupted_test_x

    def load_and_evaluate_model(self, model_info, input_data):
        """載入並評估單個模型"""
        name = model_info['name']
        path = model_info['path']
        model_type = model_info['type']
        
        print(f"\n評估模型: {name} ({path})")
        
        if not os.path.exists(path):
            print(f"錯誤: 找不到模型檔案 {path}")
            return None
        
        try:
            if model_type == 'keras':
                from original_hadnn import AttentionLayer
                custom_objects = {'AttentionLayer': AttentionLayer}
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                
                predictions = model.predict(input_data)
                
                if isinstance(predictions, list):
                    building_preds = np.argmax(predictions[0], axis=1)
                    floor_preds = np.argmax(predictions[1], axis=1) if len(predictions) > 1 else np.zeros_like(self.test_b)
                    position_preds = predictions[2] if len(predictions) > 2 else np.zeros((len(self.test_x), 2))
                else:
                    building_preds = np.zeros_like(self.test_b)
                    floor_preds = np.zeros_like(self.test_b)
                    position_preds = predictions
            
            elif model_type == 'tflite':
                # 載入 TFLite 模型並建立推論器
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                # 取得輸入和輸出張量的詳細資訊
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print(f"  TFLite 模型詳細資訊:")
                print(f"    輸入形狀: {input_details[0]['shape']}")
                print(f"    輸出數量: {len(output_details)}")
                
                # 修正：更準確的輸出映射邏輯
                output_shapes = [detail['shape'][1] for detail in output_details]
                output_names = [detail.get('name', f'output_{i}') for i, detail in enumerate(output_details)]
                print(f"    輸出形狀: {output_shapes}")
                print(f"    輸出名稱: {output_names}")
                
                # 智能輸出映射 - 優先使用名稱，然後使用形狀
                building_idx = floor_idx = position_idx = None
                
                # 首先基於輸出名稱映射
                for i, name in enumerate(output_names):
                    name_lower = name.lower()
                    if 'building' in name_lower:
                        building_idx = i
                    elif 'floor' in name_lower:
                        floor_idx = i
                    elif 'position' in name_lower:
                        position_idx = i
                
                # 如果名稱映射失敗，使用形狀映射
                if building_idx is None or floor_idx is None or position_idx is None:
                    print("    使用形狀進行輸出映射...")
                    # 從配置中獲取正確的類別數
                    expected_buildings = self.config.get('n_buildings', 3)
                    expected_floors = self.config.get('n_floors', 5)
                    
                    for i, shape in enumerate(output_shapes):
                        if shape == expected_buildings and building_idx is None:
                            building_idx = i
                        elif shape == expected_floors and floor_idx is None:
                            floor_idx = i
                        elif shape == 2 and position_idx is None:  # 位置座標
                            position_idx = i
                
                # 如果仍然無法映射，使用預設順序
                if building_idx is None:
                    building_idx = 0
                if floor_idx is None:
                    floor_idx = 1 if len(output_details) > 1 else 0
                if position_idx is None:
                    position_idx = 2 if len(output_details) > 2 else (1 if len(output_details) > 1 else 0)
                
                print(f"    最終映射 - 建築物: {building_idx}, 樓層: {floor_idx}, 位置: {position_idx}")
                
                # 批次處理預測
                batch_size = 32  # 增加批次大小提高效率
                total_samples = input_data.shape[0]
                
                building_preds_list = []
                floor_preds_list = []
                position_preds_list = []
                
                for start_idx in range(0, total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, total_samples)
                    batch_data = input_data[start_idx:end_idx]
                    
                    batch_building = []
                    batch_floor = []
                    batch_position = []
                    
                    for i in range(batch_data.shape[0]):
                        # 確保輸入數據類型正確
                        sample_input = batch_data[i:i+1].astype(input_details[0]['dtype'])
                        
                        # 檢查輸入形狀
                        expected_shape = input_details[0]['shape']
                        if sample_input.shape[1] != expected_shape[1]:
                            if sample_input.shape[1] > expected_shape[1]:
                                sample_input = sample_input[:, :expected_shape[1]]
                            elif sample_input.shape[1] < expected_shape[1]:
                                padded = np.zeros((1, expected_shape[1]), dtype=sample_input.dtype)
                                padded[:, :sample_input.shape[1]] = sample_input
                                sample_input = padded
                        
                        interpreter.set_tensor(input_details[0]['index'], sample_input)
                        interpreter.invoke()
                        
                        # 獲取所有輸出
                        outputs = []
                        for output_detail in output_details:
                            output = interpreter.get_tensor(output_detail['index'])
                            outputs.append(output[0])  # 去掉 batch 維度
                        
                        # 根據映射解析輸出
                        if building_idx < len(outputs):
                            building_output = outputs[building_idx]
                            batch_building.append(np.argmax(building_output))
                        else:
                            batch_building.append(0)
                        
                        if floor_idx < len(outputs):
                            floor_output = outputs[floor_idx]
                            batch_floor.append(np.argmax(floor_output))
                        else:
                            batch_floor.append(0)
                        
                        if position_idx < len(outputs):
                            position_output = outputs[position_idx]
                            if len(position_output) >= 2:
                                batch_position.append(position_output[:2])
                            else:
                                batch_position.append([position_output[0] if len(position_output) > 0 else 0.0, 0.0])
                        else:
                            batch_position.append([0.0, 0.0])
                    
                    building_preds_list.extend(batch_building)
                    floor_preds_list.extend(batch_floor)
                    position_preds_list.extend(batch_position)

                building_preds = np.array(building_preds_list)
                floor_preds = np.array(floor_preds_list)
                position_preds = np.array(position_preds_list)
                
                # 驗證預測結果的合理性
                print(f"    預測結果統計:")
                print(f"      建築物預測範圍: {building_preds.min()} - {building_preds.max()}")
                print(f"      樓層預測範圍: {floor_preds.min()} - {floor_preds.max()}")
                print(f"      位置預測範圍: x=[{position_preds[:,0].min():.2f}, {position_preds[:,0].max():.2f}], y=[{position_preds[:,1].min():.2f}, {position_preds[:,1].max():.2f}]")

            else:
                print(f"不支援的模型類型: {model_type}")
                return None
            
            building_accuracy = accuracy_score(self.test_b, building_preds)
            try:
                if hasattr(self, 'test_f'):
                    # 修改：只考慮建築物預測正確的樣本計算樓層準確率
                    correct_building_mask = (building_preds == self.test_b)
                    if np.any(correct_building_mask):
                        floor_true = self.test_f[correct_building_mask]
                        floor_pred = floor_preds[correct_building_mask]
                        floor_accuracy = accuracy_score(floor_true, floor_pred)
                    else:
                        floor_accuracy = 0
                else:
                    # 修改：只考慮建築物預測正確的樣本計算樓層準確率
                    correct_building_mask = (building_preds == self.test_b)
                    if np.any(correct_building_mask):
                        floor_true = self.test_y[correct_building_mask]
                        floor_pred = floor_preds[correct_building_mask]
                        floor_accuracy = accuracy_score(floor_true, floor_pred)
                    else:
                        floor_accuracy = 0
            except:
                floor_accuracy = 0
            
            position_mse = mean_squared_error(self.test_c, position_preds)
            position_rmse = np.sqrt(position_mse)
            
            euclidean_distances = np.sqrt(np.sum((self.test_c - position_preds)**2, axis=1))
            mean_error = np.mean(euclidean_distances)
            median_error = np.median(euclidean_distances)
            std_error = np.std(euclidean_distances)
            
            # 新增：計算條件位置誤差（只針對建築物和樓層都預測正確的樣本）
            try:
                if hasattr(self, 'test_f'):
                    floor_true = self.test_f
                else:
                    floor_true = self.test_y
                
                # 找出建築物和樓層都預測正確的樣本
                correct_building = (building_preds == self.test_b)
                correct_floor = (floor_preds == floor_true)
                correct_both = correct_building & correct_floor
                
                if np.any(correct_both):
                    conditional_distances = euclidean_distances[correct_both]
                    conditional_mean_error = np.mean(conditional_distances)
                    conditional_median_error = np.median(conditional_distances)
                    conditional_count = np.sum(correct_both)
                    
                    print(f"    條件位置誤差（建築物+樓層都正確的樣本）:")
                    print(f"      樣本數: {conditional_count}/{len(euclidean_distances)} ({conditional_count/len(euclidean_distances)*100:.1f}%)")
                    print(f"      平均誤差: {conditional_mean_error:.4f} 公尺")
                    print(f"      中位數誤差: {conditional_median_error:.4f} 公尺")
                else:
                    conditional_mean_error = float('inf')
                    conditional_median_error = float('inf')
                    conditional_count = 0
                    print(f"    警告: 沒有建築物和樓層都預測正確的樣本")
                    
            except Exception as e:
                print(f"    條件位置誤差計算失敗: {e}")
                conditional_mean_error = mean_error
                conditional_median_error = median_error
                conditional_count = len(euclidean_distances)
            
            result = {
                'building_accuracy': building_accuracy,
                'floor_accuracy': floor_accuracy,
                'position_mean_error': mean_error,
                'position_median_error': median_error,
                'position_std_error': std_error,
                'position_rmse': position_rmse,
                # 新增條件位置誤差指標
                'conditional_position_mean_error': conditional_mean_error,
                'conditional_position_median_error': conditional_median_error,
                'conditional_correct_count': conditional_count,
                'predictions': {
                    'building': building_preds.tolist(),
                    'floor': floor_preds.tolist(),
                    'position': position_preds.tolist()
                }
            }
            
            print("模型評估完成:")
            print(f"  建築物分類準確率: {result['building_accuracy'] * 100:.4f}%")
            print(f"  樓層分類準確率(建築物正確時): {result['floor_accuracy'] * 100:.4f}%")
            print(f"  整體位置預測平均誤差: {result['position_mean_error']:.4f}")
            print(f"  條件位置預測平均誤差: {result['conditional_position_mean_error']:.4f}")
            
            return result
        except Exception as e:
            print(f"評估模型 {name} 失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_models(self):
        """執行所有模型的比較，並加入穩健性測試"""
        # 在開始前關閉所有圖表
        plt.close('all')
        
        if not self.load_test_data():
            print("無法執行比較，因為測試資料載入失敗。")
            return
            
        # 定義穩健性測試情境
        robustness_scenarios = {
            '原始資料': {'noise': 0, 'missing_rate': 0},
            # '高斯雜訊 1dB': {'noise': 1, 'missing_rate': 0},
            # '高斯雜訊 2dB': {'noise': 2, 'missing_rate': 0},
            # '高斯雜訊 3dB': {'noise': 3, 'missing_rate': 0},
            # '高斯雜訊 4dB': {'noise': 4, 'missing_rate': 0},
            # '高斯雜訊 5dB': {'noise': 5, 'missing_rate': 0},
            # '高斯雜訊 6dB': {'noise': 6, 'missing_rate': 0},
            # '高斯雜訊 7dB': {'noise': 7, 'missing_rate': 0},
            # '高斯雜訊 8dB': {'noise': 8, 'missing_rate': 0},
            # '高斯雜訊 9dB': {'noise': 9, 'missing_rate': 0},
            # '高斯雜訊 10dB': {'noise': 10, 'missing_rate': 0},
            '設備故障 5%': {'noise': 0, 'missing_rate': 0.05},
            # '設備故障 10%': {'noise': 0, 'missing_rate': 0.1},
            # '設備故障 15%': {'noise': 0, 'missing_rate': 0.15},
            # '設備故障 20%': {'noise': 0, 'missing_rate': 0.2},
            # '設備故障 25%': {'noise': 0, 'missing_rate': 0.25},
            # '設備故障 30%': {'noise': 0, 'missing_rate': 0.3},
            # '設備故障 35%': {'noise': 0, 'missing_rate': 0.35},
            # '雜訊 1db + 故障 10%': {'noise': 1, 'missing_rate': 0.1},
            # '雜訊 2db + 故障 10%': {'noise': 2, 'missing_rate': 0.1},
            # '雜訊 3db + 故障 10%': {'noise': 3, 'missing_rate': 0.1},
            # '雜訊 4db + 故障 10%': {'noise': 4, 'missing_rate': 0.1},
            # '雜訊 5dB + 故障 10%': {'noise': 5, 'missing_rate': 0.1},
            # '雜訊 10dB + 故障 20%': {'noise': 10, 'missing_rate': 0.2}
            '雜訊 4db + 故障 10%': {'noise': 4, 'missing_rate': 0.1},
            '雜訊 7db + 故障 10%': {'noise': 7, 'missing_rate': 0.1},
            '雜訊 10db + 故障 10%': {'noise': 10, 'missing_rate': 0.1},
            
        }
        
        # 設定多次測試參數
        num_trials = 5  # 每個情境測試5次
        
        full_results = {}
        
        for scenario_name, params in robustness_scenarios.items():
            print(f"\n--- 執行情境: {scenario_name} ({num_trials}次測試) ---")
            print(f"參數: 雜訊={params['noise']}dB, 故障率={params['missing_rate']:.1%}")
            
            scenario_results = {}  # 存儲所有試驗結果 (每個模型對應多個 trial)
            
            # 添加檢驗點：為每個情境生成唯一標識符，確保結果不會混淆
            scenario_id = f"{scenario_name}_noise{params['noise']}_missing{params['missing_rate']}"
            print(f"  情境ID: {scenario_id}")
            
            # 所有情境都進行相同次數的測試，以獲得統計一致性
            for trial in range(num_trials):
                print(f"  執行第 {trial + 1}/{num_trials} 次測試...")
                
                # 使用不同的隨機種子，即使是原始資料也要有不同種子
                # 這樣可以模擬測試環境的微小變化（如數值精度、記憶體對齊等）
                seed = 42 + trial * 100
                
                # 模擬資料損壞
                corrupted_test_x = self.simulate_data_corruption(
                    noise_level=params['noise'],
                    missing_rate=params['missing_rate'],
                    random_seed=seed
                )
                
                # 應用與原始資料相同的預處理步驟
                from original_hadnn import advanced_data_preprocessing
                corrupted_test_x_enhanced, _ = advanced_data_preprocessing(corrupted_test_x)
                
                # 比較原始數據和破壞後數據的差異，用於驗證
                if trial == 0:  # 只在第一次試驗中執行驗證
                    diff = np.abs(self.test_x_enhanced - corrupted_test_x_enhanced)
                    print(f"  數據差異驗證 - 雜訊={params['noise']}dB, 故障率={params['missing_rate']:.1%}")
                    print(f"    平均差異: {np.mean(diff):.4f}")
                    print(f"    最大差異: {np.max(diff):.4f}")
                    print(f"    差異比例: {np.sum(diff > 0) / np.prod(diff.shape):.2%}")
                
                trial_results = {}
                
                # 對每個模型進行評估
                for model_info in self.models_to_compare:
                    result = self.load_and_evaluate_model(model_info, input_data=corrupted_test_x_enhanced)
                    if result is not None:
                        result['scenario_id'] = scenario_id
                        trial_results[model_info['name']] = result
                        # 新增：在 trial 當下就累積到 scenario_results
                        if model_info['name'] not in scenario_results:
                            scenario_results[model_info['name']] = []
                        scenario_results[model_info['name']].append(result)

            # 移除：原本在這裡才「合併結果」會只留下最後一次 trial
            # for model_name, result in trial_results.items():
            #     if model_name not in scenario_results:
            #         scenario_results[model_name] = []
            #     scenario_results[model_name].append(result)

            # 計算每個模型的平均結果和標準差
            averaged_results = {}
            for model_name, trial_results in scenario_results.items():
                if not trial_results:
                    continue
                metrics = [
                    'building_accuracy', 'floor_accuracy',
                    'position_mean_error', 'position_median_error',
                    'position_std_error', 'position_rmse',
                    'conditional_position_mean_error', 'conditional_position_median_error'
                ]
                averaged_result = {}
                for metric in metrics:
                    values = []
                    for result in trial_results:
                        val = result.get(metric, 0)
                        if val != float('inf') and not np.isnan(val):
                            values.append(val)
                    if values:
                        averaged_result[metric] = np.mean(values)
                        averaged_result[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0
                    else:
                        averaged_result[metric] = 0
                        averaged_result[f'{metric}_std'] = 0

                correct_counts = [result.get('conditional_correct_count', 0) for result in trial_results]
                averaged_result['conditional_correct_count'] = int(np.mean(correct_counts))
                averaged_result['conditional_correct_count_std'] = np.std(correct_counts) if len(correct_counts) > 1 else 0

                # 調整：保留最後一次預測結果供圖表使用
                averaged_result['predictions'] = trial_results[-1]['predictions']
                # 新增：保留每次 trial 的原始指標，供摘要列出每次結果
                averaged_result['trials'] = trial_results
                averaged_result['num_trials'] = len(trial_results)
                averaged_results[model_name] = averaged_result

            full_results[scenario_name] = averaged_results
            
            # 顯示此情境的結果摘要
            print(f"  情境 {scenario_name} 完成，測試了 {num_trials} 次")
            for model_name, result in averaged_results.items():
                building_acc = result['building_accuracy'] * 100
                building_std = result.get('building_accuracy_std', 0) * 100
                pos_error = result['position_mean_error']
                pos_std = result.get('position_mean_error_std', 0)
                
                # 根據情境調整顯示格式
                if scenario_name == '原始資料':
                    if building_std < 0.01 and pos_std < 0.001:  # 標準差很小，顯示為確定性結果
                        print(f"    {model_name}: 建築物準確率 {building_acc:.2f}%, "f"位置誤差 {pos_error:.4f} (確定性結果)")
                    else:
                        print(f"    {model_name}: 建築物準確率 {building_acc:.2f}±{building_std:.3f}%, "f"位置誤差 {pos_error:.4f}±{pos_std:.4f}")
                else:
                    print(f"    {model_name}: 建築物準確率 {building_acc:.2f}±{building_std:.2f}%, "f"位置誤差 {pos_error:.4f}±{pos_std:.4f}")
        
        # 顯示並生成最終報告
        self.display_comparison(full_results)
        output_dir = './model_comparison251008'
        os.makedirs(output_dir, exist_ok=True)
        self.generate_comparison_report(full_results, output_dir)
        self.generate_comparison_charts(full_results, output_dir)
        # 新增：生成穩健性摘要 Markdown（含每次結果）
        self.generate_robustness_summary(full_results, output_dir)
        
        print(f"比較報告已生成至: {output_dir}")

    def display_comparison(self, full_results):
        """顯示所有情境下的比較表格，包含標準差信息"""
        if not full_results:
            print("沒有可比較的結果。")
            return
            
        for scenario_name, results in full_results.items():
            print(f"\n=== 情境: {scenario_name} 的模型比較結果 ===")
            if not results:
                print("此情境沒有成功評估的模型。")
                continue
                
            names = list(results.keys())
            building_accuracies = [results[name]['building_accuracy'] * 100 for name in names]
            building_stds = [results[name].get('building_accuracy_std', 0) * 100 for name in names]
            floor_accuracies = [results[name]['floor_accuracy'] * 100 for name in names]
            floor_stds = [results[name].get('floor_accuracy_std', 0) * 100 for name in names]
            mean_errors = [results[name]['position_mean_error'] for name in names]
            mean_error_stds = [results[name].get('position_mean_error_std', 0) for name in names]
            median_errors = [results[name]['position_median_error'] for name in names]
            std_errors = [results[name]['position_std_error'] for name in names]
            conditional_mean_errors = [results[name].get('conditional_position_mean_error', results[name]['position_mean_error']) for name in names]
            conditional_mean_error_stds = [results[name].get('conditional_position_mean_error_std', 0) for name in names]
            conditional_counts = [results[name].get('conditional_correct_count', len(results[name]['predictions']['building'])) for name in names]
            num_trials = [results[name].get('num_trials', 1) for name in names]
            
            # 格式化帶有標準差的值
            def format_with_std(mean, std, decimals=4):
                if std > 0:
                    return f"{mean:.{decimals}f}±{std:.{decimals}f}"
                else:
                    return f"{mean:.{decimals}f}"
            
            df = pd.DataFrame({
                '模型名稱': names,
                '建築物準確率 (%)': [format_with_std(acc, std, 4) for acc, std in zip(building_accuracies, building_stds)],
                '樓層準確率 (%)': [format_with_std(acc, std, 4) for acc, std in zip(floor_accuracies, floor_stds)],
                '條件位置平均誤差 (公尺)': [format_with_std(err, std, 4) for err, std in zip(conditional_mean_errors, conditional_mean_error_stds)],
                '正確分類樣本數': conditional_counts,
                '位置中位數誤差 (公尺)': [f"{err:.4f}" for err in median_errors],
                '位置標準差 (公尺)': [f"{err:.4f}" for err in std_errors],
                '測試次數': num_trials
            })
            
            if TABULATE_AVAILABLE:
                from tabulate import tabulate
                print(tabulate(df, headers='keys', tablefmt='psql'))
            else:
                print(format_table(df))

    def generate_comparison_report(self, full_results, output_dir):
        """生成詳細的 Markdown 格式報告，包含標準差信息和圖片說明"""
        report_path = os.path.join(output_dir, 'model_comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 模型評估與穩健性比較報告\n\n")
            f.write("本報告對多個 Wi-Fi 室內定位模型在不同資料損壞情境下的效能進行了評估與比較，旨在測試模型的穩健性。\n\n")
            # f.write("**說明**：\n")
            # f.write("- **整體位置誤差**：所有測試樣本的位置預測誤差\n")
            # f.write("- **條件位置誤差**：只針對建築物和樓層都預測正確的樣本計算的位置誤差\n")
            # f.write("- **樓層準確率**：只針對建築物預測正確的樣本計算的樓層分類準確率\n")
            # f.write("- **多次測試**：每個情境進行 5 次獨立測試並報告平均值±標準差\n")
            # f.write("- **統一測試次數**：即使是原始資料也進行 5 次測試，以評估模型內部隨機性和數值穩定性\n\n")
            
            # # 添加視覺化圖表說明
            # f.write("## 📊 視覺化圖表說明\n\n")
            # f.write("本報告包含多種類型的圖表，以下是各類圖表的閱讀指南：\n\n")
            
            # f.write("### 🔹 基礎比較圖表\n\n")
            # f.write("1. **分類準確度對比圖** (`classification_accuracy_[情境].svg`)\n")
            # f.write("   - 顯示建築物和樓層分類的準確率\n")
            # f.write("   - 縱軸：準確率百分比\n")
            # f.write("   - 橫軸：不同模型\n")
            # f.write("   - 藍色柱狀：建築物分類準確率，橙色柱狀：樓層分類準確率\n")
            # f.write("   - 數值越高表示分類效果越好\n\n")
            
            # f.write("2. **位置預測誤差對比圖** (`position_errors_[情境].svg`)\n")
            # f.write("   - 顯示各模型的平均位置預測誤差\n")
            # f.write("   - 縱軸：誤差（公尺）\n")
            # f.write("   - 橫軸：不同模型\n")
            # f.write("   - 數值越低表示定位越精確\n\n")
            
            # f.write("### 🔹 誤差分布圖表\n\n")
            # f.write("3. **箱型圖** (`error_boxplot_[情境].svg`)\n")
            # f.write("   - 顯示誤差的統計分布特性\n")
            # f.write("   - 箱子：代表 25%-75% 分位數範圍（四分位距 IQR）\n")
            # f.write("   - 中線：中位數\n")
            # f.write("   - 虛線：平均值\n")
            # f.write("   - 觸鬚：延伸至 1.5×IQR 範圍\n")
            # f.write("   - 紅點：異常值（超出觸鬚範圍的樣本）\n")
            # f.write("   - 箱子越窄表示誤差分布越集中，異常值越少表示模型越穩定\n\n")
            
            # f.write("4. **小提琴圖** (`error_violin_[情境].svg`)\n")
            # f.write("   - 結合箱型圖和密度分布的優點\n")
            # f.write("   - 寬度：代表該誤差值的樣本密度\n")
            # f.write("   - 內部線條：中位數和四分位數\n")
            # f.write("   - 形狀：顯示誤差分布的詳細形態\n")
            # f.write("   - 對稱的「小提琴」形狀表示正態分布，不對稱則表示偏斜分布\n\n")
            
            # f.write("5. **累積分布函數（CDF）圖** (`error_cdf_[情境].svg`)\n")
            # f.write("   - 顯示達到特定誤差閾值的樣本百分比\n")
            # f.write("   - 橫軸：誤差值（公尺）\n")
            # f.write("   - 縱軸：累積百分比（%）\n")
            # f.write("   - 重要閾值：1m、2m、3m（用灰色虛線標示）\n")
            # f.write("   - 曲線越陡峭表示誤差越集中，左上角的曲線表示誤差越小\n")
            # f.write("   - 點狀標記：顯示在 1m、2m、3m 閾值下的精確百分比\n\n")
            
            # f.write("6. **詳細分布直方圖** (`error_detailed_distribution_[情境].svg`)\n")
            # f.write("   - 子圖形式顯示每個模型的誤差分布\n")
            # f.write("   - 藍色柱狀：誤差頻率分布\n")
            # f.write("   - 紅色虛線：平均值\n")
            # f.write("   - 橙色虛線：中位數\n")
            # f.write("   - 右上角文字：統計摘要（樣本數、標準差、90%分位數）\n")
            # f.write("   - 分布集中在左側表示大多數樣本誤差較小\n\n")
            
            # f.write("7. **統計總結表格圖** (`error_statistics_table_[情境].svg`)\n")
            # f.write("   - 以表格形式總結各模型的關鍵統計指標\n")
            # f.write("   - 平均值：所有樣本的平均誤差\n")
            # f.write("   - 中位數：排序後中間值的誤差\n")
            # f.write("   - 標準差：誤差分布的離散程度\n")
            # f.write("   - Q25/Q75：25% 和 75% 分位數\n")
            # f.write("   - P90：90% 分位數（表示 90% 的樣本誤差都小於此值）\n")
            # f.write("   - <1m/<2m/<3m：誤差小於指定閾值的樣本百分比\n\n")
            
            # f.write("### 🔹 穩健性分析圖表\n\n")
            # f.write("8. **跨情境穩健性測試圖**\n")
            # f.write("   - `robustness_building_accuracy.svg`：建築物分類在不同情境下的表現\n")
            # f.write("   - `robustness_floor_accuracy.svg`：樓層分類在不同情境下的表現\n")
            # f.write("   - `robustness_position_error.svg`：位置預測在不同情境下的表現\n")
            # f.write("   - 不同顏色柱狀代表不同模型\n")
            # f.write("   - 從左到右：原始資料→高斯雜訊→設備故障→複合干擾\n")
            # f.write("   - 觀察柱狀高度的變化程度可評估模型穩健性\n\n")
            
            # f.write("9. **穩健性評分圖** (`robustness_scores.svg`)\n")
            # f.write("   - 綜合評分，1.0 表示完美保持基準性能\n")
            # f.write("   - 評分越高表示在惡劣條件下性能保持越好\n")
            # f.write("   - 原始資料情境固定為 1.0（基準）\n")
            # f.write("   - 其他情境的評分反映相對於基準的性能保持率\n\n")
            
            # f.write("### 📈 如何解讀結果\n\n")
            # f.write("**選擇最佳模型的參考原則：**\n\n")
            # f.write("1. **準確率優先**：建築物和樓層分類準確率越高越好\n")
            # f.write("2. **誤差最小化**：位置預測誤差越小越好（特別關注條件位置誤差）\n")
            # f.write("3. **穩定性考量**：箱型圖中箱子越窄、異常值越少表示越穩定\n")
            # f.write("4. **穩健性要求**：在干擾情境下性能下降幅度越小越好\n")
            # f.write("5. **應用需求**：根據實際應用對準確率和精度的不同要求權衡選擇\n\n")
            
            # f.write("**異常情況識別：**\n\n")
            # f.write("- CDF 圖中曲線過於平緩：表示誤差分布過於分散\n")
            # f.write("- 箱型圖中異常值過多：表示模型預測不穩定\n")
            # f.write("- 穩健性評分急劇下降：表示模型對干擾敏感\n")
            # f.write("- 多峰分布（小提琴圖或直方圖）：可能存在系統性偏差\n\n")
            
            for scenario_name, results in full_results.items():
                if not results:
                    continue
                
                f.write(f"## 情境: {scenario_name}\n\n")
                
                # 添加情境特定的圖片說明
                scenario_clean = scenario_name.replace(" ", "_")
                # f.write(f"### 📸 相關視覺化圖表\n\n")
                # f.write(f"此情境的詳細分析圖表包括：\n\n")
                # f.write(f"- 📊 [分類準確度對比](./classification_accuracy_{scenario_clean}.svg)\n")
                # f.write(f"- 📈 [位置誤差對比](./position_errors_{scenario_clean}.svg)\n")
                # f.write(f"- 📦 [誤差箱型圖](./error_boxplot_{scenario_clean}.svg)\n")
                # f.write(f"- 🎻 [誤差密度分布](./error_violin_{scenario_clean}.svg)\n")
                # f.write(f"- 📉 [誤差累積分布](./error_cdf_{scenario_clean}.svg)\n")
                # f.write(f"- 📋 [詳細分布圖](./error_detailed_distribution_{scenario_clean}.svg)\n")
                # f.write(f"- 📑 [統計摘要表](./error_statistics_table_{scenario_clean}.svg)\n\n")
                
                # 根據情境提供特殊解讀建議
                if scenario_name == "原始資料":
                    f.write("**📝 解讀重點：**\n")
                    f.write("- 此為基準情境，展現各模型在理想條件下的最佳性能\n")
                    f.write("- 重點觀察各模型的絕對性能表現\n")
                    f.write("- 注意條件位置誤差與整體位置誤差的差異\n\n")
                elif "雜訊" in scenario_name:
                    f.write("**📝 解讀重點：**\n")
                    f.write("- 高斯雜訊模擬 Wi-Fi 信號的隨機擾動\n")
                    f.write("- 觀察各模型對信號噪音的抗干擾能力\n")
                    f.write("- 關注準確率下降幅度和誤差增加程度\n\n")
                    # 添加雜訊級別影響提示
                    noise_level = [int(s.replace('dB', '')) for s in scenario_name.split() if s.endswith('dB')][0] if any(s.endswith('dB') for s in scenario_name.split()) else 0
                    if noise_level > 0:
                        f.write(f"- 當前雜訊級別: {noise_level}dB (越高干擾越強)\n\n")
                elif "故障" in scenario_name:
                    f.write("**📝 解讀重點：**\n")
                    f.write("- 模擬 AP 設備故障或信號遮蔽情況\n")
                    f.write("- 評估模型在部分信息缺失時的補償能力\n")
                    f.write("- 高故障率（50%）模擬極端惡劣環境\n\n")
                elif "+" in scenario_name:
                    f.write("**📝 解讀重點：**\n")
                    f.write("- 複合干擾情境，同時存在雜訊和設備故障\n")
                    f.write("- 最具挑戰性的測試條件\n")
                    f.write("- 重點評估模型在多重壓力下的綜合表現\n\n")
                
                f.write("此情境下，各模型表現如下：\n\n")
                
                # 添加表格
                names = list(results.keys())
                building_accuracies = [results[name]['building_accuracy'] * 100 for name in names]
                building_stds = [results[name].get('building_accuracy_std', 0) * 100 for name in names]
                floor_accuracies = [results[name]['floor_accuracy'] * 100 for name in names]
                floor_stds = [results[name].get('floor_accuracy_std', 0) * 100 for name in names]
                mean_errors = [results[name]['position_mean_error'] for name in names]
                mean_error_stds = [results[name].get('position_mean_error_std', 0) for name in names]
                median_errors = [results[name]['position_median_error'] for name in names]
                std_errors = [results[name]['position_std_error'] for name in names]
                conditional_mean_errors = [results[name].get('conditional_position_mean_error', results[name]['position_mean_error']) for name in names]
                conditional_mean_error_stds = [results[name].get('conditional_position_mean_error_std', 0) for name in names]
                conditional_counts = [results[name].get('conditional_correct_count', len(results[name]['predictions']['building'])) for name in names]
                num_trials = [results[name].get('num_trials', 1) for name in names]
                
                # 格式化帶有標準差的值
                def format_with_std(mean, std, decimals=4):
                    if std > 0:
                        return f"{mean:.{decimals}f}±{std:.{decimals}f}"
                    else:
                        return f"{mean:.{decimals}f}"
                
                df = pd.DataFrame({
                    '模型名稱': names,
                    '建築物準確率 (%)': [format_with_std(acc, std, 4) for acc, std in zip(building_accuracies, building_stds)],
                    '樓層準確率 (%)': [format_with_std(acc, std, 4) for acc, std in zip(floor_accuracies, floor_stds)],
                    '條件位置平均誤差 (公尺)': [format_with_std(err, std, 4) for err, std in zip(conditional_mean_errors, conditional_mean_error_stds)],
                    '正確分類樣本數': conditional_counts,
                    '位置中位數誤差 (公尺)': [f"{err:.4f}" for err in median_errors],
                    '位置標準差 (公尺)': [f"{err:.4f}" for err in std_errors],
                    '測試次數': num_trials
                })
                
                if TABULATE_AVAILABLE:
                    from tabulate import tabulate
                    f.write(tabulate(df, headers='keys', tablefmt='github'))
                else:
                    f.write(format_table(df))
                
                f.write("\n")
                
            # 在報告末尾添加圖表文件清單
            f.write("## 📂 附錄：完整圖表清單\n\n")
            f.write("### 各情境專用圖表\n\n")
            
            scenario_names = list(full_results.keys())
            for scenario_name in scenario_names:
                scenario_clean = scenario_name.replace(" ", "_")
                f.write(f"**{scenario_name}：**\n")
                f.write(f"- `classification_accuracy_{scenario_clean}.svg` - 分類準確度對比\n")
                f.write(f"- `position_errors_{scenario_clean}.svg` - 位置誤差對比\n")
                f.write(f"- `error_boxplot_{scenario_clean}.svg` - 誤差箱型圖\n")
                f.write(f"- `error_violin_{scenario_clean}.svg` - 誤差小提琴圖\n")
                f.write(f"- `error_cdf_{scenario_clean}.svg` - 誤差累積分布函數\n")
                f.write(f"- `error_detailed_distribution_{scenario_clean}.svg` - 詳細誤差分布\n")
                f.write(f"- `error_statistics_table_{scenario_clean}.svg` - 統計摘要表格\n\n")
            
            f.write("### 跨情境分析圖表\n\n")
            f.write("- `robustness_building_accuracy.svg` - 建築物分類穩健性測試\n")
            f.write("- `robustness_floor_accuracy.svg` - 樓層分類穩健性測試\n")
            f.write("- `robustness_position_error.svg` - 位置預測穩健性測試\n")
            f.write("- `robustness_scores.svg` - 綜合穩健性評分\n\n")
            
            f.write("### 📊 建議的圖表查看順序\n\n")
            f.write("1. **快速概覽**：先查看各情境的分類準確度和位置誤差對比圖\n")
            f.write("2. **深入分析**：查看箱型圖和 CDF 圖了解誤差分布特性\n")
            f.write("3. **詳細檢視**：查看詳細分布圖和統計表格獲取具體數值\n")
            f.write("4. **穩健性評估**：查看跨情境圖表了解模型在不同條件下的表現\n")
            f.write("5. **綜合評分**：參考穩健性評分圖做出最終決策\n\n")
            
            f.write("---")
            
        print(f"比較報告已保存至: {report_path}")

    def generate_comparison_charts(self, full_results, output_dir):
        """生成並保存圖表，為每個情境創建獨立的圖表"""
        # 在開始前關閉所有圖表
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
                fig = plt.figure(figsize=(12, 8))
                df_acc.plot(kind='bar', width=0.4, align='center')
                plt.xlabel('模型')
                plt.ylabel('準確率 (%)')
                plt.title(f'不同模型的分類準確度對比 - {scenario_name}')
                plt.xticks(rotation=15)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'classification_accuracy_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
            finally:
                plt.close(fig)  # 明確關閉特定圖表

            # 位置誤差圖表
            try:
                fig = plt.figure(figsize=(12, 8))
                plt.bar(names, mean_errors, color='skyblue')
                plt.xlabel('模型')
                plt.ylabel('平均誤差 (公尺)')
                plt.title(f'不同模型的位置預測平均誤差對比 - {scenario_name}')
                plt.xticks(rotation=15)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'position_errors_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
            finally:
                plt.close(fig)  # 明確關閉特定圖表

            # 改進的位置誤差分布圖表
            self.generate_enhanced_error_distribution_charts(scenario_name, names, results, output_dir)

        # 生成跨情境比較圖表
        self.generate_cross_scenario_charts(full_results, output_dir)

    def generate_enhanced_error_distribution_charts(self, scenario_name, names, results, output_dir):
        """生成增強版的位置誤差分布圖表"""
        # 在開始前關閉所有圖表
        plt.close('all')
        
        # 準備數據
        all_errors = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, name in enumerate(names):
            errors = np.sqrt(np.sum((self.test_c - np.array(results[name]['predictions']['position']))**2, axis=1))
            all_errors[name] = errors

        # 1. 箱型圖 (Box Plot) - 顯示統計分布
        # try:
        #     fig = plt.figure(figsize=(14, 8))
        #     box_data = [all_errors[name] for name in names]
            
        #     box_plot = plt.boxplot(box_data, labels=names, patch_artist=True, 
        #                           showmeans=True, meanline=True,
        #                           flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
        #     # 為箱型圖著色
        #     for patch, color in zip(box_plot['boxes'], colors[:len(names)]):
        #         patch.set_facecolor(color)
        #         patch.set_alpha(0.7)
            
        #     plt.xlabel('模型')
        #     plt.ylabel('位置預測誤差 (公尺)')
        #     plt.title(f'位置預測誤差箱型圖 - {scenario_name}')
        #     plt.xticks(rotation=15)
        #     plt.grid(axis='y', linestyle='--', alpha=0.7)
            
        #     # 添加統計信息
        #     for i, name in enumerate(names):
        #         errors = all_errors[name]
        #         mean_err = np.mean(errors)
        #         median_err = np.median(errors)
        #         q75_err = np.percentile(errors, 75)
        #         q25_err = np.percentile(errors, 25)
                
        #         # 在圖上添加統計信息
        #         plt.text(i+1, plt.ylim()[1] * 0.95, 
        #                 f'平均: {mean_err:.3f}\n中位數: {median_err:.3f}\nIQR: {q75_err-q25_err:.3f}',
        #                 ha='center', va='top', fontsize=8,
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
            
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(output_dir, f'error_boxplot_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        # except Exception as e:
        #     print(f"箱型圖生成失敗: {e}")
        # finally:
        #     plt.close(fig)  # 明確關閉特定圖表

        # 2. 小提琴圖 (Violin Plot) - 顯示密度分布
        # try:
        #     fig = plt.figure(figsize=(14, 8))
        #     violin_parts = plt.violinplot(box_data, positions=range(1, len(names)+1), 
        #                                 showmeans=True, showmedians=True, showextrema=True)
            
        #     # 為小提琴圖著色
        #     for i, pc in enumerate(violin_parts['bodies']):
        #         pc.set_facecolor(colors[i % len(colors)])
        #         pc.set_alpha(0.7)
            
        #     plt.xticks(range(1, len(names)+1), names, rotation=15)
        #     plt.xlabel('模型')
        #     plt.ylabel('位置預測誤差 (公尺)')
        #     plt.title(f'位置預測誤差密度分布 - {scenario_name}')
        #     plt.grid(axis='y', linestyle='--', alpha=0.7)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(output_dir, f'error_violin_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        # except Exception as e:
        #     print(f"小提琴圖生成失敗: {e}")
        # finally:
        #     plt.close(fig)  # 明確關閉特定圖表

        # 3. 累積分布函數 (CDF) - 顯示誤差達到某閾值的百分比
        try:
            fig = plt.figure(figsize=(14, 8))
            
            for i, name in enumerate(names):
                errors = all_errors[name]
                sorted_errors = np.sort(errors)
                cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                
                plt.plot(sorted_errors, cumulative_prob * 100, 
                        label=name, color=colors[i % len(colors)], linewidth=2)
                
                # 添加關鍵點標記
                for threshold in [1.0, 2.0, 3.0]:
                    if threshold <= np.max(sorted_errors):
                        percentage = np.sum(errors <= threshold) / len(errors) * 100
                        idx = np.where(sorted_errors <= threshold)[0]
                        if len(idx) > 0:
                            plt.scatter(threshold, percentage, color=colors[i % len(colors)], 
                                      s=50, zorder=5, alpha=0.8)
        
            # 添加參考線
            for threshold in [1.0, 2.0, 3.0]:
                plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
                plt.text(threshold, 5, f'{threshold}m', rotation=90, va='bottom', ha='right', fontsize=9)
            
            plt.xlabel('位置預測誤差 (公尺)')
            plt.ylabel('累積百分比 (%)')
            plt.title(f'位置預測誤差累積分布函數 - {scenario_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(left=0)
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'error_cdf_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"CDF圖生成失敗: {e}")
        finally:
            plt.close(fig)  # 明確關閉特定圖表

        # 4. 改進的直方圖 - 更清晰的分布顯示
        try:
            fig = plt.figure(figsize=(14, 10))
            
            # 使用子圖分別顯示每個模型
            n_models = len(names)
            rows = (n_models + 1) // 2  # 每行最多2個
            cols = min(n_models, 2)
            
            for i, name in enumerate(names):
                plt.subplot(rows, cols, i + 1)
                errors = all_errors[name]
                
                # 計算最佳bin數量
                n_bins = min(30, max(10, int(np.sqrt(len(errors)))))


                # 使用統計量計算 bin 邊界
                bin_width = 2 * (np.percentile(errors, 75) - np.percentile(errors, 25)) / (len(errors) ** (1/3))  # Freedman-Diaconis rule
                bins = np.arange(0, np.max(errors) + bin_width, bin_width)
                
                n, bins, patches = plt.hist(errors, bins=bins, alpha=0.7, 
                                       color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
                
                # 添加統計線
                mean_err = np.mean(errors)
                median_err = np.median(errors)
                
                plt.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_err:.3f}m')
                plt.axvline(median_err, color='orange', linestyle='--', linewidth=2, label=f'中位數: {median_err:.3f}m')
                
                plt.xlabel('誤差 (公尺)')
                plt.ylabel('樣本數')
                plt.title(f'{name}')
                plt.legend(fontsize=8)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                # 添加統計信息文字
                stats_text = f'樣本數: {len(errors)}\n標準差: {np.std(errors):.3f}m\n90%分位數: {np.percentile(errors, 90):.3f}m'
                plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8)
            
            plt.suptitle(f'各模型位置預測誤差詳細分布 - {scenario_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'error_detailed_distribution_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"詳細分布圖生成失敗: {e}")
        finally:
            plt.close(fig)  # 明確關閉特定圖表

        # 5. 誤差統計總結表格圖（修正字體問題）
        try:
            fig = plt.figure(figsize=(12, 6))
            plt.axis('off')  # 隱藏軸
            
            # 準備統計數據，避免使用特殊符號
            stats_data = []
            for name in names:
                errors = all_errors[name]
                stats_row = [
                    name,
                    f"{np.mean(errors):.3f}",
                    f"{np.median(errors):.3f}",
                    f"{np.std(errors):.3f}",
                    f"{np.percentile(errors, 25):.3f}",
                    f"{np.percentile(errors, 75):.3f}",
                    f"{np.percentile(errors, 90):.3f}",
                    f"{np.sum(errors <= 1.0)/len(errors)*100:.1f}",
                    f"{np.sum(errors <= 2.0)/len(errors)*100:.1f}",
                    f"{np.sum(errors <= 3.0)/len(errors)*100:.1f}"
                ]
                stats_data.append(stats_row)
            
            # 修改表格標題，避免使用 ≤ 符號
            headers = ['模型', '平均值', '中位數', '標準差', 'Q25', 'Q75', 'P90', '<1m', '<2m', '<3m']
            
            # 創建表格
            table = plt.table(cellText=stats_data, colLabels=headers, 
                             cellLoc='center', loc='center',
                             colColours=['lightgray']*len(headers))
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)  # 調整表格大小
            
            # 為不同模型行著色
            for i in range(len(stats_data)):
                for j in range(len(headers)):
                    if j == 0:  # 模型名稱列
                        table[(i+1, j)].set_facecolor(colors[i % len(colors)])
                        table[(i+1, j)].set_alpha(0.3)
            
            plt.title(f'位置預測誤差統計總結 - {scenario_name}', fontsize=14, pad=20)
            plt.tight_layout()
            
            # 使用 try-except 處理保存過程中的字體問題
            try:
                plt.savefig(os.path.join(output_dir, f'error_statistics_table_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
            except Exception as font_error:
                print(f"表格保存時字體問題: {font_error}，嘗試使用基本字體")
                # 重設字體為更基本的選項
                plt.rcParams['font.family'] = ['Arial', 'sans-serif']
                plt.savefig(os.path.join(output_dir, f'error_statistics_table_{scenario_name.replace(" ", "_")}.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"統計表格圖生成失敗: {e}")
        finally:
            plt.close(fig)  # 明確關閉特定圖表

    def generate_cross_scenario_charts(self, full_results, output_dir):
        """生成跨情境的比較圖表"""
        # 在開始前關閉所有圖表
        plt.close('all')
        
        scenarios = list(full_results.keys())
        all_models = set()
        for results in full_results.values():
            all_models.update(results.keys())
        all_models = sorted(list(all_models))

        # 穩健性測試 - 建築物準確率
        try:
            fig = plt.figure(figsize=(15, 10))
            scenario_data = {}
            for model in all_models:
                accuracies = []
                for scenario in scenarios:
                    if model in full_results[scenario]:
                        accuracies.append(full_results[scenario][model]['building_accuracy'] * 100)
                    else:
                        accuracies.append(0)
                scenario_data[model] = accuracies

            x = np.arange(len(scenarios))
            width = 0.15
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (model, accuracies) in enumerate(scenario_data.items()):
                plt.bar(x + i * width, accuracies, width, label=model, color=colors[i % len(colors)])

            plt.xlabel('測試情境')
            plt.ylabel('建築物分類準確率 (%)')
            plt.title('不同情境下的模型穩健性測試 - 建築物分類')
            plt.xticks(x + width * (len(all_models) - 1) / 2, scenarios, rotation=15)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_building_accuracy.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"建築物穩健性圖生成失敗: {e}")
        finally:
            plt.close(fig)  # 明確關閉特定圖表

        # 穩健性測試 - 樓層準確率
        try:
            fig = plt.figure(figsize=(15, 10))
            for i, (model, _) in enumerate(scenario_data.items()):
                floor_accuracies = []
                for scenario in scenarios:
                    if model in full_results[scenario]:
                        floor_accuracies.append(full_results[scenario][model]['floor_accuracy'] * 100)
                    else:
                        floor_accuracies.append(0)
                plt.bar(x + i * width, floor_accuracies, width, label=model, color=colors[i % len(colors)])

            plt.xlabel('測試情境')
            plt.ylabel('樓層分類準確率 (%)')
            plt.title('不同情境下的模型穩健性測試 - 樓層分類')
            plt.xticks(x + width * (len(all_models) - 1) / 2, scenarios, rotation=15)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_floor_accuracy.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"樓層穩健性圖生成失敗: {e}")
        finally:
            plt.close(fig)  # 明確關閉特定圖表

        # 穩健性測試 - 位置誤差
        try:
            fig = plt.figure(figsize=(15, 10))
            for i, (model, _) in enumerate(scenario_data.items()):
                position_errors = []
                for scenario in scenarios:
                    if model in full_results[scenario]:
                        position_errors.append(full_results[scenario][model]['position_mean_error'])
                    else:
                        position_errors.append(float('inf'))
                plt.bar(x + i * width, position_errors, width, label=model, color=colors[i % len(colors)])

            plt.xlabel('測試情境')
            plt.ylabel('位置預測平均誤差 (公尺)')
            plt.title('不同情境下的模型穩健性測試 - 位置預測')
            plt.xticks(x + width * (len(all_models) - 1) / 2, scenarios, rotation=15)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_position_error.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"位置穩健性圖生成失敗: {e}")
        finally:
            plt.close(fig)  # 明確關閉特定圖表

        # 模型穩健性評分圖表
        self.generate_robustness_score_chart(full_results, output_dir, scenarios, all_models)

    def generate_robustness_score_chart(self, full_results, output_dir, scenarios, all_models):
        """生成模型穩健性評分"""
        # 計算基準情境（通常為 '原始資料'）的性能
        baseline_scenario = scenarios[0] if scenarios else None
        robustness_scores = {}

        if baseline_scenario is None:
            print("沒有基準情境，無法計算穩健性評分。")
            return

        for model in all_models:
            scores = []
            if model not in full_results[baseline_scenario]:
                # 如果基準情境沒有該模型，全部設為0
                scores = [0.0 for _ in scenarios]
            else:
                baseline_building_acc = full_results[baseline_scenario][model]['building_accuracy']
                baseline_floor_acc = full_results[baseline_scenario][model]['floor_accuracy']
                baseline_position_error = full_results[baseline_scenario][model]['position_mean_error']
                for scenario in scenarios:
                    if model in full_results[scenario]:
                        building_acc = full_results[scenario][model]['building_accuracy']
                        floor_acc = full_results[scenario][model]['floor_accuracy']
                        position_error = full_results[scenario][model]['position_mean_error']
                        # 穩健性評分：分類準確率相對保持率和位置誤差相對保持率的加權平均
                        # 位置誤差越低越好，因此用基準/當前
                        score = (
                            0.4 * (building_acc / baseline_building_acc if baseline_building_acc > 0 else 0) +
                            0.3 * (floor_acc / baseline_floor_acc if baseline_floor_acc > 0 else 0) +
                            0.3 * (baseline_position_error / position_error if position_error > 0 else 0)
                        )
                        # 限制最大值為1.0
                        score = min(score, 1.0)
                        scores.append(score)
                    else:
                        scores.append(0.0)
            robustness_scores[model] = scores

        # 繪製穩健性評分圖表
        try:
            plt.figure(figsize=(15, 10))
            width = 0.15
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            x = np.arange(len(scenarios))  # 定義 x 為情境索引

            for i, (model, scores) in enumerate(robustness_scores.items()):
                plt.bar(x + i * width, scores, width, label=model, color=colors[i % len(colors)])

            plt.xlabel('測試情境')
            plt.ylabel('穩健性評分 (1.0=基準)')
            plt.title('不同情境下的模型穩健性評分')
            plt.xticks(x + width * (len(robustness_scores) - 1) / 2, scenarios, rotation=15)
            plt.ylim(0, 1.05)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_scores.svg'), format='svg', bbox_inches='tight')
        except Exception as e:
            print(f"穩健性評分圖生成失敗: {e}")
        finally:
            plt.close()

    # 新增：統一計算穩健性分數（供圖與摘要共用）
    def compute_robustness_scores(self, full_results):
        """
        回傳:
          baseline_scenario: 基準情境名稱
          scenarios: 情境列表（有序）
          all_models: 模型名稱列表（排序）
          robustness_scores: dict[model] -> list[score per scenario]
        """
        scenarios = list(full_results.keys())
        if not scenarios:
            return None, [], [], {}

        # 優先使用「原始資料」為基準，否則取第一個
        baseline_scenario = '原始資料' if '原始資料' in scenarios else scenarios[0]

        # 蒐集所有模型
        all_models = set()
        for results in full_results.values():
            all_models.update(results.keys())
        all_models = sorted(list(all_models))

        robustness_scores = {}
        for model in all_models:
            scores = []
            # 若基準沒有該模型，後續皆為 0
            if model not in full_results.get(baseline_scenario, {}):
                scores = [0.0 for _ in scenarios]
            else:
                baseline_building_acc = full_results[baseline_scenario][model]['building_accuracy']
                baseline_floor_acc = full_results[baseline_scenario][model]['floor_accuracy']
                baseline_position_error = full_results[baseline_scenario][model]['position_mean_error']
                for scenario in scenarios:
                    if model in full_results[scenario]:
                        building_acc = full_results[scenario][model]['building_accuracy']
                        floor_acc = full_results[scenario][model]['floor_accuracy']
                        position_error = full_results[scenario][model]['position_mean_error']
                        score = (
                            0.4 * (building_acc / baseline_building_acc if baseline_building_acc > 0 else 0) +
                            0.3 * (floor_acc / baseline_floor_acc if baseline_floor_acc > 0 else 0) +
                            0.3 * (baseline_position_error / position_error if position_error > 0 else 0)
                        )
                        score = min(score, 1.0)
                        scores.append(score)
                    else:
                        scores.append(0.0)
            robustness_scores[model] = scores

        return baseline_scenario, scenarios, all_models, robustness_scores

    # 新增：輸出穩健性摘要 Markdown
    def generate_robustness_summary(self, full_results, output_dir):
        """
        生成 robustness_summary.md，包含：
        - 評分標準與測試方法
        - 模型穩健性排名（平均分數）
        - 精選情境分數表
        - 關鍵發現與時間戳
        """
        baseline_scenario, scenarios, all_models, robustness_scores = self.compute_robustness_scores(full_results)
        if not scenarios:
            print("沒有情境可生成穩健性摘要。")
            return

        # 取得測試次數（從任一結果取出）
        try:
            any_result = next(iter(next(iter(full_results.values())).values()))
            num_trials = any_result.get('num_trials', 1)
        except Exception:
            num_trials = 1

        # 計算每個模型的平均分數（排除基準情境）
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

        # 排名（由高到低）
        ranking = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)

        # 精選情境（若不存在則忽略）
        preferred = ['原始資料', '高斯雜訊 5dB', '設備故障 10%', '設備故障 35%', '雜訊 5dB + 故障 10%', '雜訊 10dB + 故障 20%']
        selected_scenarios = [s for s in preferred if s in scenarios]
        if not selected_scenarios:
            selected_scenarios = scenarios[:min(6, len(scenarios))]

        # 計算每個情境的平均分數（用於找出最具挑戰性情境，排除基準）
        scenario_avg = {}
        for i, sc in enumerate(scenarios):
            vals = [robustness_scores[m][i] for m in all_models if len(robustness_scores[m]) > i]
            scenario_avg[sc] = float(np.mean(vals)) if vals else 0.0
        toughest_scenario = min(
            ((sc, v) for sc, v in scenario_avg.items() if sc != baseline_scenario),
            key=lambda x: x[1],
            default=(scenarios[0], scenario_avg.get(scenarios[0], 0.0))
        )

        # 檔案輸出
        out_path = os.path.join(output_dir, 'robustness_summary.md')
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("# 模型穩健性評分摘要\n\n")
            f.write("本摘要基於不同資料損壞情境下的模型性能保持率計算穩健性評分。\n\n")
            f.write("**評分標準**：\n")
            f.write("- 1.0：完美保持基準性能\n")
            f.write("- 0.8-1.0：優秀的穩健性\n")
            f.write("- 0.6-0.8：良好的穩健性\n")
            f.write("- 0.4-0.6：一般的穩健性\n")
            f.write("- <0.4：較差的穩健性\n\n")
            f.write("**測試方法**：\n")
            f.write(f"- 每個情境進行 {num_trials} 次獨立測試取平均值\n")
            f.write("- 原始資料的多次測試用於評估模型內部隨機性和數值穩定性\n")
            f.write("- 損壞情境的多次測試用於獲得更可靠的穩健性評估\n\n")
            f.write("- **測試方法說明**：所有情境都進行相同次數的測試，確保統計結果的可靠性\n")
            f.write("  - 原始資料：評估模型預測的一致性和數值穩定性\n")
            f.write("  - 損壞情境：評估在不同隨機損壞模式下的平均性能\n")

            f.write("## 模型穩健性排名\n\n")
            for i, (m, avg) in enumerate(ranking, start=1):
                level = "需改進"
                f.write(f"{i}. **{m}**：平均穩健性評分 {avg:.3f} ({level})\n")
            f.write("\n")

            f.write("## 各情境詳細評分\n\n")
            # 表頭
            f.write("|                | " + " | ".join([f"{sc}" for sc in selected_scenarios]) + " |\n")
            f.write("|:---------------|" + "|".join([":" + "-"*(max(3, len(sc))) for sc in selected_scenarios]) + "|\n")
            # 各模型列
            for m in all_models:
                f.write(f"| {m} ")
                for sc in selected_scenarios:
                    idx = scenarios.index(sc)
                    val = robustness_scores[m][idx] if len(robustness_scores[m]) > idx else 0.0
                    f.write(f"| {val:.3f} ")
                f.write("|\n")

            f.write("\n## 關鍵發現\n\n")
            if ranking:
                f.write(f"- 最穩健模型：{ranking[0][0]}（平均評分：{ranking[0][1]:.3f}）\n")
            f.write(f"- 最具挑戰性情境：{toughest_scenario[0]}（平均評分：{toughest_scenario[1]:.3f}）\n")
            f.write("- 統計可靠性：多次測試可降低單次隨機性的影響，提高結論可信度\n\n")
            f.write("---\n")
            f.write(f"*報告生成時間：{now_str}*\n")
            f.write(f"*測試設定：每情境 {num_trials} 次獨立測試取平均*\n")

        print(f"穩健性摘要已生成至: {out_path}")
        
def main():
    """主函數"""
    print("=== 開始模型比較和穩健性測試 ===")
    
    comparator = ModelComparison()
    comparator.compare_models()
    
    print("=== 比較和測試完成 ===")

if __name__ == '__main__':
    main()