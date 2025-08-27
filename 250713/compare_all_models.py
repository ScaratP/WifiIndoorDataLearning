import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error
import importlib

# 檢查是否安裝了 tabulate 套件
TABULATE_AVAILABLE = importlib.util.find_spec("tabulate") is not None

# --- 解決 matplotlib 中文字體問題的修改 ---
try:
    plt.rcParams['font.family'] = ['Microsoft JhengHei']  # Windows 系統
    plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號
except KeyError:
    print("警告: 找不到指定的繁體中文字體，請更換為系統中存在的字體。")
    pass
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
                    'name': 'hadnn+rf tflite',
                    'path': './models/hadnn_n_random_forest.tflite',
                    'type': 'tflite'
                },
                {
                    'name': 'mlp tflite',
                    'path': './models/mlp.tflite',
                    'type': 'tflite'
                },
                {
                    'name': 'original hadnn tflite',
                    'path': './models/original_hadnn.tflite',
                    'type': 'tflite'
                }
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
            
    def simulate_data_corruption(self, noise_level=0, missing_rate=0):
        """
        模擬資料損壞，包括增加高斯雜訊和隨機移除資料點。
        
        參數:
            noise_level (float): 高斯雜訊的標準差 (dB)。
            missing_rate (float): 資料遺失的百分比 (0-1)。
            
        回傳:
            np.array: 經模擬損壞後的測試資料。
        """
        corrupted_test_x = self.test_x.copy()
        
        # 1. 增加高斯雜訊
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, corrupted_test_x.shape)
            corrupted_test_x = corrupted_test_x + noise
            # 確保 RSSI 值保持在合理範圍內
            corrupted_test_x = np.clip(corrupted_test_x, -120, -30)
            
        # 2. 模擬資料遺失
        if missing_rate > 0:
            num_missing = int(np.prod(corrupted_test_x.shape) * missing_rate)
            missing_indices = np.random.choice(corrupted_test_x.size, num_missing, replace=False)
            corrupted_test_x.flat[missing_indices] = -120  # 用一個極端值代表遺失
            
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
                
                # 處理輸入資料
                input_tensor = input_data.astype(input_details[0]['dtype'])
                
                building_preds_list = []
                floor_preds_list = []
                position_preds_list = []
                
                # 逐個樣本進行推論
                for i in range(input_tensor.shape[0]):
                    interpreter.set_tensor(input_details[0]['index'], input_tensor[i:i+1])
                    interpreter.invoke()
                    
                    # 取得輸出結果
                    building_output = interpreter.get_tensor(output_details[0]['index'])
                    floor_output = interpreter.get_tensor(output_details[1]['index'])
                    position_output = interpreter.get_tensor(output_details[2]['index'])
                    
                    building_preds_list.append(np.argmax(building_output))
                    floor_preds_list.append(np.argmax(floor_output))
                    
                    # 修正：確保位置預測的維度是2，以避免誤差計算失敗
                    position_preds_list.append(position_output.flatten()[:2])

                building_preds = np.array(building_preds_list)
                floor_preds = np.array(floor_preds_list)
                position_preds = np.array(position_preds_list)

            else:
                print(f"不支援的模型類型: {model_type}")
                return None
            
            building_accuracy = accuracy_score(self.test_b, building_preds)
            try:
                if hasattr(self, 'test_f'):
                    floor_accuracy = accuracy_score(self.test_f, floor_preds)
                else:
                    floor_accuracy = accuracy_score(self.test_y, floor_preds)
            except:
                floor_accuracy = 0
            
            position_mse = mean_squared_error(self.test_c, position_preds)
            position_rmse = np.sqrt(position_mse)
            
            euclidean_distances = np.sqrt(np.sum((self.test_c - position_preds)**2, axis=1))
            mean_error = np.mean(euclidean_distances)
            median_error = np.median(euclidean_distances)
            
            result = {
                'building_accuracy': building_accuracy,
                'floor_accuracy': floor_accuracy,
                'position_mean_error': mean_error,
                'position_median_error': median_error,
                'position_rmse': position_rmse,
                'predictions': {
                    'building': building_preds.tolist(),
                    'floor': floor_preds.tolist(),
                    'position': position_preds.tolist()
                }
            }
            
            print("模型評估完成:")
            print(f"  建築物分類準確率: {result['building_accuracy'] * 100:.2f}%")
            print(f"  樓層分類準確率: {result['floor_accuracy'] * 100:.2f}%")
            print(f"  位置預測平均誤差: {result['position_mean_error']:.4f}")
            print(f"  位置預測中位數誤差: {result['position_median_error']:.4f}")
            
            return result
        except Exception as e:
            print(f"評估模型 {name} 失敗: {e}")
            return None
    
    def compare_models(self):
        """執行所有模型的比較，並加入穩健性測試"""
        if not self.load_test_data():
            print("無法執行比較，因為測試資料載入失敗。")
            return
            
        # 定義穩健性測試情境
        robustness_scenarios = {
            '原始資料': {'noise': 0, 'missing_rate': 0},
            '高斯雜訊 5dB': {'noise': 5, 'missing_rate': 0},
            '設備故障 10%': {'noise': 0, 'missing_rate': 0.1},
            '設備故障 50%': {'noise': 0, 'missing_rate': 0.5},
            '雜訊 5dB + 故障 10%': {'noise': 5, 'missing_rate': 0.1}
        }
        
        full_results = {}
        
        for scenario_name, params in robustness_scenarios.items():
            print(f"\n--- 執行情境: {scenario_name} ---")
            
            # 模擬資料損壞
            corrupted_test_x = self.simulate_data_corruption(
                noise_level=params['noise'],
                missing_rate=params['missing_rate']
            )
            
            # 應用與原始資料相同的預處理步驟
            from original_hadnn import advanced_data_preprocessing
            corrupted_test_x_enhanced, _ = advanced_data_preprocessing(corrupted_test_x)
            
            self.results = {}  # 重置結果
            
            # 對每個模型進行評估
            for model_info in self.models_to_compare:
                result = self.load_and_evaluate_model(model_info, input_data=corrupted_test_x_enhanced)
                if result is not None:
                    self.results[model_info['name']] = result
            
            full_results[scenario_name] = self.results
        
        # 顯示並生成最終報告
        self.display_comparison(full_results)
        output_dir = './model_comparison'
        os.makedirs(output_dir, exist_ok=True)
        self.generate_comparison_report(full_results, output_dir)
        self.generate_comparison_charts(full_results, output_dir)
        
        print(f"比較報告已生成至: {output_dir}")
        
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
            building_accuracies = [results[name]['building_accuracy'] * 100 for name in names]
            floor_accuracies = [results[name]['floor_accuracy'] * 100 for name in names]
            mean_errors = [results[name]['position_mean_error'] for name in names]
            median_errors = [results[name]['position_median_error'] for name in names]
            
            df = pd.DataFrame({
                '模型名稱': names,
                '建築物準確率 (%)': building_accuracies,
                '樓層準確率 (%)': floor_accuracies,
                '位置平均誤差 (米)': mean_errors,
                '位置中位數誤差 (米)': median_errors
            })
            
            df['建築物準確率 (%)'] = df['建築物準確率 (%)'].map('{:.2f}'.format)
            df['樓層準確率 (%)'] = df['樓層準確率 (%)'].map('{:.2f}'.format)
            df['位置平均誤差 (米)'] = df['位置平均誤差 (米)'].map('{:.4f}'.format)
            df['位置中位數誤差 (米)'] = df['位置中位數誤差 (米)'].map('{:.4f}'.format)
            
            if TABULATE_AVAILABLE:
                from tabulate import tabulate
                print(tabulate(df, headers='keys', tablefmt='psql'))
            else:
                print(format_table(df))
            
    def generate_comparison_report(self, full_results, output_dir):
        """生成詳細的 Markdown 格式報告，包含所有情境"""
        report_path = os.path.join(output_dir, 'model_comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 模型評估與穩健性比較報告\n\n")
            f.write("本報告對多個 Wi-Fi 室內定位模型在不同資料損壞情境下的效能進行了評估與比較，旨在測試模型的穩健性。\n\n")
            
            for scenario_name, results in full_results.items():
                if not results:
                    continue
                
                f.write(f"## 情境: {scenario_name}\n\n")
                f.write("此情境下，各模型表現如下：\n\n")
                
                names = list(results.keys())
                building_accuracies = [results[name]['building_accuracy'] * 100 for name in names]
                floor_accuracies = [results[name]['floor_accuracy'] * 100 for name in names]
                mean_errors = [results[name]['position_mean_error'] for name in names]
                median_errors = [results[name]['position_median_error'] for name in names]
                
                df = pd.DataFrame({
                    '模型名稱': names,
                    '建築物準確率 (%)': building_accuracies,
                    '樓層準確率 (%)': floor_accuracies,
                    '位置平均誤差 (米)': mean_errors,
                    '位置中位數誤差 (米)': median_errors
                })
                
                f.write(df.to_markdown(index=False, floatfmt=".2f"))
                f.write("\n\n")
                
                # 找出並記錄每個情境下的最佳模型
                best_building_acc_model = names[np.argmax(building_accuracies)]
                best_floor_acc_model = names[np.argmax(floor_accuracies)]
                best_pos_error_model = names[np.argmin(mean_errors)]
                f.write(f"- **建築物準確率最高**: {best_building_acc_model}\n")
                f.write(f"- **樓層準確率最高**: {best_floor_acc_model}\n")
                f.write(f"- **位置誤差最低**: {best_pos_error_model}\n\n")
                
            f.write("---")
            
        print(f"比較報告已保存至: {report_path}")

    def generate_comparison_charts(self, full_results, output_dir):
        """生成並保存圖表，為每個情境創建獨立的圖表"""
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
            
            plt.figure(figsize=(10, 6))
            df_acc.plot(kind='bar', figsize=(12, 8), width=0.4, align='center')
            plt.xlabel('模型')
            plt.ylabel('準確率 (%)')
            plt.title(f'不同模型的分類準確度對比 - {scenario_name}')
            plt.xticks(rotation=15)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'classification_accuracy_{scenario_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 位置誤差圖表
            plt.figure(figsize=(10, 6))
            plt.bar(names, mean_errors, color='skyblue')
            plt.xlabel('模型')
            plt.ylabel('平均誤差 (米)')
            plt.title(f'不同模型的位置預測平均誤差對比 - {scenario_name}')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'position_errors_{scenario_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 誤差分佈圖表
            plt.figure(figsize=(12, 8))
            for i, name in enumerate(names):
                errors = np.sqrt(np.sum((self.test_c - results[name]['predictions']['position'])**2, axis=1))
                plt.hist(errors, bins=30, alpha=0.7, label=name)
            
            plt.xlabel('誤差 (米)')
            plt.ylabel('樣本數')
            plt.title(f'不同模型的位置預測誤差分布 - {scenario_name}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'error_distribution_{scenario_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
def main():
    """主函數"""
    print("=== 開始模型比較和穩健性測試 ===")
    
    comparator = ModelComparison()
    comparator.compare_models()
    
    print("=== 比較和測試完成 ===")

if __name__ == '__main__':
    main()