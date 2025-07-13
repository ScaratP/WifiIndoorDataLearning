import os
import sys
import numpy as np
import tensorflow as tf
import json
import glob
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# 導入 TensorFlow Lite 解釋器
from tensorflow.lite.python.interpreter import Interpreter

# 導入自定義層和必要函數
try:
    from mlp import AttentionLayer, advanced_data_preprocessing, MLPPositionModel
    from hadnn_n_random_forest import AttentionLayer as HadnnAttentionLayer
except ImportError:
    print("警告: 未能導入必要的模型模塊。確保 mlp.py 和 hadnn_n_random_forest.py 在同一目錄下。")
    
    # 定義備用的 AttentionLayer 類別
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)
            self.units = units
            
        def build(self, input_shape):
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
        
        def call(self, inputs):
            score = tf.nn.tanh(tf.matmul(inputs, self.W))
            attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
            context_vector = attention_weights * inputs
            return context_vector, attention_weights
            
        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config
    
    # 定義備用的資料預處理函數
    def advanced_data_preprocessing(rss_data):
        """簡化版的資料預處理"""
        # 替換缺失值
        missing_mask = (rss_data == -100)
        replacement_value = -95  # 一般 RSS 低值
        processed_data = np.copy(rss_data)
        processed_data[missing_mask] = replacement_value
        
        # 簡單的特徵縮放
        min_val = -100
        max_val = -30
        scaled_data = (processed_data - min_val) / (max_val - min_val)
        
        # 限制範圍在 [0, 1]
        scaled_data = np.clip(scaled_data, 0, 1)
        
        return scaled_data, None

class ModelPredictor:
    """模型預測器類別，用於載入多個模型並進行預測"""
    
    def __init__(self, models_dir="./models", config_dir="./hadnn_data", processed_data_dir="../processed_data"):
        self.models_dir = models_dir
        self.config_dir = config_dir
        self.processed_data_dir = processed_data_dir
        self.models = {}
        self.tflite_models = {}  # 存放 TFLite 解釋器
        self.bssid_list = []
        self.config = {}
        
        # 載入配置和 BSSID 列表
        self.load_config()
        
    def load_config(self):
        """載入配置檔案和 BSSID 列表"""
        config_path = os.path.join(self.config_dir, 'dataset_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                print(f"已載入資料集配置檔案: {config_path}")
                
        # 嘗試從 bssid_mapping.csv 載入 BSSID 列表
        bssid_mapping_path = os.path.join(self.processed_data_dir, 'bssid_mapping.csv')
        if os.path.exists(bssid_mapping_path):
            try:
                # 使用 pandas 讀取 CSV 檔案
                bssid_df = pd.read_csv(bssid_mapping_path)
                # 提取 bssid 欄位並轉換為小寫的列表
                self.bssid_list = [bssid.lower() for bssid in bssid_df['bssid']]
                print(f"已從 CSV 檔案載入 BSSID 列表，共有 {len(self.bssid_list)} 個 AP")
            except Exception as e:
                print(f"從 CSV 檔案讀取 BSSID 列表時出錯: {e}")
                self.bssid_list = []
        else:
            # 向後兼容：嘗試載入 bssid.json
            bssid_path = os.path.join(self.config_dir, 'bssid.json')
            if os.path.exists(bssid_path):
                with open(bssid_path, 'r', encoding='utf-8') as f:
                    self.bssid_list = json.load(f)
                    print(f"已從 JSON 檔案載入 BSSID 列表，共有 {len(self.bssid_list)} 個 AP")
            else:
                print("警告: 找不到 BSSID 列表檔案 (既不是 CSV 也不是 JSON)")
                self.bssid_list = []

    def load_models(self):
        """載入 models 目錄中所有可用的模型"""
        print(f"載入目錄 {self.models_dir} 中的模型...")
        
        # 自定義對象字典
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        # 搜索所有 .h5 模型檔案
        h5_files = glob.glob(os.path.join(self.models_dir, '*.h5'))
        
        for model_file in h5_files:
            model_name = os.path.basename(model_file).replace('.h5', '')
            try:
                print(f"載入模型: {model_file}")
                model = load_model(model_file, custom_objects=custom_objects)
                self.models[model_name] = model
                print(f"  模型載入成功: {model_name}")
            except Exception as e:
                print(f"  載入模型 {model_name} 時出錯: {e}")
        
        # 特別檢查 mlp.h5 和 hadnn_n_random_forest.h5
        special_models = ['mlp', 'hadnn_n_random_forest']
        for model_name in special_models:
            model_file = os.path.join(self.models_dir, f'{model_name}.h5')
            if os.path.exists(model_file) and model_name not in self.models:
                try:
                    if model_name == 'mlp':
                        # 使用 MLPPositionModel 載入
                        model = MLPPositionModel.load(self.models_dir)
                        self.models[model_name] = model
                        print(f"  使用專用類別載入成功: {model_name}")
                    # 可以添加其他特殊模型的載入方式
                except Exception as e:
                    print(f"  使用專用類別載入 {model_name} 時出錯: {e}")
        
        # 載入 TensorFlow Lite 模型
        self.load_tflite_models()
        
        if not self.models and not self.tflite_models:
            print("警告: 沒有找到任何可用的模型!")
            return False
            
        print(f"成功載入 {len(self.models)} 個 TensorFlow 模型和 {len(self.tflite_models)} 個 TFLite 模型")
        return True
    
    def load_tflite_models(self):
        """載入 models 目錄中所有可用的 TensorFlow Lite 模型"""
        tflite_files = glob.glob(os.path.join(self.models_dir, '*.tflite'))
        
        for tflite_file in tflite_files:
            model_name = os.path.basename(tflite_file).replace('.tflite', '')
            try:
                print(f"載入 TFLite 模型: {tflite_file}")
                # 建立 TFLite 解釋器並分配張量
                interpreter = Interpreter(model_path=tflite_file)
                interpreter.allocate_tensors()
                
                # 獲取輸入和輸出細節
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # 存儲解釋器和詳細資訊
                self.tflite_models[model_name] = {
                    'interpreter': interpreter,
                    'input_details': input_details,
                    'output_details': output_details
                }
                
                print(f"  TFLite 模型載入成功: {model_name}")
                print(f"  輸入形狀: {input_details[0]['shape']}")
                print(f"  輸出數量: {len(output_details)}")
                
            except Exception as e:
                print(f"  載入 TFLite 模型 {model_name} 時出錯: {e}")
    
    def predict_tflite(self, model_info, input_data):
        """使用 TensorFlow Lite 模型進行預測"""
        interpreter = model_info['interpreter']
        input_details = model_info['input_details']
        output_details = model_info['output_details']
        
        # 確保輸入數據格式正確
        input_shape = input_details[0]['shape']
        if input_data.shape != (1, input_shape[1]):
            print(f"警告: 輸入形狀不匹配，調整大小從 {input_data.shape} 到 {(1, input_shape[1])}")
            
            # 如果輸入形狀不匹配，嘗試調整
            if input_data.shape[1] > input_shape[1]:
                # 截斷
                input_data = input_data[:, :input_shape[1]]
            elif input_data.shape[1] < input_shape[1]:
                # 補零
                padded_input = np.zeros((1, input_shape[1]))
                padded_input[:, :input_data.shape[1]] = input_data
                input_data = padded_input
        
        # 設置輸入張量
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        
        # 執行推論
        interpreter.invoke()
        
        # 獲取輸出
        outputs = []
        for output_detail in output_details:
            outputs.append(interpreter.get_tensor(output_detail['index']))
            
        # 解析預測結果
        if len(output_details) >= 3:
            # 建築、樓層和位置輸出
            building_output = outputs[0]
            floor_output = outputs[1]
            position_output = outputs[2]
            
            building_pred = np.argmax(building_output[0]) if building_output.shape[-1] > 1 else None
            floor_pred = np.argmax(floor_output[0]) if floor_output.shape[-1] > 1 else None
            position_pred = position_output[0]
            
            return building_pred, floor_pred, position_pred
        
        elif len(output_details) == 2:
            # 樓層和位置輸出
            floor_output = outputs[0]
            position_output = outputs[1]
            
            floor_pred = np.argmax(floor_output[0]) if floor_output.shape[-1] > 1 else None
            position_pred = position_output[0]
            
            return None, floor_pred, position_pred
        
        elif len(output_details) == 1:
            # 只有位置輸出或分類輸出
            output = outputs[0]
            
            if output.shape[-1] == 2:  # 假設為 (x, y) 座標
                return None, None, output[0]
            else:  # 假設為分類結果
                return np.argmax(output[0]), None, None
            
        return None, None, None
        
    def preprocess_input(self, bssids, rssis):
        """將輸入的 BSSID 和 RSSI 轉換為模型可用的輸入格式"""
        if not self.bssid_list:
            print("錯誤: 未載入完整的 BSSID 列表，無法預處理輸入數據")
            return None
            
        # 創建一個全為 -100 的向量 (表示未偵測到的 AP)
        input_vector = np.full(len(self.bssid_list), -100.0)
        
        # 填入輸入的 RSSI 值
        for bssid, rssi in zip(bssids, rssis):
            if bssid in self.bssid_list:
                idx = self.bssid_list.index(bssid)
                input_vector[idx] = float(rssi)
            else:
                print(f"警告: BSSID {bssid} 不在預設列表中，將被忽略")
        
        # 擴展為二維數組 (batch_size=1)
        input_matrix = input_vector.reshape(1, -1)
        
        # 預處理數據
        input_processed, _ = advanced_data_preprocessing(input_matrix)
        
        return input_processed
        
    def predict_all_models(self, input_data):
        """使用所有載入的模型進行預測"""
        if not self.models and not self.tflite_models:
            print("錯誤: 沒有載入任何模型")
            return {}
            
        results = {}
        
        # 使用標準 TensorFlow 模型預測
        for model_name, model in self.models.items():
            try:
                print(f"使用模型 {model_name} 進行預測...")
                
                # 檢查是否為 MLPPositionModel 或其他自定義模型類別
                if hasattr(model, 'predict') and callable(getattr(model, 'predict')) and not isinstance(model, tf.keras.Model):
                    # 使用自定義模型的預測方法
                    predictions = model.predict(input_data)
                    
                    # 解析預測結果 (假設返回三元組: building_output, floor_output, position_pred)
                    if isinstance(predictions, tuple) and len(predictions) >= 3:
                        building_probs = predictions[0][0]  # 取第一個樣本
                        floor_probs = predictions[1][0] if len(predictions) > 1 else None
                        position_pred = predictions[2][0] if len(predictions) > 2 else None
                        
                        building_pred = np.argmax(building_probs) if building_probs is not None else None
                        floor_pred = np.argmax(floor_probs) if floor_probs is not None else None
                    else:
                        # 無法解析預測結果
                        building_pred = None
                        floor_pred = None
                        position_pred = predictions  # 假設直接返回位置
                else:
                    # 使用標準 Keras 模型進行預測
                    predictions = model.predict(input_data)
                    
                    # 解析預測結果 
                    if isinstance(predictions, list):
                        if len(predictions) >= 1:
                            building_probs = predictions[0][0]
                            building_pred = np.argmax(building_probs)
                        else:
                            building_pred = None
                            building_probs = None
                            
                        if len(predictions) >= 2:
                            floor_probs = predictions[1][0]
                            floor_pred = np.argmax(floor_probs)
                        else:
                            floor_pred = None
                            floor_probs = None
                            
                        if len(predictions) >= 3:
                            position_pred = predictions[2][0]
                        else:
                            position_pred = None
                    else:
                        # 單一輸出模型 (假設為位置預測)
                        building_pred = None
                        floor_pred = None
                        position_pred = predictions[0] if len(predictions.shape) > 1 else predictions

                # 保存結果
                result = {
                    'building_prediction': building_pred,
                    'floor_prediction': floor_pred,
                    'position_prediction': position_pred
                }
                
                # 將序號轉換為名稱 (如果有映射信息)
                if building_pred is not None and 'building_mapping' in self.config:
                    rev_mapping = {v: k for k, v in self.config.get('building_mapping', {}).items()}
                    result['building_name'] = rev_mapping.get(building_pred, f"未知建築 ({building_pred})")
                
                if floor_pred is not None and 'floor_mapping' in self.config:
                    rev_mapping = {v: k for k, v in self.config.get('floor_mapping', {}).items()}
                    result['floor_name'] = rev_mapping.get(floor_pred, f"未知樓層 ({floor_pred})")
                
                # 顯示預測結果
                print(f"  模型 {model_name} 預測結果:")
                if building_pred is not None:
                    building_name = result.get('building_name', f"Building {building_pred}")
                    print(f"  - 建築物: {building_name}")
                
                if floor_pred is not None:
                    floor_name = result.get('floor_name', f"Floor {floor_pred}")
                    print(f"  - 樓層: {floor_name}")
                
                if position_pred is not None and hasattr(position_pred, 'shape'):
                    print(f"  - 位置座標: {position_pred}")
                
                results[model_name] = result
                
            except Exception as e:
                print(f"使用模型 {model_name} 預測時出錯: {e}")
                
        # 使用 TFLite 模型預測
        for model_name, model_info in self.tflite_models.items():
            try:
                print(f"使用 TFLite 模型 {model_name} 進行預測...")
                
                # 使用 TFLite 解釋器進行預測
                building_pred, floor_pred, position_pred = self.predict_tflite(model_info, input_data)
                
                # 保存結果
                result = {
                    'building_prediction': building_pred,
                    'floor_prediction': floor_pred,
                    'position_prediction': position_pred
                }
                
                # 將序號轉換為名稱 (如果有映射信息)
                if building_pred is not None and 'building_mapping' in self.config:
                    rev_mapping = {v: k for k, v in self.config.get('building_mapping', {}).items()}
                    result['building_name'] = rev_mapping.get(building_pred, f"未知建築 ({building_pred})")
                
                if floor_pred is not None and 'floor_mapping' in self.config:
                    rev_mapping = {v: k for k, v in self.config.get('floor_mapping', {}).items()}
                    result['floor_name'] = rev_mapping.get(floor_pred, f"未知樓層 ({floor_pred})")
                
                # 顯示預測結果
                print(f"  TFLite 模型 {model_name} 預測結果:")
                if building_pred is not None:
                    building_name = result.get('building_name', f"Building {building_pred}")
                    print(f"  - 建築物: {building_name}")
                
                if floor_pred is not None:
                    floor_name = result.get('floor_name', f"Floor {floor_pred}")
                    print(f"  - 樓層: {floor_name}")
                
                if position_pred is not None and hasattr(position_pred, 'shape'):
                    print(f"  - 位置座標: {position_pred}")
                
                results[f"{model_name}_tflite"] = result
                
            except Exception as e:
                print(f"使用 TFLite 模型 {model_name} 預測時出錯: {e}")
        
        return results

def display_prediction_results(results):
    """顯示所有模型的預測結果"""
    if not results:
        print("沒有預測結果可顯示")
        return
    
    # 創建顯示表格
    headers = ["模型", "建築物", "樓層", "位置座標 (x, y)"]
    table_data = []
    
    for model_name, result in results.items():
        building = result.get('building_name', str(result.get('building_prediction', 'N/A')))
        floor = result.get('floor_name', str(result.get('floor_prediction', 'N/A')))
        
        position = result.get('position_prediction', None)
        if position is not None and hasattr(position, 'shape'):
            position_str = f"({position[0]:.3f}, {position[1]:.3f})" if len(position) >= 2 else str(position)
        else:
            position_str = "N/A"
        
        table_data.append([model_name, building, floor, position_str])
    
    # 使用 tabulate 顯示結果表格
    try:
        print("\n預測結果比較:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        print("\n預測結果比較:")
        print(" | ".join(headers))
        print("-" * 80)
        for row in table_data:
            print(" | ".join(str(item) for item in row))

def parse_bssid_input(input_str):
    """解析 BSSID 輸入字串"""
    # 移除空格和額外字符
    input_str = input_str.strip().replace('"', '').replace("'", "")
    
    # 嘗試解析為 JSON
    try:
        bssid_list = json.loads(input_str)
        if isinstance(bssid_list, list):
            return bssid_list
    except:
        pass
    
    # 嘗試作為以逗號分隔的字符串解析
    if ',' in input_str:
        return [item.strip() for item in input_str.split(',')]
    
    # 返回單個元素列表
    return [input_str]

def parse_rssi_input(input_str):
    """解析 RSSI 輸入字串"""
    # 移除空格和額外字符
    input_str = input_str.strip().replace('"', '').replace("'", "")
    
    # 嘗試解析為 JSON
    try:
        rssi_list = json.loads(input_str)
        if isinstance(rssi_list, list):
            return [float(r) for r in rssi_list]
    except:
        pass
    
    # 嘗試作為以逗號分隔的字符串解析
    if ',' in input_str:
        return [float(item.strip()) for item in input_str.split(',')]
    
    # 返回單個元素列表
    return [float(input_str)]

def interactive_test():
    """互動式測試模式"""
    # 適應當前檔案結構
    processed_data_dir = "processed_data"
    # 檢查相對路徑是否存在，不存在則嘗試上層目錄
    if not os.path.exists(processed_data_dir):
        processed_data_dir = "../processed_data"
    
    predictor = ModelPredictor(processed_data_dir=processed_data_dir)
    
    if not predictor.bssid_list:
        print("錯誤: 未能載入 BSSID 列表，無法繼續")
        return
    
    # 載入模型
    if not predictor.load_models():
        print("錯誤: 未能載入任何模型，無法繼續")
        return
    
    while True:
        print("\n" + "="*60)
        print("WiFi 室內定位模型測試器 - 互動模式")
        print("="*60)
        
        print("\n輸入選項:")
        print("1. 手動輸入 BSSID 和 RSSI 值")
        print("2. 載入預設測試資料")
        print("3. 顯示可用的 BSSID 列表")
        print("4. 退出")
        
        choice = input("\n請選擇操作 [1-4]: ").strip()
        
        if choice == '1':
            # 手動輸入模式
            print("\n請輸入 BSSID 和對應的 RSSI 值")
            print("輸入格式: 可以是單個值、以逗號分隔的清單、或 JSON 格式的陣列")
            print("輸入範例: ")
            print("  BSSID: 00:11:22:33:44:55")
            print("  RSSI: -75")
            print("或")
            print("  BSSID: 00:11:22:33:44:55, 66:77:88:99:AA:BB")
            print("  RSSI: -75, -80")
            
            bssid_input = input("\nBSSID: ").strip()
            rssi_input = input("RSSI: ").strip()
            
            try:
                bssids = parse_bssid_input(bssid_input)
                rssis = parse_rssi_input(rssi_input)
                
                # 確保 BSSID 和 RSSI 列表長度一致
                if len(bssids) != len(rssis):
                    print(f"錯誤: BSSID ({len(bssids)}個) 和 RSSI ({len(rssis)}個) 數量不一致")
                    continue
                
                print(f"\n已讀取 {len(bssids)} 個 WiFi AP 信號:")
                for bssid, rssi in zip(bssids, rssis):
                    print(f"  {bssid}: {rssi} dBm")
                
                # 預處理輸入
                input_data = predictor.preprocess_input(bssids, rssis)
                
                if input_data is not None:
                    # 使用所有模型預測
                    results = predictor.predict_all_models(input_data)
                    
                    # 顯示結果
                    display_prediction_results(results)
            except Exception as e:
                print(f"錯誤: 無法處理輸入 - {e}")
        
        elif choice == '2':
            # 載入預設測試數據
            print("\n載入預設測試資料...")
            
            # 這裡可以添加一些預設的測試案例
            test_cases = [
                {
                    'name': '測試點 1',
                    'bssids': ['00:11:22:33:44:55', '66:77:88:99:AA:BB'],
                    'rssis': [-65, -78]
                },
                {
                    'name': '測試點 2',
                    'bssids': ['00:11:22:33:44:55', 'CC:DD:EE:FF:00:11'],
                    'rssis': [-72, -85]
                }
            ]
            
            # 讓用戶選擇測試案例
            print("\n可用的測試案例:")
            for i, case in enumerate(test_cases):
                print(f"{i+1}. {case['name']} ({len(case['bssids'])} 個 AP)")
            
            case_choice = input("\n請選擇測試案例 (或輸入 0 返回): ")
            
            try:
                case_idx = int(case_choice) - 1
                if case_idx == -1:
                    continue
                    
                if 0 <= case_idx < len(test_cases):
                    case = test_cases[case_idx]
                    print(f"\n使用測試案例: {case['name']}")
                    for bssid, rssi in zip(case['bssids'], case['rssis']):
                        print(f"  {bssid}: {rssi} dBm")
                    
                    # 預處理輸入
                    input_data = predictor.preprocess_input(case['bssids'], case['rssis'])
                    
                    if input_data is not None:
                        # 使用所有模型預測
                        results = predictor.predict_all_models(input_data)
                        
                        # 顯示結果
                        display_prediction_results(results)
                else:
                    print("錯誤: 無效的選擇")
            except ValueError:
                print("錯誤: 請輸入有效的數字")
        
        elif choice == '3':
            # 顯示可用的 BSSID 列表
            if predictor.bssid_list:
                print(f"\n可用的 BSSID 列表 (共 {len(predictor.bssid_list)} 個):")
                for i, bssid in enumerate(predictor.bssid_list):
                    print(f"{i+1}. {bssid}")
                    if i >= 99:  # 只顯示前100個
                        print(f"...以及另外 {len(predictor.bssid_list) - 100} 個 (省略顯示)")
                        break
            else:
                print("\n無可用的 BSSID 列表")
        
        elif choice == '4':
            # 退出
            print("\n感謝使用 WiFi 室內定位模型測試器！")
            break
        
        else:
            print("無效的選擇，請重新輸入")

if __name__ == "__main__":
    print("=== WiFi 室內定位模型測試器 ===")
    interactive_test()
