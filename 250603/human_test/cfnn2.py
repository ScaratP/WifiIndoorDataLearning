import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import argparse
import sys
from scipy.spatial.distance import cdist

class LocationPredictor:
    def __init__(self, model_path, config_path, mapping_path, labels_path):
        # 載入模型
        self.model = keras.models.load_model(model_path)
        
        # 載入配置和映射
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # 載入 bssid 映射
        self.bssid_mapping = pd.read_csv(mapping_path)
        self.bssid_dict = dict(zip(self.bssid_mapping['bssid'], self.bssid_mapping['index']))
        
        # 載入標籤映射
        with open(labels_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        # 反轉建築物和樓層映射
        self.building_inv_map = {v: k for k, v in self.label_mapping.get('building_map', {}).items()}
        self.floor_inv_map = {v: k.split('_')[1] for k, v in self.label_mapping.get('floor_map', {}).items()}
        
        # 載入參考點資料
        self.reference_points = self._load_reference_points()
        
        print(f"已載入模型: {model_path}")
        print(f"可識別的建築物: {list(self.building_inv_map.values())}")
        print(f"已載入 {len(self.reference_points)} 個參考點")
    
    def _load_reference_points(self):
        """載入所有參考點資訊"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        points_dir = os.path.join(base_dir, '../points', 'scan13')
        reference_points = []
        
        # 載入所有建築物的參考點
        for building_file in os.listdir(points_dir):
            if building_file.endswith('.json'):
                file_path = os.path.join(points_dir, building_file)
                try:
                    with open(file_path, 'r') as f:
                        building_data = json.load(f)
                        for point in building_data:
                            if 'x' in point and 'y' in point and 'name' in point:
                                reference_points.append({
                                    'name': point['name'],
                                    'building': point.get('building', building_file.split('_')[0]),
                                    'floor': point.get('floor', ''),
                                    'x': point['x'],
                                    'y': point['y'],
                                    'id': point.get('id', '')
                                })
                except Exception as e:
                    print(f"警告: 讀取參考點文件 {file_path} 失敗: {e}")
        
        return reference_points
    
    def find_nearest_reference_point(self, predicted_location, building=None, floor=None):
        """找出最接近預測位置的參考點"""
        if not self.reference_points:
            return None
            
        # 過濾同一建築物和樓層的參考點
        filtered_points = self.reference_points
        if building:
            filtered_points = [p for p in filtered_points if p['building'] == building or building in p['building']]
        if floor:
            filtered_points = [p for p in filtered_points if str(p['floor']) == str(floor)]
            
        if not filtered_points:
            # 如果沒有找到符合建築物和樓層的參考點，使用所有參考點
            filtered_points = self.reference_points
            
        # 計算到每個參考點的距離
        pred_coords = np.array([[predicted_location['x'], predicted_location['y']]])
        ref_coords = np.array([[p['x'], p['y']] for p in filtered_points])
        
        distances = cdist(pred_coords, ref_coords, metric='euclidean')[0]
        
        # 找出最近的參考點
        nearest_idx = np.argmin(distances)
        nearest_point = filtered_points[nearest_idx]
        nearest_distance = distances[nearest_idx]
        
        return {
            'name': nearest_point['name'],
            'building': nearest_point['building'],
            'floor': nearest_point['floor'],
            'x': nearest_point['x'],
            'y': nearest_point['y'],
            'distance': nearest_distance
        }
    
    def scan_wifi(self):
        """模擬掃描 WiFi 訊號 (實際應用中應替換為真實的掃描函數)"""
        print("請輸入 WiFi 掃描結果 (每行格式: bssid,RSSI)")
        print("完成輸入後請輸入空行")
        print("範例: 00:11:22:33:44:55,-65")
        
        wifi_readings = []
        while True:
            line = input().strip()
            if not line:
                break
                
            try:
                bssid, rssi = line.split(',')
                bssid = bssid.strip()
                rssi = float(rssi.strip())
                wifi_readings.append({
                    'bssid': bssid,
                    'level': rssi
                })
            except Exception as e:
                print(f"輸入格式錯誤: {e}")
        
        return wifi_readings
    
    def preprocess_wifi_data(self, wifi_readings):
        """將 WiFi 讀數轉換為模型輸入特徵向量"""
        # 建立特徵向量 (所有 bssid 的初始信號強度設為 -100 dBm)
        feature_vector = np.full((len(self.bssid_dict),), -100.0)
        
        # 填入已知的訊號強度
        mapped_count = 0
        for reading in wifi_readings:
            bssid = reading['bssid']
            if bssid in self.bssid_dict:
                idx = self.bssid_dict[bssid]
                feature_vector[idx] = reading['level']
                mapped_count += 1
        
        print(f"找到 {mapped_count}/{len(wifi_readings)} 個已知的 bssid")
        
        # 標準化特徵 (使用模型訓練時的參數)
        mean_val = -80.0  # 假設的平均值
        std_val = 15.0    # 假設的標準差
        feature_vector = (feature_vector - mean_val) / std_val
        
        return np.expand_dims(feature_vector, axis=0)  # 添加批次維度
    
    def predict(self, wifi_readings):
        """預測位置"""
        # 預處理 WiFi 數據
        X = self.preprocess_wifi_data(wifi_readings)
        
        # 進行預測
        predictions = self.model.predict(X, verbose=0)
        
        if isinstance(predictions, list) and len(predictions) == 3:  # HADNN 模型有三個輸出 [位置, 建築, 樓層]
            coords_pred, building_pred, floor_pred = predictions
            
            # 解碼建築物和樓層
            building_id = np.argmax(building_pred[0])
            floor_id = np.argmax(floor_pred[0])
            building = self.building_inv_map.get(building_id, "未知建築")
            floor = self.floor_inv_map.get(floor_id, "未知樓層")
            
            # 位置坐標
            x, y = coords_pred[0]
            
            result = {
                'building': building,
                'floor': floor,
                'x': float(x),
                'y': float(y),
                'confidence': {
                    'building': float(building_pred[0][building_id]),
                    'floor': float(floor_pred[0][floor_id])
                }
            }
        else:  # 單一輸出模型 (只預測位置)
            # 如果是單一輸出，假設它是位置坐標
            if isinstance(predictions, list):
                coords_pred = predictions[0]
            else:
                coords_pred = predictions
                
            x, y = coords_pred[0]
            
            result = {
                'building': "未知建築",
                'floor': "未知樓層",
                'x': float(x),
                'y': float(y)
            }
            
        # 尋找最接近的參考點
        nearest_point = self.find_nearest_reference_point(
            result, building=result.get('building'), floor=result.get('floor')
        )
        
        if nearest_point:
            result['nearest_point'] = nearest_point
            
        return result

def main():
    # 設置模型和配置路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '../trained_models', 'CFNN2')  # 使用表現最佳的CFNN2模型
    config_path = os.path.join(base_dir, '../hadnn_data', 'dataset_config.json')
    mapping_path = os.path.join(base_dir, '../processed_data', 'bssid_mapping.csv')
    labels_path = os.path.join(base_dir, '../processed_data', 'label_mappings.json')
    
    # 檢查檔案是否存在
    if not os.path.exists(model_dir):
        print(f"錯誤: 找不到模型目錄 {model_dir}")
        
        # 列出可用的模型
        models_dir = os.path.join(base_dir, 'trained_models')
        available_models = [m for m in os.listdir(models_dir) 
                           if os.path.isdir(os.path.join(models_dir, m))]
        
        if available_models:
            print(f"可用的模型: {', '.join(available_models)}")
            model_name = input("請選擇一個模型: ").strip()
            model_dir = os.path.join(base_dir, 'trained_models', model_name)
        else:
            print("找不到任何可用模型，程式結束")
            sys.exit(1)
    
    # 建立位置預測器
    predictor = LocationPredictor(
        model_dir, config_path, mapping_path, labels_path
    )
    
    # 掃描 WiFi 訊號或加載樣本數據
    print("\n====== WiFi 室內定位測試程式 ======")
    print("選擇輸入模式:")
    print("1. 手動輸入 WiFi 訊號")
    print("2. 使用樣本數據")
    
    choice = input("請選擇 (1/2): ").strip()
    
    if choice == '2':
        # 使用範例數據
        print("使用樣本 WiFi 掃描數據...")
        # 讀取樣本數據 (從json中讀取第一個數據點)
        sample_path = os.path.join(base_dir, 'points', 'scan13', 'se1_13.json')
        try:
            with open(sample_path, 'r') as f:
                samples = json.load(f)
                if samples:
                    sample_index = 0  # 預設使用第一個樣本
                    wifi_readings = samples[sample_index]['wifiReadings']
                    sample_name = samples[sample_index].get('name', '未知點位')
                    print(f"已載入參考點 '{sample_name}' 的 {len(wifi_readings)} 個 WiFi 訊號樣本")
                else:
                    print("樣本數據為空，切換到手動輸入模式")
                    wifi_readings = predictor.scan_wifi()
        except Exception as e:
            print(f"讀取樣本數據失敗: {e}")
            wifi_readings = predictor.scan_wifi()
    else:
        # 手動輸入
        wifi_readings = predictor.scan_wifi()
    
    if not wifi_readings:
        print("沒有輸入任何 WiFi 訊號，無法進行預測")
        return
    
    # 進行預測
    print("\n正在預測位置...")
    result = predictor.predict(wifi_readings)
    
    # 顯示結果
    print("\n====== 預測結果 ======")
    print(f"建築物: {result['building']}")
    print(f"樓層: {result['floor']}")
    print(f"座標: ({result['x']:.2f}, {result['y']:.2f})")
    
    if 'confidence' in result:
        print(f"建築物置信度: {result['confidence']['building']:.4f}")
        print(f"樓層置信度: {result['confidence']['floor']:.4f}")
        
    if 'nearest_point' in result:
        nearest = result['nearest_point']
        print("\n最接近的參考點:")
        print(f"名稱: {nearest['name']}")
        print(f"位置: 建築 {nearest['building']}, 樓層 {nearest['floor']}")
        print(f"座標: ({nearest['x']:.2f}, {nearest['y']:.2f})")
        print(f"距離: {nearest['distance']:.2f} 公尺")

if __name__ == "__main__":
    main()