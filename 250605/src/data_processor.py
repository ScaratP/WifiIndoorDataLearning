import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.target_networks = ['Ap-nttu', 'Ap2-nttu', 'eduroam', 'Guest', 'NTTU']
        self.scaler = StandardScaler()
        self.building_encoder = {}
        self.floor_encoder = {}
        
    def load_scan_data(self):
        """從 scan13 資料夾載入 JSON 文件"""
        scan_data = []
        
        # 載入 se1_13.json (建築物1)
        se1_path = os.path.join(self.data_path, 'se1_13.json')
        if os.path.exists(se1_path):
            with open(se1_path, 'r', encoding='utf-8') as f:
                se1_data = json.load(f)
                for point in se1_data:
                    if point.get('wifiReadings'):
                        point['building'] = 'SE1'
                        point['floor'] = 1
                        scan_data.append(point)
        
        # 載入 se2_13.json (建築物2) 
        se2_path = os.path.join(self.data_path, 'se2_13.json')
        if os.path.exists(se2_path):
            with open(se2_path, 'r', encoding='utf-8') as f:
                se2_data = json.load(f)
                for point in se2_data:
                    if point.get('wifiReadings'):
                        point['building'] = 'SE2'
                        point['floor'] = 2
                        scan_data.append(point)
        
        return scan_data
    
    def extract_wifi_features(self, wifi_readings):
        """提取指定網路的 WiFi 特徵"""
        features = defaultdict(list)
        
        for reading in wifi_readings:
            # 檢查是否包含目標網路
            ssid = reading.get('ssid', '')
            if any(network in ssid for network in self.target_networks):
                bssid = reading.get('bssid', '')
                rssi = reading.get('level', -100)  # 預設 -100 dBm
                
                if bssid:
                    features[bssid].append(rssi)
        
        # 計算每個 BSSID 的平均 RSSI
        averaged_features = {}
        for bssid, rssi_list in features.items():
            averaged_features[bssid] = np.mean(rssi_list)
            
        return averaged_features
    
    def create_feature_matrix(self, scan_data):
        """建立特徵矩陣"""
        # 收集所有 BSSID
        all_bssids = set()
        processed_data = []
        
        for point in scan_data:
            wifi_features = self.extract_wifi_features(point.get('wifiReadings', []))
            if wifi_features:  # 只保留有 WiFi 數據的點
                all_bssids.update(wifi_features.keys())
                processed_data.append({
                    'wifi_features': wifi_features,
                    'x': point['x'],
                    'y': point['y'], 
                    'building': point['building'],
                    'floor': point['floor'],
                    'name': point.get('name', ''),
                    'id': point.get('id', '')
                })
        
        # 創建排序的 BSSID 列表
        bssid_list = sorted(list(all_bssids))
        
        # 建立特徵矩陣
        X = []
        buildings = []
        floors = []
        positions = []
        metadata = []
        
        for data in processed_data:
            # WiFi 特徵向量
            feature_vector = []
            for bssid in bssid_list:
                rssi = data['wifi_features'].get(bssid, -100)  # 預設 -100 dBm
                feature_vector.append(rssi)
            
            X.append(feature_vector)
            buildings.append(data['building'])
            floors.append(data['floor'])
            positions.append([data['x'], data['y']])
            metadata.append({
                'name': data['name'],
                'id': data['id']
            })
        
        return np.array(X), buildings, floors, np.array(positions), metadata, bssid_list
    
    def encode_labels(self, buildings, floors):
        """編碼建築物和樓層標籤"""
        # 建築物編碼
        unique_buildings = sorted(list(set(buildings)))
        self.building_encoder = {building: i for i, building in enumerate(unique_buildings)}
        building_labels = [self.building_encoder[b] for b in buildings]
        
        # 樓層編碼 (按建築物分組)
        building_floor_combinations = sorted(list(set(zip(buildings, floors))))
        self.floor_encoder = {combo: i for i, combo in enumerate(building_floor_combinations)}
        floor_labels = [self.floor_encoder[(b, f)] for b, f in zip(buildings, floors)]
        
        return np.array(building_labels), np.array(floor_labels)
    
    def split_by_building_floor(self, X, buildings, floors, positions):
        """按建築物樓層組合分割數據"""
        data_splits = {}
        
        for i, (building, floor) in enumerate(zip(buildings, floors)):
            key = f"{building}_F{floor}"
            if key not in data_splits:
                data_splits[key] = {
                    'X': [],
                    'positions': [],
                    'indices': []
                }
            
            data_splits[key]['X'].append(X[i])
            data_splits[key]['positions'].append(positions[i])
            data_splits[key]['indices'].append(i)
        
        # 轉換為 numpy 陣列
        for key in data_splits:
            data_splits[key]['X'] = np.array(data_splits[key]['X'])
            data_splits[key]['positions'] = np.array(data_splits[key]['positions'])
            data_splits[key]['indices'] = np.array(data_splits[key]['indices'])
        
        return data_splits
    
    def normalize_features(self, X_train, X_test=None):
        """正規化特徵"""
        X_train_normalized = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_normalized = self.scaler.transform(X_test)
            return X_train_normalized, X_test_normalized
        
        return X_train_normalized
    
    def save_processed_data(self, save_path, **data_dict):
        """儲存處理後的數據"""
        os.makedirs(save_path, exist_ok=True)
        
        for name, data in data_dict.items():
            np.save(os.path.join(save_path, f"{name}.npy"), data)
    
    def process_all_data(self):
        """完整的數據處理流程"""
        print("正在載入 scan13 數據...")
        scan_data = self.load_scan_data()
        print(f"載入了 {len(scan_data)} 個數據點")
        
        print("正在提取 WiFi 特徵...")
        X, buildings, floors, positions, metadata, bssid_list = self.create_feature_matrix(scan_data)
        print(f"特徵矩陣形狀: {X.shape}")
        print(f"使用了 {len(bssid_list)} 個 BSSID")
        
        print("正在編碼標籤...")
        building_labels, floor_labels = self.encode_labels(buildings, floors)
        
        print("正在按建築樓層分割數據...")
        data_splits = self.split_by_building_floor(X, buildings, floors, positions)
        
        return {
            'X': X,
            'building_labels': building_labels,
            'floor_labels': floor_labels,
            'positions': positions,
            'buildings': buildings,
            'floors': floors,
            'metadata': metadata,
            'bssid_list': bssid_list,
            'data_splits': data_splits,
            'building_encoder': self.building_encoder,
            'floor_encoder': self.floor_encoder
        }

if __name__ == "__main__":
    # 測試數據處理器
    data_path = "/workspace/points/scan13"
    processor = DataProcessor(data_path)
    
    try:
        results = processor.process_all_data()
        print("\n數據處理完成！")
        print(f"建築物類別: {list(results['building_encoder'].keys())}")
        print(f"樓層組合: {list(results['floor_encoder'].keys())}")
        print(f"數據分割: {list(results['data_splits'].keys())}")
        
        # 儲存處理結果
        processor.save_processed_data(
            "../data/processed",
            X=results['X'],
            building_labels=results['building_labels'],
            floor_labels=results['floor_labels'],
            positions=results['positions']
        )
        print("數據已儲存到 data/processed/ 目錄")
        
    except Exception as e:
        print(f"數據處理錯誤: {e}")
