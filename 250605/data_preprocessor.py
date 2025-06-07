import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import re
import shutil

class WifiDataPreprocessor:
    """WiFi數據預處理器：篩選特定SSID並轉換為模型輸入格式"""
    
    def __init__(self, input_folder, output_folder, target_ssids=None):
        """
        初始化WiFi數據預處理器
        
        參數:
            input_folder (str): 輸入資料夾路徑，包含原始WiFi掃描數據
            output_folder (str): 輸出資料夾路徑，存放處理後的數據
            target_ssids (set): 要保留的SSID集合，預設為None
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # 設定目標SSID，如果沒有提供則使用預設值
        if target_ssids is None:
            self.target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'guest', 'nttu'}
        else:
            self.target_ssids = {ssid.lower() for ssid in target_ssids}
        
        # 確保輸出資料夾存在
        os.makedirs(output_folder, exist_ok=True)
        
    def extract_wifi_data(self):
        """
        從輸入資料夾中提取並過濾WiFi數據
        
        返回:
            list: 包含過濾後WiFi數據的列表
        """
        print(f"從 {self.input_folder} 提取資料...")
        
        all_data = []
        
        # 讀取所有JSON檔案
        json_files = glob.glob(os.path.join(self.input_folder, '*.json'))
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for record in data:
                if not record.get('wifiReadings'):
                    continue
                    
                # 過濾SSID並計算平均值
                ssid_bssid_data = defaultdict(lambda: {'levels': [], 'bssid': None})
                
                for reading in record['wifiReadings']:
                    ssid = reading.get('ssid', '').lower()
                    bssid = reading.get('bssid')
                    level = reading.get('level')
                    
                    if ssid in self.target_ssids and level is not None:
                        key = f"{ssid}_{bssid}"
                        ssid_bssid_data[key]['levels'].append(level)
                        ssid_bssid_data[key]['bssid'] = bssid
                
                # 計算平均值
                avg_readings = []
                for key, data in ssid_bssid_data.items():
                    if data['levels']:
                        ssid = key.split('_')[0]
                        avg_readings.append({
                            'ssid': ssid,
                            'bssid': data['bssid'],
                            'avgLevel': sum(data['levels']) / len(data['levels'])
                        })
                
                if avg_readings:  # 只保留有目標SSID的點
                    all_data.append({
                        'id': record.get('id'),
                        'name': record.get('name'),
                        'x': record.get('x'),
                        'y': record.get('y'),
                        'timestamp': record.get('timestamp'),
                        'imageId': record.get('imageId'),
                        'scanCount': record.get('scanCount'),
                        'avgReadings': avg_readings
                    })
        
        print(f"成功提取 {len(all_data)} 個參考點")
        return all_data
    
    def parse_location_info(self, name):
        """
        從位置名稱解析建築物和樓層資訊
        
        參數:
            name (str): 位置名稱
            
        返回:
            tuple: (building, floor) 建築物和樓層資訊
        """
        if not name:
            return None, None
        
        name = name.lower()
        
        # 解析建築物
        building = None
        if name.startswith('sea'):
            building = 'sea'
        elif name.startswith('seb'):
            building = 'seb'
        elif name.startswith('sec'):
            building = 'sec'
        
        # 解析樓層
        floor = None
        if building:
            # 移除建築物前綴
            remaining = name[3:]
            
            # 嘗試提取數字部分作為樓層
            floor_match = re.search(r'(\d+)', remaining)
            if floor_match:
                floor_num = int(floor_match.group(1))
                # 根據數字範圍推斷樓層
                if floor_num >= 101 and floor_num <= 120:
                    floor = 1
                elif floor_num >= 201 and floor_num <= 220:
                    floor = 2
                elif floor_num >= 301 and floor_num <= 320:
                    floor = 3
                elif floor_num >= 401 and floor_num <= 420:
                    floor = 4
                elif floor_num >= 501 and floor_num <= 520:
                    floor = 5
                else:
                    # 直接使用百位數
                    floor = floor_num // 100
        
        return building, floor
    
    def create_vector_format(self, data):
        """
        將資料轉換為向量格式
        
        參數:
            data (list): 從extract_wifi_data提取的數據
            
        返回:
            tuple: (vector_data, bssid_info) 向量格式數據和BSSID信息
        """
        # 收集所有唯一的BSSID，按SSID分組
        ssid_bssids = defaultdict(set)
        
        for record in data:
            for reading in record['avgReadings']:
                ssid = reading['ssid']
                bssid = reading['bssid']
                ssid_bssids[ssid].add(bssid)
        
        # 建立有序的BSSID清單
        ordered_bssids = []
        bssid_info = []
        
        for ssid in sorted(self.target_ssids):
            if ssid in ssid_bssids:
                for bssid in sorted(ssid_bssids[ssid]):
                    ordered_bssids.append(bssid)
                    bssid_info.append({'ssid': ssid, 'bssid': bssid})
        
        # 建立向量
        vector_data = []
        
        for record in data:
            # 解析建築物和樓層
            building, floor = self.parse_location_info(record['name'])
            
            if building is None or floor is None:
                continue  # 跳過無法解析的記錄
            
            # 建立RSS向量
            rss_vector = [-100] * len(ordered_bssids)  # 預設值-100
            
            # 填入實際測量值
            bssid_to_level = {}
            for reading in record['avgReadings']:
                bssid_to_level[reading['bssid']] = reading['avgLevel']
            
            for i, bssid in enumerate(ordered_bssids):
                if bssid in bssid_to_level:
                    rss_vector[i] = bssid_to_level[bssid]
            
            vector_data.append({
                'id': record['id'],
                'name': record['name'],
                'x': record['x'],
                'y': record['y'],
                'building': building,
                'floor': floor,
                'rssVector': rss_vector
            })
        
        return vector_data, bssid_info
    
    def create_building_floor_labels(self, vector_data):
        """
        建立建築物和樓層的標籤映射
        
        參數:
            vector_data (list): 向量格式數據
            
        返回:
            tuple: (vector_data, building_to_id, floor_to_id) 更新的向量數據和映射
        """
        # 建築物映射
        buildings = sorted(set(record['building'] for record in vector_data))
        building_to_id = {building: i for i, building in enumerate(buildings)}
        
        # 樓層映射
        floors = sorted(set(record['floor'] for record in vector_data))
        floor_to_id = {floor: i for i, floor in enumerate(floors)}
        
        # 為每個記錄添加標籤
        for record in vector_data:
            record['building_id'] = building_to_id[record['building']]
            record['floor_id'] = floor_to_id[record['floor']]
        
        return vector_data, building_to_id, floor_to_id
    
    def save_to_csv(self, vector_data, bssid_info, output_dir=None):
        """
        儲存為CSV格式
        
        參數:
            vector_data (list): 向量格式數據
            bssid_info (list): BSSID信息
            output_dir (str): 輸出目錄，預設為None使用預設輸出資料夾
            
        返回:
            tuple: (vector_data, building_mapping, floor_mapping) 處理後的vector_data和映射信息
        """
        if output_dir is None:
            output_dir = self.output_folder
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加建築物和樓層標籤
        vector_data, building_mapping, floor_mapping = self.create_building_floor_labels(vector_data)
        
        # 準備資料框
        data_rows = []
        for record in vector_data:
            row = {
                'id': record['id'],
                'name': record['name'],
                'x': record['x'],
                'y': record['y'],
                'building': record['building'],
                'floor': record['floor'],
                'building_id': record['building_id'],
                'floor_id': record['floor_id']
            }
            
            # 添加RSS向量
            for i, level in enumerate(record['rssVector']):
                row[f'rss_{i}'] = level
            
            data_rows.append(row)
        
        # 儲存主要資料
        df = pd.DataFrame(data_rows)
        df.to_csv(os.path.join(output_dir, 'nttu_wifi_data.csv'), index=False)
        
        # 儲存BSSID資訊
        bssid_df = pd.DataFrame(bssid_info)
        bssid_df['index'] = range(len(bssid_df))
        bssid_df.to_csv(os.path.join(output_dir, 'bssid_mapping.csv'), index=False)
        
        # 儲存映射資訊
        with open(os.path.join(output_dir, 'label_mappings.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'building_mapping': building_mapping,
                'floor_mapping': floor_mapping
            }, f, ensure_ascii=False, indent=2)
        
        print(f"資料已儲存到 {output_dir}")
        print(f"共 {len(df)} 個參考點")
        print(f"共 {len(bssid_info)} 個BSSID")
        print(f"建築物: {df['building'].unique()}")
        print(f"樓層: {sorted(df['floor'].unique())}")
        
        return vector_data, building_mapping, floor_mapping
    
    def prepare_train_test_split(self, vector_data, test_ratio=0.2, random_seed=42):
        """
        準備訓練和測試資料集
        
        參數:
            vector_data (list): 向量格式數據
            test_ratio (float): 測試集比例，預設為0.2
            random_seed (int): 隨機種子，預設為42
            
        返回:
            tuple: (train_data, test_data) 訓練和測試資料集
        """
        np.random.seed(random_seed)
        
        # 按建築物和樓層分組
        building_floor_groups = defaultdict(list)
        for record in vector_data:
            key = (record['building'], record['floor'])
            building_floor_groups[key].append(record)
        
        train_data = []
        test_data = []
        
        # 為每個組進行分割
        for key, records in building_floor_groups.items():
            # 隨機打亂順序
            indices = np.arange(len(records))
            np.random.shuffle(indices)
            
            # 計算分割點
            split_idx = int(len(records) * (1 - test_ratio))
            
            # 分割資料
            for i in range(len(indices)):
                if i < split_idx:
                    train_data.append(records[indices[i]])
                else:
                    test_data.append(records[indices[i]])
        
        print(f"訓練集大小: {len(train_data)}")
        print(f"測試集大小: {len(test_data)}")
        
        return train_data, test_data
    
    def prepare_hadnn_format(self, vector_data, hadnn_output_dir=None):
        """
        準備HADNN模型格式資料
        
        參數:
            vector_data (list): 向量格式數據
            hadnn_output_dir (str): HADNN輸出目錄，預設為None使用預設輸出資料夾下的hadnn子目錄
        
        返回:
            dict: 包含HADNN格式資料的字典
        """
        if hadnn_output_dir is None:
            hadnn_output_dir = os.path.join(self.output_folder, 'hadnn')
        
        os.makedirs(hadnn_output_dir, exist_ok=True)
        
        # 準備訓練和測試數據
        train_data, test_data = self.prepare_train_test_split(vector_data)
        
        # 獲取RSS特徵維度
        n_rss = len(train_data[0]['rssVector']) if train_data else 0
        
        # 確定建築物和樓層數量
        buildings = sorted(set(record['building'] for record in vector_data))
        floors = sorted(set(record['floor'] for record in vector_data))
        n_buildings = len(buildings)
        n_floors = len(floors)
        
        # 提取座標並進行標準化
        train_x = np.array([record['rssVector'] for record in train_data])
        train_y = np.array([[record['x'], record['y']] for record in train_data])
        train_b = np.array([[record['building_id']] for record in train_data])
        train_c = np.array([[record['floor_id']] for record in train_data])
        
        test_x = np.array([record['rssVector'] for record in test_data])
        test_y = np.array([[record['x'], record['y']] for record in test_data])
        test_b = np.array([[record['building_id']] for record in test_data])
        test_c = np.array([[record['floor_id']] for record in test_data])
        
        # 計算座標標準化參數
        lo_mean = float(np.mean(train_y[:, 0]))
        lo_std = float(np.std(train_y[:, 0]))
        la_mean = float(np.mean(train_y[:, 1]))
        la_std = float(np.std(train_y[:, 1]))
        
        # 標準化座標
        train_y[:, 0] = (train_y[:, 0] - lo_mean) / lo_std
        train_y[:, 1] = (train_y[:, 1] - la_mean) / la_std
        
        test_y[:, 0] = (test_y[:, 0] - lo_mean) / lo_std
        test_y[:, 1] = (test_y[:, 1] - la_mean) / la_std
        
        # 儲存數據
        np.save(os.path.join(hadnn_output_dir, 'train_x.npy'), train_x)
        np.save(os.path.join(hadnn_output_dir, 'train_y.npy'), train_y)
        np.save(os.path.join(hadnn_output_dir, 'train_b.npy'), train_b)
        np.save(os.path.join(hadnn_output_dir, 'train_c.npy'), train_c)
        
        np.save(os.path.join(hadnn_output_dir, 'test_x.npy'), test_x)
        np.save(os.path.join(hadnn_output_dir, 'test_y.npy'), test_y)
        np.save(os.path.join(hadnn_output_dir, 'test_b.npy'), test_b)
        np.save(os.path.join(hadnn_output_dir, 'test_c.npy'), test_c)
        
        # 儲存配置
        config = {
            'n_rss': n_rss,
            'n_buildings': n_buildings,
            'n_floors': n_floors,
            'num_examples': len(train_data),
            'num_test_examples': len(test_data),
            'lo_mean': lo_mean,
            'lo_std': lo_std,
            'la_mean': la_mean,
            'la_std': la_std,
            'buildings': buildings,
            'floors': floors
        }
        
        with open(os.path.join(hadnn_output_dir, 'dataset_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"HADNN格式資料已儲存到 {hadnn_output_dir}")
        
        return {
            'train_x': train_x,
            'train_y': train_y,
            'train_b': train_b,
            'train_c': train_c,
            'test_x': test_x,
            'test_y': test_y,
            'test_b': test_b,
            'test_c': test_c,
            'config': config
        }    
    
    def process(self):
        """執行完整數據預處理流程"""
        print(f"開始處理 {self.input_folder} 中的WiFi數據...")
        
        # 提取數據
        data = self.extract_wifi_data()
        
        # 轉換為向量格式
        vector_data, bssid_info = self.create_vector_format(data)
        
        # 儲存為CSV並獲取標籤映射
        vector_data, building_mapping, floor_mapping = self.save_to_csv(vector_data, bssid_info)
        
        # 準備HADNN格式
        hadnn_data = self.prepare_hadnn_format(vector_data)
        
        print("數據預處理完成")
        return vector_data, bssid_info, hadnn_data

if __name__ == "__main__":
    input_folder = "/workspace/points/scan13"
    output_folder = "./processed_data"
    
    # 設定目標SSID
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'guest', 'nttu'}
    
    # 建立預處理器並執行
    preprocessor = WifiDataPreprocessor(input_folder, output_folder, target_ssids)
    preprocessor.process()
