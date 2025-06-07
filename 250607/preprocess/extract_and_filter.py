import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import glob

def extract_wifi_data_with_filter(folder_path, target_ssids=None):
    """
    從 scan13 資料夾中提取並過濾 WiFi 資料
    """
    if target_ssids is None:
        target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    all_data = []
    
    # 讀取所有 JSON 檔案
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data:
            if not record.get('wifiReadings'):
                continue
                
            # 過濾 SSID 並計算平均值
            ssid_bssid_data = defaultdict(lambda: {'levels': [], 'bssid': None})
            
            for reading in record['wifiReadings']:
                ssid = reading.get('ssid', '').lower()
                bssid = reading.get('bssid')
                level = reading.get('level')
                
                if ssid in target_ssids and level is not None:
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
            
            if avg_readings:  # 只保留有目標 SSID 的點
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
    
    return all_data

def parse_location_info(name):
    """
    從位置名稱解析建築物和樓層資訊，不包含0樓
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
        import re
        floor_match = re.search(r'(\d+)', remaining)
        if floor_match:
            floor_num = int(floor_match.group(1))
            # 根據數字範圍推斷樓層，確保沒有0樓
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
                # 直接使用百位數，但確保至少是1
                floor = max(1, floor_num // 100)
    
    return building, floor

if __name__ == "__main__":
    # 測試函數
    folder_path = "../points/scan13"
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    extracted_data = extract_wifi_data_with_filter(folder_path, target_ssids)
    print(f"提取了 {len(extracted_data)} 個參考點")
    
    # 測試位置解析
    for record in extracted_data[:5]:
        building, floor = parse_location_info(record['name'])
        print(f"Name: {record['name']}, Building: {building}, Floor: {floor}")
