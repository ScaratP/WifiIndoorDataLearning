import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import glob

def extract_wifi_data_with_filter(folder_path, target_ssids=None):
    """
    從 scan13 資料夾中提取並過濾 WiFi 資料
    返回: (all_data, file_info)
    """
    if target_ssids is None:
        target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    all_data = []
    file_info = {
        'processed_files': [],
        'total_files': 0,
        'total_records': 0,
        'valid_records': 0
    }
    
    # 讀取所有 JSON 檔案
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    file_info['total_files'] = len(json_files)
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        file_stats = {
            'filename': file_name,
            'total_records': 0,
            'valid_records': 0,
            'file_size': os.path.getsize(json_file)
        }
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_stats['total_records'] = len(data)
            file_info['total_records'] += len(data)
            
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
                    file_stats['valid_records'] += 1
                    file_info['valid_records'] += 1
            
            file_info['processed_files'].append(file_stats)
            
        except Exception as e:
            print(f"   警告：無法處理檔案 {file_name}: {str(e)}")
            file_stats['error'] = str(e)
            file_info['processed_files'].append(file_stats)
    
    return all_data, file_info

def parse_location_info(name):
    """
    從位置名稱解析建築物、樓層和房間資訊
    返回: (building, floor, room, point_name)
    """
    if not name:
        return None, None, None, None
    
    name = name.lower().strip()
    
    # 解析建築物
    building = None
    if name.startswith('sea'):
        building = 'sea'
    elif name.startswith('seb'):
        building = 'seb'
    elif name.startswith('sec'):
        building = 'sec'
    else:
        return None, None, None, None
    
    # 解析樓層和房間
    floor = None
    room = None
    point_name = None
    
    if building:
        # 移除建築物前綴
        remaining = name[3:]
        
        # 嘗試提取完整的房間號碼
        import re
        room_match = re.search(r'(\d+)', remaining)
        if room_match:
            room_num = int(room_match.group(1))
            room = room_num
            
            # 根據房間號碼推斷樓層
            if room_num >= 101 and room_num <= 199:
                floor = 1
            elif room_num >= 201 and room_num <= 299:
                floor = 2
            elif room_num >= 301 and room_num <= 399:
                floor = 3
            elif room_num >= 401 and room_num <= 499:
                floor = 4
            elif room_num >= 501 and room_num <= 599:
                floor = 5
            else:
                # 使用百位數作為樓層，但確保至少是1
                floor = max(1, room_num // 100)
            
            # 建立明確的點名稱
            point_name = f"{building.upper()}{room_num:03d}"
    
    return building, floor, room, point_name

if __name__ == "__main__":
    # 測試函數
    folder_path = "../points/scan13"
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    extracted_data, file_info = extract_wifi_data_with_filter(folder_path, target_ssids)
    print(f"提取了 {len(extracted_data)} 個參考點")
    print(f"處理的檔案資訊: {json.dumps(file_info, ensure_ascii=False, indent=2)}")
    
    # 測試位置解析
    for record in extracted_data[:5]:
        building, floor, room, point_name = parse_location_info(record['name'])
        print(f"Name: {record['name']}, Building: {building}, Floor: {floor}, Room: {room}, Point Name: {point_name}")
