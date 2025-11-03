import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import glob

def extract_wifi_data_with_filter(folder_path, target_ssids=None):
    """
    從 scan13 資料夾中提取並過濾 WiFi 資料
    (★ v2 修改版：不再計算平均值，而是回傳每一筆原始掃描)
    返回: (all_data, file_info)
    """
    print("   [extract_and_filter]：正在提取資料 (v2 - 原始掃描模式)...")
    if target_ssids is None:
        target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'nttu'}
    
    all_data = []
    file_info = {
        'processed_files': [],
        'total_files': 0,
        'total_raw_records': 0,
        'valid_scans_processed': 0 # <--- 修改：計算有效的「單筆掃描」
    }
    
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    file_info['total_files'] = len(json_files)
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        file_stats = {
            'filename': file_name,
            'total_raw_records': 0,
            'valid_scans': 0,
            'file_size': os.path.getsize(json_file)
        }
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_stats['total_raw_records'] = len(data)
            file_info['total_raw_records'] += len(data)
            
            # 遍歷 JSON 中的每一個「地點」
            for record in data:
                raw_readings = record.get('wifiReadings')
                if not raw_readings:
                    continue
                    
                # --- ★★★ 關鍵修改 ★★★ ---
                # 我們不再計算平均值 (avgReadings)
                # 而是直接處理原始的 wifiReadings 列表
                
                # 1. 先過濾一次，只保留 target_ssids
                filtered_readings = []
                for reading in raw_readings:
                    ssid = reading.get('ssid', '').lower()
                    level = reading.get('level')
                    if ssid in target_ssids and level is not None:
                        filtered_readings.append(reading)
                
                # 2. 如果這個地點有收集到任何目標 SSID
                if filtered_readings:
                    # 把這個「地點的 metadata」和「過濾後的原始掃描列表」一起存入
                    all_data.append({
                        'id': record.get('id'),
                        'name': record.get('name'),
                        'x': record.get('x'),
                        'y': record.get('y'),
                        'timestamp': record.get('timestamp'),
                        'imageId': record.get('imageId'),
                        'scanCount': record.get('scanCount'),
                        'wifiReadings': filtered_readings # <--- 存入原始掃描
                    })
                    # (我們假設一次掃描就是一筆有效資料)
                    file_stats['valid_scans'] += 1
                    file_info['valid_scans_processed'] += 1
            
            file_info['processed_files'].append(file_stats)
            
        except Exception as e:
            print(f"   警告：無法處理檔案 {file_name}: {str(e)}")
            file_stats['error'] = str(e)
            file_info['processed_files'].append(file_stats)
    
    print(f"   [extract_and_filter]：處理了 {file_info['total_raw_records']} 個原始地點，")
    print(f"   [extract_and_filter]：共提取了 {len(all_data)} 筆有效掃描資料。")
    return all_data, file_info

# --- (parse_location_info 和 if __name__ == "__main__": 保持不變) ---
def parse_location_info(name):
    """
    從位置名稱解析建築物、樓層和房間資訊
    支援房間號碼格式 (如 sea101) 和走廊格式 (如 sea1)
    返回: (building, floor, room, point_name)
    """
    if not name:
        return None, None, None, None
    
    name = name.lower().strip()
    
    building = None
    if name.startswith('sea'):
        building = 'sea'
    elif name.startswith('seb'):
        building = 'seb'
    elif name.startswith('sec'):
        building = 'sec'
    else:
        return None, None, None, None
    
    floor = None
    room = None
    point_name = None
    
    if building:
        remaining = name[3:]
        import re
        number_match = re.search(r'(\d+)', remaining)
        if number_match:
            number = int(number_match.group(1))
            if number >= 100:
                room = number
                floor = max(1, number // 100)
                point_name = f"{building.upper()}{room:03d}"
            else:
                floor = number
                room = None
                point_name = f"{building.upper()}{floor}F_CORRIDOR"
    
    return building, floor, room, point_name

if __name__ == "__main__":
    folder_path = "../points/scan13"
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'nttu'}
    
    extracted_data, file_info = extract_wifi_data_with_filter(folder_path, target_ssids)
    print(f"提取了 {len(extracted_data)} 筆掃描資料")
    print(f"處理的檔案資訊: {json.dumps(file_info, ensure_ascii=False, indent=2)}")
    
    for record in extracted_data[:5]:
        building, floor, room, point_name = parse_location_info(record['name'])
        print(f"Name: {record['name']}, Building: {building}, Floor: {floor}, Room: {room}, Point Name: {point_name}")