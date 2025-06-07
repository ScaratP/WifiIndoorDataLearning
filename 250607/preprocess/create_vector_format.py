import json
import pandas as pd
import numpy as np
from collections import defaultdict
from extract_and_filter import extract_wifi_data_with_filter, parse_location_info
import os

def create_vector_format(data, target_ssids=None):
    """
    將資料轉換為向量格式，適用於 HADNN 模型
    """
    if target_ssids is None:
        target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    # 收集所有唯一的 BSSID，按 SSID 分組
    ssid_bssids = defaultdict(set)
    
    for record in data:
        for reading in record['avgReadings']:
            ssid = reading['ssid']
            bssid = reading['bssid']
            if ssid in target_ssids:
                ssid_bssids[ssid].add(bssid)
    
    # 建立有序的 BSSID 清單
    ordered_bssids = []
    bssid_info = []
    
    for ssid in sorted(target_ssids):
        if ssid in ssid_bssids:
            for bssid in sorted(ssid_bssids[ssid]):
                ordered_bssids.append(bssid)
                bssid_info.append({'ssid': ssid, 'bssid': bssid})
    
    # 建立向量
    vector_data = []
    
    for record in data:
        # 解析建築物和樓層
        building, floor = parse_location_info(record['name'])
        
        if building is None or floor is None:
            continue  # 跳過無法解析的記錄
        
        # 建立 RSS 向量
        rss_vector = [-100] * len(ordered_bssids)  # 預設值 -100
        
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

def create_building_floor_labels(vector_data):
    """
    建立建築物和樓層的標籤映射
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

def save_to_csv(vector_data, bssid_info, output_dir):
    """
    儲存為 CSV 格式
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 添加 RSS 向量
        for i, level in enumerate(record['rssVector']):
            row[f'rss_{i}'] = level
        
        data_rows.append(row)
    
    # 儲存主要資料
    df = pd.DataFrame(data_rows)
    df.to_csv(os.path.join(output_dir, 'nttu_wifi_data.csv'), index=False)
    
    # 儲存 BSSID 資訊
    bssid_df = pd.DataFrame(bssid_info)
    bssid_df['index'] = range(len(bssid_df))
    bssid_df.to_csv(os.path.join(output_dir, 'bssid_mapping.csv'), index=False)
    
    print(f"資料已儲存到 {output_dir}")
    print(f"共 {len(df)} 個參考點")
    print(f"共 {len(bssid_info)} 個 BSSID")
    print(f"建築物: {df['building'].unique()}")
    print(f"樓層: {sorted(df['floor'].unique())}")

if __name__ == "__main__":
    # 執行資料處理
    folder_path = "../points/scan13"
    output_dir = "../processed_data"
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    # 提取資料
    extracted_data = extract_wifi_data_with_filter(folder_path, target_ssids)
    
    # 轉換為向量格式
    vector_data, bssid_info = create_vector_format(extracted_data, target_ssids)
    
    # 建立標籤
    vector_data, building_mapping, floor_mapping = create_building_floor_labels(vector_data)
    
    # 儲存為 CSV
    save_to_csv(vector_data, bssid_info, output_dir)
    
    # 儲存映射資訊
    with open(os.path.join(output_dir, 'label_mappings.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'building_mapping': building_mapping,
            'floor_mapping': floor_mapping
        }, f, ensure_ascii=False, indent=2)
