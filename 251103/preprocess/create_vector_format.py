import json
import pandas as pd
import numpy as np
from collections import defaultdict
from extract_and_filter import extract_wifi_data_with_filter, parse_location_info
import os
from geopy.distance import geodesic # <--- ★ 務必保留

# --- ★ 轉換公式的路徑 ★ ---
TRANSFORM_CONFIG_PATH = '../processed_data/transformation_data.json'

def create_vector_format(data, target_ssids=None):
    """
    將資料轉換為向量格式
    (★ v3 修改版：處理原始掃描列表，並即時計算公尺座標)
    """
    print("   [create_vector_format]：正在轉換向量 (v3 - 原始掃描模式)...")
    if target_ssids is None:
        target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'nttu'}
    
    # --- ★ 載入轉換公式 (不變) ★ ---
    print(f"   正在從 {TRANSFORM_CONFIG_PATH} 載入座標轉換公式...")
    try:
        with open(TRANSFORM_CONFIG_PATH, 'r', encoding='utf-8') as f:
            transform_data = json.load(f)
        matrices = {int(img_id): (np.array(M_lon), np.array(M_lat)) 
                    for img_id, (M_lon, M_lat) in transform_data['matrices'].items()}
        origins = {int(img_id): tuple(origin) 
                   for img_id, origin in transform_data['origins'].items()}
        print(f"   ✅ 成功載入 {len(matrices)} 張地圖的轉換公式。")
    except FileNotFoundError:
        print(f"   ❌ 錯誤: 找不到轉換設定檔: {TRANSFORM_CONFIG_PATH}")
        return None, None
    except Exception as e:
        print(f"   ❌ 錯誤: 讀取轉換設定檔失敗: {e}")
        return None, None
    
    # --- ★ 修改：BSSID 收集 (現在 data 結構不同) ★ ---
    # data 裡的每筆 record 包含 'wifiReadings' 列表
    ssid_bssids = defaultdict(set)
    for record in data:
        for reading in record['wifiReadings']: # <--- 直接讀取 wifiReadings
            ssid = reading['ssid'].lower() # 確保是小寫
            bssid = reading['bssid']
            if ssid in target_ssids:
                ssid_bssids[ssid].add(bssid)
    
    ordered_bssids = []
    bssid_info = []
    for ssid in sorted(target_ssids):
        if ssid in ssid_bssids:
            for bssid in sorted(ssid_bssids[ssid]):
                ordered_bssids.append(bssid)
                bssid_info.append({'ssid': ssid, 'bssid': bssid})
    
    print(f"   [create_vector_format]：建立了 {len(ordered_bssids)} 個 BSSID 的向量。")

    # --- ★ 修改：建立向量 (不再查找，而是計算) ★ ---
    vector_data = []
    points_calibrated = 0
    points_skipped = 0
    
    # 遍歷由 extract_and_filter 傳來的每一筆「掃描資料」
    for record in data:
        # 解析建築物、樓層 (這部分不變)
        building, floor, room, point_name = parse_location_info(record['name'])
        
        if building is None or floor is None or point_name is None:
            points_skipped += 1
            continue
            
        # --- ★ 即時計算公尺座標 (不變) ★ ---
        image_id = record.get('imageId')
        if image_id not in matrices:
            points_skipped += 1
            continue 

        M_lon, M_lat = matrices[image_id]
        origin_lat, origin_lon = origins[image_id]
        
        x_img = float(record['x']) if record['x'] is not None else 0.0
        y_img = float(record['y']) if record['y'] is not None else 0.0
        img_point = np.array([x_img, y_img, 1]) 

        pred_lon = img_point @ M_lon
        pred_lat = img_point @ M_lat

        y_meter = geodesic((origin_lat, origin_lon), (pred_lat, origin_lon)).meters
        x_meter = geodesic((origin_lat, origin_lon), (origin_lat, pred_lon)).meters
        
        if pred_lat < origin_lat: y_meter *= -1
        if pred_lon < origin_lon: x_meter *= -1
            
        points_calibrated += 1
        
        # --- ★ 關鍵修改：從 wifiReadings 建立 RSS 向量 ★ ---
        rss_vector = [-100] * len(ordered_bssids)
        
        # 建立這「單次」掃描的 BSSID -> Level 映射
        bssid_to_level = {}
        for reading in record['wifiReadings']: # <--- 讀取原始掃描
            bssid = reading['bssid']
            level = reading.get('level')
            bssid_to_level[bssid] = level
        
        # 填入向量
        for i, bssid in enumerate(ordered_bssids):
            if bssid in bssid_to_level:
                rss_vector[i] = bssid_to_level[bssid]
        
        vector_data.append({
            'id': record['id'],
            'name': record['name'],
            'x_percent': x_img, 
            'y_percent': y_img, 
            'x_meter': x_meter, # 儲存公尺座標
            'y_meter': y_meter, # 儲存公尺座標
            'building': building,
            'floor': floor,
            'room': room,
            'point_name': point_name,
            'rssVector': rss_vector
        })
    
    print(f"   [create_vector_format]：向量轉換完成：{points_calibrated} 筆掃描成功計算公尺座標。")
    if points_skipped > 0:
        print(f"   [create_vector_format]：⚠️ 警告：{points_skipped} 筆掃描因無法解析或地圖缺乏轉換公式而被跳過。")
        
    return vector_data, bssid_info

# --- (create_building_floor_point_labels 函數不變) ---
def create_building_floor_point_labels(vector_data):
    buildings = sorted(set(record['building'] for record in vector_data))
    building_to_id = {building: i for i, building in enumerate(buildings)}
    floors = sorted(set(record['floor'] for record in vector_data))
    floor_to_id = {floor: i for i, floor in enumerate(floors)}
    points = sorted(set(record['point_name'] for record in vector_data))
    point_to_id = {point: i for i, point in enumerate(points)}
    for record in vector_data:
        record['building_id'] = building_to_id[record['building']]
        record['floor_id'] = floor_to_id[record['floor']]
        record['point_id'] = point_to_id[record['point_name']]
    return vector_data, building_to_id, floor_to_id, point_to_id

# --- (save_to_csv 函數不變) ---
# 它本來就設計為儲存 x_meter, y_meter
def save_to_csv(vector_data, bssid_info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data_rows = []
    for record in vector_data:
        row = {
            'id': record['id'],
            'name': record['name'],
            'x_percent': record['x_percent'], 
            'y_percent': record['y_percent'], 
            'x_meter': record['x_meter'], 
            'y_meter': record['y_meter'], 
            'building': record['building'],
            'floor': record['floor'],
            'room': record['room'],
            'point_name': record['point_name'],
            'building_id': record['building_id'],
            'floor_id': record['floor_id'],
            'point_id': record['point_id']
        }
        for i, level in enumerate(record['rssVector']):
            row[f'rss_{i}'] = level
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(os.path.join(output_dir, 'nttu_wifi_data.csv'), index=False)
    
    bssid_df = pd.DataFrame(bssid_info)
    bssid_df['index'] = range(len(bssid_df))
    bssid_df.to_csv(os.path.join(output_dir, 'bssid_mapping.csv'), index=False)
    
    print(f"資料已儲存到 {output_dir}")
    print(f"共 {len(df)} 筆掃描資料 (資料量已增加！)") # <--- 修改日誌
    print(f"共 {len(bssid_info)} 個 BSSID")
    print(f"建築物: {df['building'].unique()}")
    print(f"樓層: {sorted(df['floor'].unique())}")
    print(f"點位數量: {len(df['point_name'].unique())}")
    if 'x_meter' in df.columns:
        print(f"座標範圍 (公尺): x=[{df['x_meter'].min():.2f}, {df['x_meter'].max():.2f}], y=[{df['y_meter'].min():.2f}, {df['y_meter'].max():.2f}]")

# --- (主執行區塊 __main__ 不變) ---
if __name__ == "__main__":
    folder_path = "../points/scan13"
    output_dir = "../processed_data"
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'nttu'}
    
    extracted_data, file_info = extract_wifi_data_with_filter(folder_path, target_ssids)
    
    vector_data, bssid_info = create_vector_format(extracted_data, target_ssids)
    
    if vector_data is not None:
        vector_data, building_mapping, floor_mapping, point_mapping = create_building_floor_point_labels(vector_data)
        save_to_csv(vector_data, bssid_info, output_dir)
        
        with open(os.path.join(output_dir, 'label_mappings.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'building_mapping': building_mapping,
                'floor_mapping': floor_mapping,
                'point_mapping': point_mapping
            }, f, ensure_ascii=False, indent=2)
    else:
        print("❌ 預處理因座標校正失敗而中止。")