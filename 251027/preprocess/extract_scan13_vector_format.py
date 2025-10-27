import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

def load_bssid_mapping(mapping_path):
    bssid_list = []
    bssid_info = []
    with open(mapping_path, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                ssid, bssid, idx = parts[:3]
                bssid_list.append(bssid.lower())
                bssid_info.append({'ssid': ssid, 'bssid': bssid})
    return bssid_list, bssid_info

def parse_location_info(name):
    if not name:
        return None, None
    name = name.lower()
    building = None
    if name.startswith('sea'):
        building = 'sea'
    elif name.startswith('seb'):
        building = 'seb'
    elif name.startswith('sec'):
        building = 'sec'
    floor = None
    if building:
        remaining = name[3:]
        import re
        floor_match = re.search(r'(\d+)', remaining)
        if floor_match:
            floor_num = int(floor_match.group(1))
            if 101 <= floor_num <= 120:
                floor = 1
            elif 201 <= floor_num <= 220:
                floor = 2
            elif 301 <= floor_num <= 320:
                floor = 3
            elif 401 <= floor_num <= 420:
                floor = 4
            elif 501 <= floor_num <= 520:
                floor = 5
            else:
                floor = max(1, floor_num // 100)
    return building, floor

def main():
    mapping_path = os.path.join('../processed_data', 'bssid_mapping.csv')
    scan13_dir = os.path.join('../points', 'scan13')
    output_path = os.path.join('../processed_data', 'scan13_vector_format.csv')

    bssid_list, bssid_info = load_bssid_mapping(mapping_path)
    n_bssid = len(bssid_list)

    # 先收集所有 building/floor
    building_set = set()
    floor_set = set()
    records = []
    for fname in os.listdir(scan13_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(scan13_dir, fname)
        with open(fpath, encoding='utf-8') as fin:
            points = json.load(fin)
            for pt in points:
                building, floor = parse_location_info(pt.get('name', ''))
                if building is None or floor is None:
                    continue
                building_set.add(building)
                floor_set.add(floor)
                for w in pt.get('wifiReadings', []):
                    # 每一筆掃描都獨立
                    row = {
                        'id': pt.get('id', ''),
                        'name': pt.get('name', ''),
                        'x': pt.get('x', ''),
                        'y': pt.get('y', ''),
                        'building': building,
                        'floor': floor,
                        'wifi': pt.get('wifiReadings', [])
                    }
                    records.append(row)
    # 建立 building/floor id
    building_list = sorted(building_set)
    floor_list = sorted(floor_set)
    building_to_id = {b: i for i, b in enumerate(building_list)}
    floor_to_id = {f: i for i, f in enumerate(floor_list)}

    # 重新產生每一筆掃描的 RSS 向量
    data_rows = []
    for row in records:
        rss_vector = [-100] * n_bssid
        # 只取這一筆掃描的 wifiReadings
        bssid_to_level = {}
        for w in row['wifi']:
            bssid = w.get('bssid', '').lower()
            level = w.get('level', None)
            if bssid in bssid_list and level is not None:
                bssid_to_level[bssid] = level
        for i, bssid in enumerate(bssid_list):
            if bssid in bssid_to_level:
                rss_vector[i] = bssid_to_level[bssid]
        out_row = {
            'id': row['id'],
            'name': row['name'],
            'x': row['x'],
            'y': row['y'],
            'building': row['building'],
            'floor': row['floor'],
            'building_id': building_to_id[row['building']],
            'floor_id': floor_to_id[row['floor']]
        }
        for i, level in enumerate(rss_vector):
            out_row[f'rss_{i}'] = level
        data_rows.append(out_row)

    df = pd.DataFrame(data_rows)
    df.to_csv(output_path, index=False)
    print(f'已輸出: {output_path}，共 {len(df)} 筆')

if __name__ == '__main__':
    main()
