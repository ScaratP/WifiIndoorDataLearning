import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re

print("--- NTTU Wi-Fi 數據預處理腳本 (v3.0 - 含 BSSID 映射表) ---")

# --- 1. 定義你的資料路徑和規則 ---
SCAN_DIR = "points/scan13"
MATRIX_FILE = "points/transformation_data.json"
OUTPUT_DIR = "sample/data/NTTU_WIFI_DATA" # 輸出目錄，給 call_data.py 讀取

VALID_SSIDS = {'ap-nttu', 'ap2-nttu', 'nttu', 'eduroam'}
NO_SIGNAL_VALUE = 100 
TEST_SPLIT_SIZE = 0.2 
RANDOM_SEED = 42 # 確保每次切割都一樣

# --- 2. 定義輔助函式 ---

def parse_name_to_labels(name_str):
    """
    穩健性 enhanced：解析 'sec3', 'sec112', 'sea4dark', 'seb107-a' 等格式
    回傳 (building_name, floor_number)
    """
    try:
        name_str = name_str.strip() 
        # 匹配開頭的英文字母 (建築) 和 緊隨其後的第一個數字 (樓層)
        match = re.match(r'^([a-zA-Z]+)(\d)', name_str) 
        
        if match:
            building = match.group(1).upper() 
            floor = int(match.group(2))       
            return building, floor
        else:
            print(f"警告：無法解析 'name': {name_str}")
            return None, None
    except Exception as e:
        print(f"錯誤：解析 'name' {name_str} 失敗: {e}")
        return None, None

def transform_coords(x, y, image_id, matrices_dict):
    """
    使用 transformation_data.json 的矩陣進行座標轉換
    """
    try:
        image_id_str = str(image_id)
        if image_id_str not in matrices_dict:
            print(f"警告：找不到 Image ID {image_id_str} 的轉換矩陣")
            return None, None
            
        matrix = matrices_dict[image_id_str]
        longitude = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
        latitude  = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
        return longitude, latitude
        
    except Exception as e:
        print(f"錯誤：座標轉換失敗 (ImageID: {image_id}): {e}")
        return None, None

# (★ 結報函式 - 保持不變) 
def generate_data_report(master_df, train_df, val_df, bssid_count, building_map, floor_set):
    """
    產生並儲存一份詳細的資料分佈結報。
    """
    report_lines = []
    
    report_lines.append("--- NTTU Wi-Fi 數據預處理結報 ---")
    report_lines.append("\n" + "="*40)
    report_lines.append(" 1. 總體摘要 (Summary)")
    report_lines.append("="*40)
    report_lines.append(f"  > BSSID 總數 (特徵數): {bssid_count}")
    report_lines.append(f"  > 建築總數: {len(building_map)}")
    report_lines.append(f"  > 樓層總數: {len(floor_set)} (樓層: {sorted(list(floor_set))})")
    report_lines.append(f"  > 總資料筆數 (指紋數): {len(master_df)}")
    report_lines.append(f"  > 訓練集筆數: {len(train_df)} ({len(train_df)/len(master_df):.1%})")
    report_lines.append(f"  > 驗證集筆數: {len(val_df)} ({len(val_df)/len(master_df):.1%})")

    id_to_building = {v: k for k, v in building_map.items()}

    def get_dist_report(df, title):
        lines = []
        lines.append("\n" + "="*40)
        lines.append(f" 2. {title} (共 {len(df)} 筆)")
        lines.append("="*40)
        lines.append(f"{'建築':<10} {'樓層':<5} {'資料筆數':<10} {'佔比':<8}")
        lines.append(f"{'-'*10:<10} {'-'*5:<5} {'-'*10:<10} {'-'*8:<8}")
        
        if len(df) == 0:
            lines.append(" (無資料)")
            return lines

        dist = df.groupby(['BUILDINGID', 'FLOOR']).size().sort_index()
        
        for (bid, floor), count in dist.items():
            building_name = id_to_building.get(bid, 'UNKNOWN')
            percentage = (count / len(df)) * 100
            lines.append(f"{building_name:<10} {floor:<5} {count:<10} ({percentage:>5.1f}%)")
        return lines

    report_lines.extend(get_dist_report(master_df, "整體資料集 (Master Dataset) 分佈"))
    report_lines.extend(get_dist_report(train_df, "訓練集 (Training Set) 分佈"))
    report_lines.extend(get_dist_report(val_df, "驗證集 (Validation Set) 分佈"))
    report_lines.append("\n--- 結報結束 ---")
    
    return "\n".join(report_lines)

# --- 3. 載入座標轉換矩陣 ---
print(f"正在載入轉換矩陣: {MATRIX_FILE}...")
try:
    with open(MATRIX_FILE, 'r', encoding='utf-8') as f:
        coord_matrices = json.load(f)['matrices']
    print("轉換矩陣載入成功。")
except Exception as e:
    print(f"嚴重錯誤：無法載入 {MATRIX_FILE}！腳本中止。 {e}")
    exit()

# --- 4. 【第一階段】掃描所有 JSON，建立 BSSID 和 建築 的 Master List ---
print(f"開始掃描 {SCAN_DIR} 中的所有 .json 檔案 (第一階段)...")

# (★ 升級 1：從 set 改成 map，以便儲存 SSID ★)
# all_bssid_set = set()
all_bssid_map = {} # { "bssid": "ssid" }

all_building_set = set()
all_floor_set = set()
json_files_to_process = []
raw_data_cache = [] 

for filename in os.listdir(SCAN_DIR):
    if filename.endswith('.json'):
        filepath = os.path.join(SCAN_DIR, filename)
        json_files_to_process.append(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                scan_data_list = json.load(f)
                raw_data_cache.append(scan_data_list) 
                
                for entry in scan_data_list:
                    building, floor = parse_name_to_labels(entry['name'])
                    if building:
                        all_building_set.add(building)
                    if floor is not None:
                        all_floor_set.add(floor)
                        
                    for reading in entry['wifiReadings']:
                        ssid = reading['ssid'].lower()
                        bssid = reading['bssid']
                        if ssid in VALID_SSIDS:
                            # (★ 升級 2：儲存 BSSID 和 SSID 的對應關係 ★)
                            # all_bssid_set.add(bssid)
                            all_bssid_map[bssid] = ssid
                            
        except Exception as e:
            print(f"警告：讀取或解析 {filename} 失敗: {e}")

# (★ 升級 3：從 map 的 keys 建立 bssid_list ★)
# bssid_list = sorted(list(all_bssid_set))
bssid_list = sorted(list(all_bssid_map.keys()))
bssid_to_index = {bssid: i for i, bssid in enumerate(bssid_list)}
N_FEATURES = len(bssid_list)

building_list = sorted(list(all_building_set))
building_to_id = {name: i for i, name in enumerate(building_list)}
N_BUILDINGS = len(building_list)

print("第一階段完成。")
print(f"  > 總共找到 {N_FEATURES} 個符合條件的 BSSID (Wi-Fi AP)。")
print(f"  > 總共找到 {N_BUILDINGS} 棟建築: {building_to_id}")
print(f"  > 總共找到 {len(all_floor_set)} 個樓層 (從 {min(all_floor_set)} 到 {max(all_floor_set)})。")


# --- 5. 【第二階段】處理資料並建立 DataFrame ---
print("開始處理所有掃描資料 (第二階段)...")

all_processed_rows = []

for scan_data_list in raw_data_cache: 
    for entry in scan_data_list:
        building_name, floor_id = parse_name_to_labels(entry['name'])
        if not building_name or floor_id is None:
            continue 
            
        building_id = building_to_id[building_name]
        
        longitude, latitude = transform_coords(entry['x'], entry['y'], entry['imageId'], coord_matrices)
        if longitude is None:
            continue 

        scans_by_time = defaultdict(dict)
        for reading in entry['wifiReadings']:
            ssid = reading['ssid'].lower()
            bssid = reading['bssid']
            
            # (★ 升級 4：這裡的邏輯不變，因為 bssid_to_index 依然正確 ★)
            if ssid in VALID_SSIDS and bssid in bssid_to_index:
                scan_time = reading['scanTime']
                scans_by_time[scan_time][bssid] = reading['level']
        
        for scan_time, bssid_level_map in scans_by_time.items():
            feature_vector = [NO_SIGNAL_VALUE] * N_FEATURES
            
            for bssid, level in bssid_level_map.items():
                index = bssid_to_index[bssid]
                feature_vector[index] = level
                
            row = feature_vector + [longitude, latitude, floor_id, building_id]
            all_processed_rows.append(row)

# --- 6. 建立最終 DataFrame ---
print("第二階段完成。開始建立 DataFrame...")

column_names = bssid_list + ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']
master_df = pd.DataFrame(all_processed_rows, columns=column_names)

master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
master_df.dropna(inplace=True)

print(f"  > 總共產生 {len(master_df)} 筆有效的指紋資料。")

# 隨機打亂資料
master_df = master_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# 切割訓練集和驗證集
train_df, val_df = train_test_split(master_df, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED)

# --- 7. 儲存 CSV 檔案 ---
print("開始儲存 CSV 檔案...")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"  > 已建立輸出資料夾: {OUTPUT_DIR}")

train_path = os.path.join(OUTPUT_DIR, "trainingData_nttu.csv")
val_path = os.path.join(OUTPUT_DIR, "validationData_nttu.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print(f"  > 已儲存訓練集: {train_path} (共 {len(train_df)} 筆)")
print(f"  > 已儲存驗證集: {val_path} (共 {len(val_df)} 筆)")

# --- 8. (★ 全新功能：儲存 BSSID 映射表 ★) ---
print("開始儲存 BSSID 映射表...")
try:
    mapping_data = []
    # (我們使用 bssid_list 來確保順序與欄位一致)
    for i, bssid in enumerate(bssid_list):
        ssid = all_bssid_map.get(bssid, "unknown") # 從 map 中查回 ssid
        mapping_data.append({'ssid': ssid, 'bssid': bssid, 'index': i})
    
    mapping_df = pd.DataFrame(mapping_data, columns=['ssid', 'bssid', 'index'])
    
    mapping_path = os.path.join(OUTPUT_DIR, "bssid_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    
    print(f"  > BSSID 映射表已儲存至: {mapping_path}")
except Exception as e:
    print(f"錯誤：儲存 BSSID 映射表失敗: {e}")


# --- 9. 產生並儲存資料結報 ---
print("開始產生資料結報...")
report_content = generate_data_report(master_df, train_df, val_df, N_FEATURES, building_to_id, all_floor_set)

# 儲存結報
report_path = os.path.join(OUTPUT_DIR, "data_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
    
print(f"  > 結報已儲存至: {report_path}")

# 同時在螢幕上印出結報
print(report_content)

print("\n--- 處理完畢！ ---")