import xml.etree.ElementTree as ET
import json
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
import re

# --- 設定檔案名稱 ---
XML_FILE = 'NTTU_MAP.xml'
JSON_FILE = 'classroom_points.json'
# ★ 輸出檔案 (我們把設定檔存在 processed_data，方便管理)
OUTPUT_CSV_CHECK = '../processed_data/classroom_points_with_meters_CHECK.csv' 
OUTPUT_TRANSFORM_CONFIG = '../processed_data/transformation_data.json'

def parse_xml_coordinates(xml_file):
    """
    解析 NTTU_MAP.xml 檔案，
    返回一個 {教室名稱: (經度, 緯度)} 的字典。
    """
    print(f"正在解析 {xml_file}...")
    coords_dict = {}
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for placemark in root.iter('Placemark'):
            name_tag = placemark.find('name')
            point_tag = placemark.find('Point')
            
            if name_tag is not None and point_tag is not None:
                name = name_tag.text.strip().lower() # 轉換為小寫
                coords_tag = point_tag.find('coordinates')
                
                if coords_tag is not None:
                    coords_text = coords_tag.text.strip()
                    try:
                        lon, lat, _ = coords_text.split(',')
                        coords_dict[name] = (float(lon), float(lat))
                    except (ValueError, TypeError) as e:
                        print(f"  - 警告: 無法解析 {name} 的座標 '{coords_text}': {e}")
                        
    except ET.ParseError as e:
        print(f"❌ 錯誤: 解析 XML 失敗: {e}")
        return None
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到檔案 {xml_file}")
        return None

    print(f"  ✅ 完成解析，找到 {len(coords_dict)} 個經緯度標記。")
    return coords_dict

def get_base_name(name_lower):
    """
    從 'sec101f', 'sec101b' 中提取 'sec101'
    """
    match = re.match(r'^(sea|seb|sec)\d+', name_lower)
    if match:
        return match.group(0) # 返回匹配到的部分 (例如 'sec101')
    return name_lower # 如果不匹配，返回原來的名稱

def load_and_preprocess_json(json_file):
    """
    載入 classroom_points.json，
    並建立 'base_name' 欄位用於匹配。
    """
    print(f"正在載入 {json_file}...")
    try:
        df = pd.read_json(json_file)
        df = df.rename(columns={'x': 'x_img', 'y': 'y_img'})
        df['name_lower'] = df['name'].str.lower()
        df['base_name'] = df['name_lower'].apply(get_base_name)
        print(f"  ✅ 完成載入，找到 {len(df)} 個教室點位。")
        return df
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到檔案 {json_file}")
        return None
    except Exception as e:
        print(f"❌ 錯誤: 載入 JSON 失敗: {e}")
        return None

def calculate_and_save_transforms(df_points, xml_coords):
    """
    為每一張地圖 (imageId) 計算其轉換矩陣，
    並將矩陣和原點儲存到 JSON 檔案中。
    """
    print("正在計算轉換矩陣...")
    
    # --- 1. 建立控制點 (Control Points) ---
    print("  - 正在計算控制點 (平均 'f' 和 'b' 點)...")
    df_control_points = df_points.groupby(['imageId', 'base_name']).agg(
        x_img_avg=('x_img', 'mean'),
        y_img_avg=('y_img', 'mean')
    ).reset_index()

    # --- 2. 匹配 XML 經緯度 ---
    print("  - 正在匹配控制點與 XML 經緯度...")
    df_control_points['coords'] = df_control_points['base_name'].map(xml_coords)
    
    matched_controls = df_control_points.dropna(subset=['coords'])
    unmatched_controls = df_control_points[df_control_points['coords'].isna()]['base_name'].unique()
    
    if len(unmatched_controls) > 0:
        print(f"  ⚠️ 警告：以下 {len(unmatched_controls)} 個 base_name 在 XML 中找不到對應點:")
        print(f"      {unmatched_controls}")
        
    if len(matched_controls) == 0:
        print("❌ 致命錯誤：沒有任何控制點匹配成功！")
        return None
        
    matched_controls[['lon', 'lat']] = pd.DataFrame(matched_controls['coords'].tolist(), index=matched_controls.index)
    
    # --- 3. 分別為每張地圖計算轉換矩陣 ---
    transformation_matrices = {}
    coordinate_origins = {}
    
    for image_id, group_df in matched_controls.groupby('imageId'):
        print(f"\n--- 正在處理 imageId: {image_id} (有 {len(group_df)} 個控制點) ---")
        
        if len(group_df) < 3:
            print(f"  ⚠️ 跳過 imageId {image_id}：有效控制點少於 3 個，無法計算轉換矩陣。")
            continue

        A = np.hstack([
            group_df[['x_img_avg', 'y_img_avg']].values,
            np.ones((len(group_df), 1))
        ])
        
        B_lon = group_df['lon'].values
        B_lat = group_df['lat'].values

        try:
            M_lon, _, _, _ = np.linalg.lstsq(A, B_lon, rcond=None)
            M_lat, _, _, _ = np.linalg.lstsq(A, B_lat, rcond=None)
            
            print(f"  ✅ 成功解出 imageId {image_id} 的轉換矩陣。")
            # ★ 儲存為可 JSON 序列化的 list
            transformation_matrices[image_id] = (M_lon.tolist(), M_lat.tolist())

            origin_lat = group_df['lat'].min()
            origin_lon = group_df['lon'].min()
            coordinate_origins[image_id] = (origin_lat, origin_lon)
            print(f"  - 座標原點 (Origin) 設為: ({origin_lat:.6f}, {origin_lon:.6f})")

        except np.linalg.LinAlgError as e:
            print(f"  ❌ 錯誤: 解矩陣失敗 for imageId {image_id}: {e}")
            continue

    # --- 4. ★★★ 儲存轉換資料到 JSON 檔案 ★★★ ---
    print("\n--- 正在儲存轉換設定檔 ---")
    if not transformation_matrices:
        print("❌ 錯誤：沒有成功計算任何轉換矩陣，無法儲存設定檔。")
        return False
        
    transform_data = {
        'matrices': transformation_matrices,
        'origins': coordinate_origins
    }
    
    try:
        # 確保 processed_data 資料夾存在
        os.makedirs(os.path.dirname(OUTPUT_TRANSFORM_CONFIG), exist_ok=True)
        with open(OUTPUT_TRANSFORM_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(transform_data, f, indent=2)
        print(f"  ✅ 成功！已將轉換公式儲存至: {OUTPUT_TRANSFORM_CONFIG}")
        return True
    except Exception as e:
        print(f"  ❌ 錯誤: 儲存轉換 JSON 失敗: {e}")
        return False

# (主函數)
def main():
    # 步驟 1: 解析 XML
    xml_coords = parse_xml_coordinates(XML_FILE)
    if xml_coords is None:
        return

    # 步驟 2: 載入 JSON 並預處理
    df_points = load_and_preprocess_json(JSON_FILE)
    if df_points is None:
        return
        
    # 步驟 3: 計算並儲存轉換公式
    success = calculate_and_save_transforms(df_points, xml_coords)
    
    if success:
        print("\n下一步：")
        print("1. 已產生 'processed_data/transformation_data.json'。")
        print("2. 請用我接下來給的 `create_vector_format.py` (v3) 覆蓋掉舊檔案。")
        print("3. 然後，直接執行 `python preprocess/run_preprocessing.py` 即可！")

if __name__ == "__main__":
    main()