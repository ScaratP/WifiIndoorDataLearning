# 檔案：preprocess/run_preprocessing.py (修改後版本)

import os
import sys
import numpy as np
import json
import datetime

# --- 修改 1：導入新的函數 ---
# 添加路徑 (確保能找到 preprocess 資料夾內的模組)
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 如果 run_preprocessing.py 在 preprocess 外層，這行可能需要

from extract_and_filter import extract_wifi_data_with_filter
from create_vector_format import create_vector_format, create_building_floor_point_labels, save_to_csv
# 從 hadnn_adapter 導入新的 prepare_split_data 函數
from hadnn_adapter import prepare_split_data
# 舊的 prepare_for_hadnn 不再需要，可以移除 import (如果你之前有 import 的話)

def main():
    """主要預處理流程 (已更新為 1+8 模型架構)"""
    print("=== NTTU WiFi 資料預處理 (1+8 模型架構) ===")

    start_time = datetime.datetime.now()

    # --- 保持不變：初始化資訊收集 ---
    preprocessing_info = {
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "1+8 Split Model (Classifier + 8 Regressors)", # 標註架構
        "file_info": {},
        "data_stats": {},
        "building_floor_stats": {},
        "point_stats": {},
        "signal_stats": {}
    }

    # --- 保持不變：設定路徑 ---
    # 假設 run_preprocessing.py 在 preprocess 的上一層目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scan13_folder = os.path.join(current_dir, "../points/scan13")
    processed_dir = os.path.join(current_dir, "../processed_data")
    # --- 修改 2：指定新的輸出資料夾 ---
    hadnn_split_dir = os.path.join(current_dir, "../hadnn_data_split") # 新的輸出資料夾

    # --- 保持不變：設定目標 SSID ---
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'nttu'}
    preprocessing_info["target_ssids"] = list(target_ssids)

    print(f"\n[步驟 1/5] 從 {scan13_folder} 提取資料...")
    extracted_data, file_info = extract_wifi_data_with_filter(scan13_folder, target_ssids)
    preprocessing_info["file_info"] = file_info
    print(f"   處理了 {file_info.get('total_files', 0)} 個檔案")
    print(f"   提取了 {len(extracted_data)} 個有效參考點")

    if not extracted_data:
        print("❌ 錯誤：未能提取任何有效資料，預處理中止。")
        return

    print("\n[步驟 2/5] 轉換為向量格式...")
    vector_data, bssid_info = create_vector_format(extracted_data, target_ssids)
    print(f"   建立了 {len(bssid_info)} 個 BSSID 的向量")
    preprocessing_info["data_stats"]["num_reference_points"] = len(vector_data) # 使用 vector_data 的長度
    preprocessing_info["data_stats"]["num_bssids"] = len(bssid_info)

    print("\n[步驟 3/5] 建立標籤映射...")
    vector_data, building_mapping, floor_mapping, point_mapping = create_building_floor_point_labels(vector_data)
    print(f"   建築物: {list(building_mapping.keys())}")
    print(f"   樓層: {list(floor_mapping.keys())}")
    print(f"   點位: {len(point_mapping)} 個")
    preprocessing_info["building_mapping"] = building_mapping
    preprocessing_info["floor_mapping"] = floor_mapping
    preprocessing_info["point_mapping_count"] = len(point_mapping)

    print("\n[步驟 4/5] 儲存為 CSV 格式...")
    save_to_csv(vector_data, bssid_info, processed_dir)
    # 儲存映射資訊 (保持不變)
    label_map_path = os.path.join(processed_dir, 'label_mappings.json')
    try:
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump({
                'building_mapping': building_mapping,
                'floor_mapping': floor_mapping,
                'point_mapping': point_mapping
            }, f, ensure_ascii=False, indent=2)
        print(f"   標籤映射已儲存至: {label_map_path}")
    except Exception as e:
        print(f"   ❌ 錯誤: 無法儲存標籤映射檔案: {e}")


    # --- 修改 3：呼叫新的資料分割函數 ---
    print(f"\n[步驟 5/5] 準備 1+8 模型格式 (輸出至 {hadnn_split_dir})...")
    try:
        # 呼叫 hadnn_adapter.py 中的 prepare_split_data
        prepare_split_data(processed_dir, hadnn_split_dir)
        print("   ✅ 1+8 模型資料準備完成。")
    except Exception as e:
        print(f"   ❌ 錯誤: 在準備 1+8 模型資料時發生問題: {e}")
        import traceback
        traceback.print_exc() # 印出詳細錯誤訊息
        print("   預處理中止。")
        return # 如果這步失敗，後續無法進行

    # --- 修改 4：暫時註解掉舊的分析和報告部分 ---
    # (因為 analyze_data_quality 和 generate_preprocessing_report 可能需要重寫以適應新結構)
    # print("\n[步驟 6/X] 執行高級資料品質分析...(暫時跳過)")
    # # analysis_results = analyze_data_quality(dataset, hadnn_dir) # 舊的 dataset 物件可能不存在了
    # # preprocessing_info["data_quality_analysis"] = analysis_results

    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    preprocessing_info["processing_time"] = {
        "seconds": processing_time,
        "formatted": str(datetime.timedelta(seconds=processing_time))
    }

    print("\n=== ✅ 預處理流程完成 ===")
    print(f"總處理時間: {preprocessing_info['processing_time']['formatted']}")
    print(f"處理後資料位置:")
    print(f"  CSV & BSSID 映射: {processed_dir}")
    print(f"  1+8 模型 NumPy 檔 & 設定檔: {hadnn_split_dir}") # 指向新資料夾

    # --- 簡化報告儲存 ---
    preprocessing_info_path = os.path.join(processed_dir, 'preprocessing_info_split.json') # 使用新檔名
    try:
        # 只儲存基本資訊
        simple_info = {
            "timestamp": preprocessing_info["timestamp"],
            "architecture": preprocessing_info["architecture"],
            "target_ssids": preprocessing_info["target_ssids"],
            "num_reference_points": preprocessing_info["data_stats"]["num_reference_points"],
            "num_bssids": preprocessing_info["data_stats"]["num_bssids"],
            "processing_time": preprocessing_info["processing_time"]
        }
        with open(preprocessing_info_path, 'w', encoding='utf-8') as f:
            json.dump(simple_info, f, ensure_ascii=False, indent=2)
        print(f"\n簡易預處理資訊已保存至: {preprocessing_info_path}")
    except Exception as e:
        print(f"   ⚠️ 警告: 無法儲存簡易預處理資訊檔案: {e}")

if __name__ == "__main__":
    main()