import os
import sys
import numpy as np  # 添加 NumPy 導入

# 添加路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_and_filter import extract_wifi_data_with_filter
from create_vector_format import create_vector_format, create_building_floor_point_labels, save_to_csv
from hadnn_adapter import prepare_for_hadnn
import json
import datetime

def main():
    """主要預處理流程"""
    print("=== NTTU WiFi 資料預處理 (支援點分類與位置回歸) ===")
    
    # 記錄開始時間
    start_time = datetime.datetime.now()
    
    # 初始化收集預處理資訊的字典
    preprocessing_info = {
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_info": {},
        "data_stats": {},
        "building_floor_stats": {},
        "point_stats": {},
        "signal_stats": {}
    }
    
    # 設定路徑
    scan13_folder = "../points/scan13"
    processed_dir = "../processed_data"
    hadnn_dir = "../hadnn_data"
    
    # 設定目標 SSID
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam', 'nttu'}
    preprocessing_info["target_ssids"] = list(target_ssids)
    
    print(f"1. 從 {scan13_folder} 提取資料...")
    
    # 步驟 1: 提取並過濾資料
    extracted_data, file_info = extract_wifi_data_with_filter(scan13_folder, target_ssids)
    
    # 儲存檔案處理資訊
    preprocessing_info["file_info"] = file_info
    
    # 印出檔案處理資訊
    print(f"   處理了 {file_info['total_files']} 個檔案")
    print(f"   總記錄數: {file_info['total_records']}")
    print(f"   有效記錄數: {file_info['valid_records']}")
    print(f"   提取了 {len(extracted_data)} 個參考點")
    
    # 步驟 2: 轉換為向量格式
    print("2. 轉換為向量格式...")
    vector_data, bssid_info = create_vector_format(extracted_data, target_ssids)
    print(f"   建立了 {len(bssid_info)} 個 BSSID 的向量")
    
    preprocessing_info["data_stats"]["num_reference_points"] = len(extracted_data)
    preprocessing_info["data_stats"]["num_bssids"] = len(bssid_info)
    preprocessing_info["data_stats"]["bssid_info"] = bssid_info[:5] + ["..."] if len(bssid_info) > 5 else bssid_info  # 僅保存部分BSSID資訊
    
    # 步驟 3: 建立標籤
    print("3. 建立建築物、樓層和點位標籤...")
    vector_data, building_mapping, floor_mapping, point_mapping = create_building_floor_point_labels(vector_data)
    print(f"   建築物: {list(building_mapping.keys())}")
    print(f"   樓層: {list(floor_mapping.keys())}")
    print(f"   點位: {len(point_mapping)} 個")
    
    # 計算建築物和樓層分布比例
    building_floor_stats = calculate_building_floor_distribution(vector_data)
    preprocessing_info["building_floor_stats"] = building_floor_stats
    
    # 顯示建築物分布比例
    print("\n   建築物分布比例:")
    for building, stats in building_floor_stats["buildings"].items():
        print(f"   - {building}: {stats['count']} 個點位 ({stats['percentage']:.2f}%)")
    
    # 顯示樓層分布比例
    print("\n   樓層分布比例:")
    for floor, stats in building_floor_stats["floors"].items():
        print(f"   - {floor}樓: {stats['count']} 個點位 ({stats['percentage']:.2f}%)")
    
    preprocessing_info["building_mapping"] = building_mapping
    preprocessing_info["floor_mapping"] = floor_mapping
    preprocessing_info["point_mapping_count"] = len(point_mapping)
    
    # 顯示點位類型分佈
    room_points = [p for p in point_mapping.keys() if 'CORRIDOR' not in p]
    corridor_points = [p for p in point_mapping.keys() if 'CORRIDOR' in p]
    print(f"   - 房間點位: {len(room_points)} 個")
    print(f"   - 走廊點位: {len(corridor_points)} 個")
    if corridor_points:
        print(f"   - 走廊點位範例: {corridor_points[:5]}")
    
    preprocessing_info["point_stats"]["room_points_count"] = len(room_points)
    preprocessing_info["point_stats"]["corridor_points_count"] = len(corridor_points)
    preprocessing_info["point_stats"]["corridor_examples"] = corridor_points[:5] if corridor_points else []
    
    # 步驟 4: 儲存為 CSV
    print("4. 儲存為 CSV 格式...")
    save_to_csv(vector_data, bssid_info, processed_dir)
    
    # 儲存映射資訊
    with open(os.path.join(processed_dir, 'label_mappings.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'building_mapping': building_mapping,
            'floor_mapping': floor_mapping,
            'point_mapping': point_mapping
        }, f, ensure_ascii=False, indent=2)
    
    # 步驟 5: 準備 HADNN 模型格式
    print("5. 準備 HADNN 模型格式...")
    print("   注意：如果看到與縮放相關的警告，系統會嘗試使用更穩健的方法")
    dataset = prepare_for_hadnn(processed_dir, hadnn_dir)
    
    # 步驟 6: 執行高級資料品質分析
    print("6. 執行高級資料品質分析...")
    analysis_results = analyze_data_quality(dataset, hadnn_dir)
    preprocessing_info["data_quality_analysis"] = analysis_results
    
    # 記錄結束時間和處理耗時
    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    preprocessing_info["processing_time"] = {
        "seconds": processing_time,
        "formatted": str(datetime.timedelta(seconds=processing_time))
    }
    
    print("\n=== 預處理完成 ===")
    print(f"總處理時間: {preprocessing_info['processing_time']['formatted']}")
    print(f"處理後資料位置:")
    print(f"  CSV 格式: {processed_dir}")
    print(f"  HADNN 格式: {hadnn_dir}")
    print(f"\n資料摘要:")
    print(f"  參考點數量: {len(vector_data)}")
    print(f"  RSS 特徵數: {dataset.n_rss}")
    print(f"  建築物數: {dataset.n_buildings}")
    print(f"  樓層數: {dataset.n_floors}")
    print(f"  點位數: {dataset.n_points}")
    
    # 位置回歸相關統計
    print(f"\n位置回歸統計:")
    print(f"  原始座標範圍: x=[{dataset.coordinates[:, 0].min():.2f}, {dataset.coordinates[:, 0].max():.2f}], "f"y=[{dataset.coordinates[:, 1].min():.2f}, {dataset.coordinates[:, 1].max():.2f}]")
    print(f"  標準化後座標範圍: x=[{dataset.train_c[:, 0].min():.2f}, {dataset.train_c[:, 0].max():.2f}], "f"y=[{dataset.train_c[:, 1].min():.2f}, {dataset.train_c[:, 1].max():.2f}]")
    
    # 添加數據品質檢查信息
    missing_percentage = (dataset.rss_data == -100).mean() * 100
    preprocessing_info["signal_stats"]["missing_percentage"] = float(missing_percentage)
    preprocessing_info["signal_stats"]["rss_range"] = {
        "min": float(dataset.rss_data.min()),
        "max": float(dataset.rss_data.max())
    }
    preprocessing_info["signal_stats"]["normalized_range"] = {
        "min": float(dataset.train_x.min()),
        "max": float(dataset.train_x.max())
    }
    
    print(f"\n資料品質檢查:")
    print(f"  訊號缺失比例: {missing_percentage:.2f}%")
    print(f"  RSS 值範圍: {dataset.rss_data.min()} 到 {dataset.rss_data.max()}")
    print(f"  標準化後訓練資料範圍: {dataset.train_x.min():.2f} 到 {dataset.train_x.max():.2f}")
    print(f"  標準差接近零的特徵比例: {(np.std(dataset.rss_data, axis=0) < 0.1).mean() * 100:.2f}%")
    
    print("\n模型訓練建議:")
    print("  1. 點分類模型: 使用 RSS 特徵預測 point_id (包含房間和走廊)")
    print("  2. 位置回歸模型: 使用 RSS 特徵預測 (x, y) 座標")
    print("  3. 階層模型: 先預測建築物，再預測樓層，最後預測精確位置")
    print("  4. 考慮房間/走廊分類: 走廊通常有更複雜的信號傳播特性")
    
    # 保存完整的預處理資訊
    preprocessing_info_path = os.path.join(processed_dir, 'preprocessing_info.json')
    with open(preprocessing_info_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n預處理資訊已保存至: {preprocessing_info_path}")
    
    # 生成簡易報告文本
    generate_preprocessing_report(preprocessing_info, processed_dir)

def calculate_building_floor_distribution(vector_data):
    """計算建築物和樓層分布比例"""
    total_points = len(vector_data)
    if total_points == 0:
        return {"buildings": {}, "floors": {}}
    
    # 計算建築物分布
    building_counts = {}
    for record in vector_data:
        building = record.get('building', 'unknown')
        if building not in building_counts:
            building_counts[building] = 0
        building_counts[building] += 1
    
    building_stats = {}
    for building, count in building_counts.items():
        building_stats[building] = {
            "count": count,
            "percentage": (count / total_points) * 100
        }
    
    # 計算樓層分布
    floor_counts = {}
    for record in vector_data:
        floor = record.get('floor', 'unknown')
        if floor not in floor_counts:
            floor_counts[floor] = 0
        floor_counts[floor] += 1
    
    floor_stats = {}
    for floor, count in floor_counts.items():
        floor_stats[str(floor)] = {
            "count": count,
            "percentage": (count / total_points) * 100
        }
    
    # 計算建築物-樓層組合分布
    building_floor_counts = {}
    for record in vector_data:
        building = record.get('building', 'unknown')
        floor = record.get('floor', 'unknown')
        key = f"{building}-{floor}"
        if key not in building_floor_counts:
            building_floor_counts[key] = 0
        building_floor_counts[key] += 1
    
    building_floor_combo_stats = {}
    for combo, count in building_floor_counts.items():
        building_floor_combo_stats[combo] = {
            "count": count,
            "percentage": (count / total_points) * 100
        }
    
    return {
        "total_points": total_points,
        "buildings": building_stats,
        "floors": floor_stats,
        "building_floor_combinations": building_floor_combo_stats
    }

def generate_preprocessing_report(info, output_dir):
    """生成簡易的預處理報告文本"""
    report_path = os.path.join(output_dir, 'preprocessing_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== NTTU WiFi 資料預處理報告 =====\n\n")
        f.write(f"預處理時間: {info['timestamp']}\n")
        f.write(f"處理耗時: {info['processing_time']['formatted']}\n\n")
        
        f.write("=== 檔案處理資訊 ===\n")
        f.write(f"處理檔案總數: {info['file_info']['total_files']}\n")
        f.write(f"總記錄數: {info['file_info']['total_records']}\n")
        f.write(f"有效記錄數: {info['file_info']['valid_records']}\n")
        f.write(f"提取參考點數: {info['data_stats']['num_reference_points']}\n")
        f.write(f"BSSID 數量: {info['data_stats']['num_bssids']}\n\n")
        
        f.write("=== 建築物和樓層分布 ===\n")
        for building, stats in info['building_floor_stats']['buildings'].items():
            f.write(f"建築物 {building}: {stats['count']} 個點位 ({stats['percentage']:.2f}%)\n")
        
        f.write("\n")
        for floor, stats in info['building_floor_stats']['floors'].items():
            f.write(f"{floor}樓: {stats['count']} 個點位 ({stats['percentage']:.2f}%)\n")
        
        f.write("\n=== 點位統計 ===\n")
        f.write(f"點位總數: {info['point_mapping_count']}\n")
        f.write(f"房間點位: {info['point_stats']['room_points_count']}\n")
        f.write(f"走廊點位: {info['point_stats']['corridor_points_count']}\n\n")
        
        f.write("=== 信號統計 ===\n")
        f.write(f"訊號缺失比例: {info['signal_stats']['missing_percentage']:.2f}%\n")
        f.write(f"RSS 值範圍: {info['signal_stats']['rss_range']['min']} 到 {info['signal_stats']['rss_range']['max']}\n")
        
    print(f"預處理簡易報告已保存至: {report_path}")

def analyze_data_quality(dataset, output_dir):
    """執行更深入的資料品質分析，識別可能的問題"""
    analysis_results = {}
    
    # 1. 檢查每棟建築物的樣本數量
    # 將浮點型標籤轉換為整數類型
    building_labels_int = dataset.building_labels.astype(np.int64)
    building_counts = np.bincount(building_labels_int)
    analysis_results['building_counts'] = {
        'building_ids': list(range(len(building_counts))),
        'counts': building_counts.tolist()
    }
    
    # 檢查不平衡問題
    min_building_count = np.min(building_counts)
    max_building_count = np.max(building_counts)
    imbalance_ratio = max_building_count / min_building_count if min_building_count > 0 else float('inf')
    analysis_results['building_imbalance'] = {
        'min_count': int(min_building_count),
        'max_count': int(max_building_count),
        'imbalance_ratio': float(imbalance_ratio)
    }
    
    # 2. 分析每棟建築物的 WiFi 信號品質
    building_signal_quality = {}
    for b_id in range(len(building_counts)):
        if building_counts[b_id] == 0:
            continue
        
        # 獲取該建築的樣本 (使用整數標籤)
        b_indices = np.where(building_labels_int == b_id)[0]
        b_data = dataset.rss_data[b_indices]
        
        # 計算信號缺失率
        missing_rate = (b_data == -100).mean()
        
        # 計算信號強度統計
        valid_signals = b_data[b_data != -100]
        if len(valid_signals) > 0:
            mean_signal = np.mean(valid_signals)
            std_signal = np.std(valid_signals)
        else:
            mean_signal = 0
            std_signal = 0
        
        building_signal_quality[b_id] = {
            'missing_rate': float(missing_rate),
            'mean_signal': float(mean_signal),
            'std_signal': float(std_signal),
            'sample_count': int(len(b_indices))
        }
    
    analysis_results['building_signal_quality'] = building_signal_quality
    
    # 3. 識別區分建築物的關鍵 AP
    # 計算每個 AP 在區分建築物上的重要性
    importance_scores = []
    
    for ap_idx in range(dataset.rss_data.shape[1]):
        ap_data = dataset.rss_data[:, ap_idx]
        
        # 計算每棟建築物該 AP 的平均信號
        ap_building_means = []
        for b_id in range(len(building_counts)):
            if building_counts[b_id] == 0:
                ap_building_means.append(-100)
                continue
                
            # 使用整數標籤
            b_indices = np.where(building_labels_int == b_id)[0]
            b_ap_data = ap_data[b_indices]
            b_ap_valid = b_ap_data[b_ap_data != -100]
            
            if len(b_ap_valid) > 0:
                ap_building_means.append(float(np.mean(b_ap_valid)))
            else:
                ap_building_means.append(-100)
        
        # 計算建築物間信號差異
        ap_building_means = np.array(ap_building_means)
        valid_means = ap_building_means[ap_building_means != -100]
        
        if len(valid_means) >= 2:  # 至少需要兩棟建築物有該 AP 的信號
            # 計算建築物間信號差異的方差
            variance = np.var(valid_means)
            # 計算信號存在率
            existence_rate = len(valid_means) / len(ap_building_means)
            # 綜合得分 = 方差 * 存在率
            importance = variance * existence_rate
        else:
            importance = 0
        
        importance_scores.append({
            'ap_idx': ap_idx,
            'importance': float(importance),
            'building_means': ap_building_means.tolist()
        })
    
    # 根據重要性排序
    importance_scores.sort(key=lambda x: x['importance'], reverse=True)
    analysis_results['important_aps'] = importance_scores[:20]  # 保留前20個最重要的 AP
    
    # 將分析結果保存為 JSON
    with open(os.path.join(output_dir, 'data_quality_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)
    
    # 生成報告摘要
    print("\n資料品質分析摘要:")
    print(f"  - 建築物樣本數平衡度: {1/imbalance_ratio:.2f} (1.0 為完全平衡)")
    
    # 找出信號品質最差的建築物
    worst_building_id = min(building_signal_quality.items(), key=lambda x: x[1]['mean_signal'])[0]
    print(f"  - 信號品質最差的建築物: {worst_building_id}, "f"缺失率: {building_signal_quality[worst_building_id]['missing_rate']:.2%}, "f"平均信號: {building_signal_quality[worst_building_id]['mean_signal']:.1f} dBm")
    
    # 用於建築物分類的前5個關鍵 AP
    print("  - 建築物分類的關鍵 AP 索引:", [ap['ap_idx'] for ap in importance_scores[:5]])
    
    return analysis_results

if __name__ == "__main__":
    main()
