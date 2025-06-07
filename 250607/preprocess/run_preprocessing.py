import os
import sys
import numpy as np  # 添加 NumPy 導入

# 添加路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_and_filter import extract_wifi_data_with_filter
from create_vector_format import create_vector_format, create_building_floor_labels, save_to_csv
from hadnn_adapter import prepare_for_hadnn
import json

def main():
    """主要預處理流程"""
    print("=== NTTU WiFi 資料預處理 ===")
    
    # 設定路徑
    scan13_folder = "../points/scan13"
    processed_dir = "../processed_data"
    hadnn_dir = "../hadnn_data"
    
    # 設定目標 SSID
    target_ssids = {'ap-nttu', 'ap2-nttu', 'eduroam'}
    
    print(f"1. 從 {scan13_folder} 提取資料...")
    
    # 步驟 1: 提取並過濾資料
    extracted_data = extract_wifi_data_with_filter(scan13_folder, target_ssids)
    print(f"   提取了 {len(extracted_data)} 個參考點")
    
    # 步驟 2: 轉換為向量格式
    print("2. 轉換為向量格式...")
    vector_data, bssid_info = create_vector_format(extracted_data, target_ssids)
    print(f"   建立了 {len(bssid_info)} 個 BSSID 的向量")
    
    # 步驟 3: 建立標籤
    print("3. 建立建築物和樓層標籤...")
    vector_data, building_mapping, floor_mapping = create_building_floor_labels(vector_data)
    print(f"   建築物: {list(building_mapping.keys())}")
    print(f"   樓層: {list(floor_mapping.keys())}")
    
    # 步驟 4: 儲存為 CSV
    print("4. 儲存為 CSV 格式...")
    save_to_csv(vector_data, bssid_info, processed_dir)
    
    # 儲存映射資訊
    with open(os.path.join(processed_dir, 'label_mappings.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'building_mapping': building_mapping,
            'floor_mapping': floor_mapping
        }, f, ensure_ascii=False, indent=2)
    
    # 步驟 5: 準備 HADNN 格式
    print("5. 準備 HADNN 模型格式...")
    print("   注意：如果看到與縮放相關的警告，系統會嘗試使用更穩健的方法")
    dataset = prepare_for_hadnn(processed_dir, hadnn_dir)
    
    # 【新增】步驟 6: 執行高級資料品質分析
    print("6. 執行高級資料品質分析...")
    analyze_data_quality(dataset, hadnn_dir)
    
    print("\n=== 預處理完成 ===")
    print(f"處理後資料位置:")
    print(f"  CSV 格式: {processed_dir}")
    print(f"  HADNN 格式: {hadnn_dir}")
    print(f"\n資料摘要:")
    print(f"  參考點數量: {len(vector_data)}")
    print(f"  RSS 特徵數: {dataset.n_rss}")
    print(f"  建築物數: {dataset.n_buildings}")
    print(f"  樓層數: {dataset.n_floors}")
    
    # 添加數據質量檢查信息
    print(f"\n資料質量檢查:")
    missing_percentage = (dataset.rss_data == -100).mean() * 100
    print(f"  訊號缺失比例: {missing_percentage:.2f}%")
    print(f"  RSS 值範圍: {dataset.rss_data.min()} 到 {dataset.rss_data.max()}")
    print(f"  標準化後訓練資料範圍: {dataset.train_x.min():.2f} 到 {dataset.train_x.max():.2f}")
    print(f"  標準差接近零的特徵比例: {(np.std(dataset.rss_data, axis=0) < 0.1).mean() * 100:.2f}%")
    
    print("\n注意：如果仍然看到標準化相關警告，這通常不會影響定位性能，")
    print("      因為神經網路可以適應不同尺度的特徵。")
    print("\n若建築物分類準確度仍不理想，請考慮使用改進的模型架構，")
    print("執行 model_improvements.py 來訓練提升的模型。")

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
    
    # 找出信號質量最差的建築物
    worst_building_id = min(building_signal_quality.items(), 
                           key=lambda x: x[1]['mean_signal'])[0]
    print(f"  - 信號質量最差的建築物: {worst_building_id}, "
          f"缺失率: {building_signal_quality[worst_building_id]['missing_rate']:.2%}, "
          f"平均信號: {building_signal_quality[worst_building_id]['mean_signal']:.1f} dBm")
    
    # 用於建築物分類的前5個關鍵 AP
    print("  - 建築物分類的關鍵 AP 索引:", 
          [ap['ap_idx'] for ap in importance_scores[:5]])
    
    return analysis_results

if __name__ == "__main__":
    main()
