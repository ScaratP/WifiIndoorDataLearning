import json
import os

def filter_wifi_data(input_file, output_file):
    # 讀取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 要篩選的SSID
    target_ssids = ['Ap-nttu', 'Ap2-nttu', 'NTTU', 'eduroam']
    
    # 篩選結果
    filtered_data = []
    
    # 處理每個定位點
    for point in data:
        # 複製定位點資訊
        filtered_point = dict(point)
        
        # 篩選wifi讀數
        if 'wifiReadings' in point:
            filtered_readings = []
            
            for reading in point['wifiReadings']:
                # 檢查是否有任何屬性包含目標SSID/BSSID
                match_found = False
                
                # 檢查SSID
                if 'ssid' in reading and reading['ssid'] in target_ssids:
                    match_found = True
                # 檢查BSSID
                elif 'bssid' in reading and any(target in reading['bssid'] for target in target_ssids):
                    match_found = True
                
                # 如果找到匹配，則添加到過濾結果中
                if match_found:
                    filtered_readings.append(reading)
            
            # 更新該點的wifi讀數
            filtered_point['wifiReadings'] = filtered_readings
        
        # 將處理後的點加入結果
        filtered_data.append(filtered_point)
    
    # 寫入輸出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"篩選完成，已保存到: {output_file}")

if __name__ == "__main__":
    input_file = "c:\\Users\\nttucsie\\Desktop\\ap_data\\points\\scan13\\NTTUSEWifiIndoorPoints\\se_all_13.json"
    output_file = "c:\\Users\\nttucsie\\Desktop\\ap_data\\points\\scan13\\NTTUSEWifiIndoorPoints\\se_all_13_filtered.json"
    
    # 檢查輸入文件是否存在
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到輸入檔案 {input_file}")
        exit(1)
    
    filter_wifi_data(input_file, output_file)
