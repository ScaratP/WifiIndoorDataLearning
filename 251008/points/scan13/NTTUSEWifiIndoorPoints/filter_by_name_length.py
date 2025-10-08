import json
import os

def filter_by_name_length(input_file, output_file):
    # 讀取JSON檔案
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 篩選結果
    filtered_data = []
    
    # 處理每個項目
    for item in data:
        # 如果項目中有name欄位且長度大於4
        if 'name' in item and isinstance(item['name'], str) and len(item['name']) > 4:
            # 只保留需要的欄位
            filtered_item = {
                'id': item.get('id', ''),
                'name': item.get('name', ''),
                'x': item.get('x', 0),
                'y': item.get('y', 0),
                'imageId': item.get('imageId', 0)
            }
            filtered_data.append(filtered_item)
    
    # 寫入輸出檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"篩選完成，已保存到: {output_file}")
    print(f"共篩選出 {len(filtered_data)} 條記錄")

if __name__ == "__main__":
    input_file = "c:\\Users\\nttucsie\\Desktop\\ap_data\\points\\scan13\\NTTUSEWifiIndoorPoints\\se_all_13.json"
    output_file = "c:\\Users\\nttucsie\\Desktop\\ap_data\\points\\scan13\\NTTUSEWifiIndoorPoints\\classroom_points.json"
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到輸入檔案 {input_file}")
        exit(1)
    
    filter_by_name_length(input_file, output_file)
