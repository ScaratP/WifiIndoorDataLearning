import os
import numpy as np
import tensorflow as tf
import pandas as pd

# 設定檔案路徑
TFLITE_MODEL_PATH = os.path.join("tflite_models", "enhanced_hadnn_model.tflite")
BSSID_CSV_PATH = os.path.join("processed_data", "bssid_mapping.csv")

def load_bssid_mapping(csv_path):
    df = pd.read_csv(csv_path)
    # 支援不同欄位名稱
    if 'bssid' in df.columns:
        bssid_list = [b.lower() for b in df['bssid']]
    else:
        bssid_list = [b.lower() for b in df.iloc[:,1]]
    return bssid_list

def build_rss_vector(bssid_list, input_pairs):
    rss_vector = np.full(len(bssid_list), -100, dtype=np.float32)
    for bssid, rssi in input_pairs:
        bssid = bssid.lower()
        if bssid in bssid_list:
            idx = bssid_list.index(bssid)
            try:
                rss_vector[idx] = float(rssi)
            except:
                pass
    return rss_vector

def main():
    # 載入 BSSID 映射
    if not os.path.exists(BSSID_CSV_PATH):
        print(f"找不到 BSSID 映射檔: {BSSID_CSV_PATH}")
        return
    bssid_list = load_bssid_mapping(BSSID_CSV_PATH)
    print(f"已載入 {len(bssid_list)} 個 BSSID 特徵")
    
    # 載入 TFLite 模型
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"找不到 TFLite 模型: {TFLITE_MODEL_PATH}")
        return
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("請逐行輸入 bssid rssi（中間空格），輸入空白行開始計算位置：")
    input_pairs = []
    while True:
        line = input().strip()
        if not line:
            break
        parts = line.split()
        if len(parts) == 2:
            input_pairs.append((parts[0], parts[1]))
        else:
            print("格式錯誤，請輸入：bssid rssi")
    
    if not input_pairs:
        print("未輸入任何資料")
        return
    
    # 組裝 RSS 向量
    rss_vector = build_rss_vector(bssid_list, input_pairs)
    # TFLite 輸入通常需要 batch 維度
    input_data = np.expand_dims(rss_vector, axis=0).astype(np.float32)
    
    # 設定輸入
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 取得輸出
    # 假設 output_details[0] 是建築 softmax, [1] 是樓層 softmax, [2] 是位置 (x, y)
    output_data = interpreter.get_tensor(output_details[-1]['index'])
    pred_xy = output_data[0]
    print(f"預測座標: x = {pred_xy[0]:.2f}, y = {pred_xy[1]:.2f}")

    # 預測建築
    if len(output_details) >= 3:
        building_logits = interpreter.get_tensor(output_details[0]['index'])
        floor_logits = interpreter.get_tensor(output_details[1]['index'])
        pred_building = int(np.argmax(building_logits[0]))
        pred_floor = int(np.argmax(floor_logits[0]))
        print(f"預測建築ID: {pred_building}")
        print(f"預測樓層ID: {pred_floor}")
    else:
        print("模型輸出格式不符，無法取得建築/樓層預測")

if __name__ == "__main__":
    main()
