import os
import numpy as np
import tensorflow as tf
import pandas as pd
from model_improvements import AttentionLayer

MODEL_PATH = os.path.join("enhanced_models", "enhanced_hadnn_model.h5")
BSSID_CSV_PATH = os.path.join("processed_data", "bssid_mapping.csv")

def load_bssid_mapping(csv_path):
    df = pd.read_csv(csv_path)
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
    if not os.path.exists(BSSID_CSV_PATH):
        print(f"找不到 BSSID 映射檔: {BSSID_CSV_PATH}")
        return
    bssid_list = load_bssid_mapping(BSSID_CSV_PATH)
    print(f"已載入 {len(bssid_list)} 個 BSSID 特徵")

    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型檔: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})
    
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

    rss_vector = build_rss_vector(bssid_list, input_pairs)
    input_data = np.expand_dims(rss_vector, axis=0).astype(np.float32)
    preds = model.predict(input_data)
    # 預設 HADNN 輸出: [building_softmax, floor_softmax, xy]
    if isinstance(preds, list) and len(preds) >= 3:
        building_logits, floor_logits, xy_pred = preds
        pred_building = int(np.argmax(building_logits[0]))
        pred_floor = int(np.argmax(floor_logits[0]))
        pred_xy = xy_pred[0]
        print(f"預測座標: x = {pred_xy[0]:.2f}, y = {pred_xy[1]:.2f}")
        print(f"預測建築ID: {pred_building}")
        print(f"預測樓層ID: {pred_floor}")
    else:
        print("模型輸出格式不符，無法取得建築/樓層預測")
        print(f"模型原始輸出: {preds}")

if __name__ == "__main__":
    main()
