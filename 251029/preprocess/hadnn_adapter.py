# 檔案：preprocess/hadnn_adapter.py (更新 define_groups 邏輯)

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# --- 新增/修改：定義你的新群組規則 ---
def define_groups(df):
    """
    根據 building 和 floor 欄位，建立新的群組規則：
    - 1樓 (不分建築) -> se1
    - 2樓 (不分建築) -> se2
    - 3樓 (不分建築) -> se3
    - sea4, sea5, seb4, sec4, sec5 維持不變
    回傳 DataFrame 新增 'group' 欄位。
    """
    print("   正在定義資料群組 (合併 1-3 樓)...")

    # 1. 建立 'group' 欄位，預設 'other'
    df['group'] = 'other'

    # 2. 定義新的群組規則
    try:
        # 確保 floor 是整數型態
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
        df.dropna(subset=['floor'], inplace=True)
        df['floor'] = df['floor'].astype(int)

        # --- 新規則 ---
        # 合併所有建築的 1 樓為 se1
        df.loc[df['floor'] == 1, 'group'] = 'se1'
        # 合併所有建築的 2 樓為 se2
        df.loc[df['floor'] == 2, 'group'] = 'se2'
        # 合併所有建築的 3 樓為 se3
        df.loc[df['floor'] == 3, 'group'] = 'se3'

        # --- 維持不變的規則 ---
        df.loc[(df['building'] == 'sea') & (df['floor'] == 4), 'group'] = 'sea4'
        df.loc[(df['building'] == 'sea') & (df['floor'] == 5), 'group'] = 'sea5'
        df.loc[(df['building'] == 'seb') & (df['floor'] == 4), 'group'] = 'seb4'
        df.loc[(df['building'] == 'sec') & (df['floor'] == 4), 'group'] = 'sec4'
        df.loc[(df['building'] == 'sec') & (df['floor'] == 5), 'group'] = 'sec5'

    except Exception as e:
        print(f"   錯誤：在定義群組時發生問題：{e}")
        print("   請檢查 nttu_wifi_data.csv 中的 'building' 和 'floor' 欄位是否正確。")
        raise

    print("   ✅ 新群組定義完成。")
    return df

# --- 主要的 prepare_split_data 函數 (保持不變) ---
def prepare_split_data(data_path, output_dir):
    """
    為 1+N 模型（1個分類器 + N個回歸器）準備資料。
    輸出檔案將儲存到 output_dir。 (N 會根據 define_groups 的結果自動決定)
    """
    print(f"--- 開始為 1+N 模型準備資料 ---")
    print(f"來源資料夾: {data_path}")
    print(f"輸出資料夾: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 載入 CSV 並定義群組 ---
    main_csv_path = os.path.join(data_path, 'nttu_wifi_data.csv')
    if not os.path.exists(main_csv_path):
        print(f"❌ 錯誤: 找不到主要資料檔 {main_csv_path}")
        print("   請確認已執行 run_preprocessing.py 的前幾個步驟以產生此檔案。")
        return

    print(f"\n[步驟 1/4] 正在從 {main_csv_path} 載入資料...")
    try:
        df = pd.read_csv(main_csv_path)
    except Exception as e:
        print(f"❌ 錯誤: 無法讀取 CSV 檔案: {e}")
        return

    # *** 呼叫更新後的 define_groups ***
    df = define_groups(df)

    # 檢查並顯示群組分布
    print("\n資料群組分布情況：")
    group_counts = df['group'].value_counts()
    print(group_counts)

    # 過濾掉 'other' 群組的資料
    df_filtered = df[df['group'] != 'other'].copy()

    if len(df_filtered) == 0:
        print("\n❌ 錯誤：過濾 'other' 群組後沒有剩下任何有效資料！")
        print("   請檢查 define_groups 函數中的規則是否與 CSV 檔案中的內容匹配。")
        return
    else:
        print(f"\n保留用於訓練的總資料筆數: {len(df_filtered)}")

    # --- 2. 準備「模型一：Group Classifier」的資料 ---
    print("\n[步驟 2/4] 正在準備『Group Classifier』(模型 1) 的資料...")

    # 建立群組標籤 (ID 從 0 開始)
    group_names = sorted(df_filtered['group'].unique()) # 自動偵測所有非 'other' 的群組名稱
    print(f"   偵測到 {len(group_names)} 個有效群組: {group_names}")

    group_to_id = {name: i for i, name in enumerate(group_names)}
    id_to_group = {i: name for name, i in group_to_id.items()}

    df_filtered['group_id'] = df_filtered['group'].map(group_to_id)

    # 提取 RSSI 特徵欄位名稱
    rss_columns = [col for col in df.columns if col.startswith('rss_')]
    if not rss_columns:
        print("❌ 錯誤：在 CSV 檔案中找不到任何 'rss_' 開頭的欄位。")
        return

    # 提取 RSSI 數據和群組標籤
    x_all = df_filtered[rss_columns].values.astype(np.float32)
    y_all_group_ids = df_filtered['group_id'].values.astype(np.int32)

    # 標準化 RSSI
    print("   正在將 RSSI 訊號縮放至 [0, 1] 範圍...")
    x_all_scaled = np.clip(x_all, -100.0, 0.0)
    x_all_scaled = (x_all_scaled + 100.0) / 100.0
    print(f"   RSSI 資料維度: {x_all_scaled.shape}")

    # 分割訓練集和測試集
    print("   正在分割訓練集與測試集 (80/20)...")
    indices = np.arange(len(x_all_scaled))
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y_all_group_ids
        )
    except ValueError as e:
         print(f"❌ 錯誤：無法進行分層抽樣。可能是某些群組的樣本數太少 (少於 2 筆)。錯誤訊息：{e}")
         print("   請檢查各群組的樣本數：\n", group_counts)
         return

    # 儲存 NumPy 陣列檔案
    print("   正在儲存分類器的 npy 檔案...")
    np.save(os.path.join(output_dir, 'train_x_classifier.npy'), x_all_scaled[train_idx])
    np.save(os.path.join(output_dir, 'train_y_classifier.npy'), y_all_group_ids[train_idx])
    np.save(os.path.join(output_dir, 'test_x_classifier.npy'), x_all_scaled[test_idx])
    np.save(os.path.join(output_dir, 'test_y_classifier.npy'), y_all_group_ids[test_idx])

    print("   ✅ 分類器資料儲存完畢。")
    print(f"      訓練集大小: {len(train_idx)}, 測試集大小: {len(test_idx)}")
    print(f"      標籤 ID 映射: {group_to_id}")

    # --- 3. 準備 N 個「Coordinate Regressors」的資料 ---
    num_groups = len(group_names)
    print(f"\n[步驟 3/4] 正在準備 {num_groups} 份獨立的『Coordinate Regressors』資料...")

    coord_scaler_config = {}

    for group_name in group_names: # 使用偵測到的 group_names
        print(f"\n   --- 處理群組: {group_name} ---")

        df_group = df_filtered[df_filtered['group'] == group_name]
        print(f"      找到 {len(df_group)} 筆資料")

        if len(df_group) < 2:
            print(f"      ⚠️ 警告：群組 {group_name} 的資料筆數少於 2，無法分割訓練/測試集，將跳過此群組。")
            continue

        x_group = df_group[rss_columns].values.astype(np.float32)
        c_group_original = df_group[['x', 'y']].values.astype(np.float32)

        x_group_scaled = np.clip(x_group, -100.0, 0.0)
        x_group_scaled = (x_group_scaled + 100.0) / 100.0

        print(f"      正在對 {group_name} 的座標進行獨立標準化...")
        scaler = StandardScaler()
        c_group_scaled = scaler.fit_transform(c_group_original)
        print(f"      標準化後座標 Mean: {np.mean(c_group_scaled, axis=0)}, Std: {np.std(c_group_scaled, axis=0)}")

        if scaler.scale_[0] == 0 or scaler.scale_[1] == 0:
             print(f"      ⚠️ 警告：群組 {group_name} 的 X 或 Y 座標標準差為 0。")
             mean_x, std_x = (0.0, 1.0)
             mean_y, std_y = (0.0, 1.0)
        else:
             mean_x = float(scaler.mean_[0])
             std_x = float(scaler.scale_[0])
             mean_y = float(scaler.mean_[1])
             std_y = float(scaler.scale_[1])

        coord_scaler_config[group_name] = {
            'mean_x': mean_x, 'std_x': std_x,
            'mean_y': mean_y, 'std_y': std_y,
        }

        print("      正在分割訓練集與測試集...")
        indices_group = np.arange(len(x_group_scaled))
        train_idx_g, test_idx_g = train_test_split(indices_group, test_size=0.2, random_state=42)

        print("      正在儲存回歸器的 npy 檔案...")
        np.save(os.path.join(output_dir, f'train_x_{group_name}.npy'), x_group_scaled[train_idx_g])
        np.save(os.path.join(output_dir, f'train_c_{group_name}.npy'), c_group_scaled[train_idx_g])
        np.save(os.path.join(output_dir, f'test_x_{group_name}.npy'), x_group_scaled[test_idx_g])
        np.save(os.path.join(output_dir, f'test_c_{group_name}.npy'), c_group_scaled[test_idx_g])
        np.save(os.path.join(output_dir, f'test_c_original_{group_name}.npy'), c_group_original[test_idx_g])

        print(f"      ✅ {group_name} 回歸器資料儲存完畢。")
        print(f"         訓練集大小: {len(train_idx_g)}, 測試集大小: {len(test_idx_g)}")

    # --- 4. 儲存所有 Config 檔案 ---
    print("\n[步驟 4/4] 正在儲存設定檔...")

    coord_config_path = os.path.join(output_dir, 'coord_scaler_config.json')
    try:
        with open(coord_config_path, 'w', encoding='utf-8') as f:
            json.dump(coord_scaler_config, f, ensure_ascii=False, indent=2)
        print(f"   ✅ {len(coord_scaler_config)} 份座標模型的標準化參數已儲存至: {coord_config_path}")
    except Exception as e:
        print(f"   ❌ 錯誤: 無法儲存座標設定檔: {e}")

    classifier_config_path = os.path.join(output_dir, 'classifier_config.json')
    try:
        with open(classifier_config_path, 'w', encoding='utf-8') as f:
            json.dump({"group_mapping": id_to_group}, f, ensure_ascii=False, indent=2)
        print(f"   ✅ 分類器標籤映射已儲存至: {classifier_config_path}")
    except Exception as e:
        print(f"   ❌ 錯誤: 無法儲存分類器設定檔: {e}")

# --- 主執行區塊 (保持不變) ---
if __name__ == "__main__":
    data_path = "../processed_data"
    output_dir = "../hadnn_data_split" # 輸出到同一個新資料夾

    prepare_split_data(data_path, output_dir)

    print("\n--- ✅ 資料預處理 (更新群組規則) 全部完成 ---")
    print(f"所有 1+N 模型的資料都已準備好，儲存在: {output_dir}")
    print("\n下一步建議：")
    print("1. 檢查輸出資料夾中的檔案和 `coord_scaler_config.json` 的群組數量是否符合預期。")
    print("2. 開始訓練『Group Classifier』模型。")
    print("3. 使用迴圈訓練 N 個獨立的『Coordinate Regressor』模型。")