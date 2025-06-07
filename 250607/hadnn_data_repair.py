"""
HADNN數據結構修復工具
用於修復訓練數據和測試數據中的標籤混亂、維度不匹配等問題
"""
import os
import numpy as np
import json

def repair_hadnn_data_structure(data_dir):
    """
    修復HADNN數據結構中常見的問題
    
    參數:
        data_dir (str): 包含HADNN數據文件的目錄
    
    返回:
        bool: 是否成功修復
    """
    print(f"檢查並修復 {data_dir} 中的HADNN數據結構...")
    
    # 檢查必要文件是否存在
    required_files = [
        'train_x.npy', 'train_y.npy', 'train_b.npy', 'train_c.npy',
        'test_x.npy', 'test_y.npy', 'test_b.npy', 'test_c.npy',
        'dataset_config.json'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"缺少必要文件: {file}")
            return False
    
    try:
        # 載入數據
        train_x = np.load(os.path.join(data_dir, 'train_x.npy'))
        train_y = np.load(os.path.join(data_dir, 'train_y.npy'))
        train_b = np.load(os.path.join(data_dir, 'train_b.npy'))
        train_c = np.load(os.path.join(data_dir, 'train_c.npy'))
        
        test_x = np.load(os.path.join(data_dir, 'test_x.npy'))
        test_y = np.load(os.path.join(data_dir, 'test_y.npy'))
        test_b = np.load(os.path.join(data_dir, 'test_b.npy'))
        test_c = np.load(os.path.join(data_dir, 'test_c.npy'))
        
        # 載入配置
        with open(os.path.join(data_dir, 'dataset_config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("原始數據形狀:")
        print(f"  train_x: {train_x.shape}")
        print(f"  train_y: {train_y.shape}")
        print(f"  train_b: {train_b.shape}")
        print(f"  train_c: {train_c.shape}")
        print(f"  test_x: {test_x.shape}")
        print(f"  test_y: {test_y.shape}")
        print(f"  test_b: {test_b.shape}")
        print(f"  test_c: {test_c.shape}")
        
        # 修復常見問題
        
        # 1. 確保建築物標籤是一維的
        if len(train_b.shape) > 1:
            train_b = train_b.reshape(-1)
            print(f"修復 train_b 形狀為: {train_b.shape}")
        if len(test_b.shape) > 1:
            test_b = test_b.reshape(-1)
            print(f"修復 test_b 形狀為: {test_b.shape}")
            
        # 2. 檢查train_c和test_c，確定它們是位置座標還是樓層標籤
        coords_like_train_c = len(train_c.shape) > 1 and train_c.shape[1] == 2
        coords_like_test_c = len(test_c.shape) > 1 and test_c.shape[1] == 2
        
        # 3. 確保位置座標是二維的 [樣本數, 2]
        if len(train_y.shape) == 1:
            if train_y.shape[0] % 2 == 0 and train_y.shape[0] // 2 == train_x.shape[0]:
                # 可能是 x 和 y 座標連續存放
                half_len = train_y.shape[0] // 2
                train_y = np.column_stack((train_y[:half_len], train_y[half_len:]))
                print(f"修復 train_y 形狀為: {train_y.shape}")
            else:
                # 可能是只有單一維度的座標
                train_y = np.column_stack((train_y, np.zeros_like(train_y)))
                print(f"擴展 train_y 為二維座標: {train_y.shape}")
        
        if len(test_y.shape) == 1:
            if test_y.shape[0] % 2 == 0 and test_y.shape[0] // 2 == test_x.shape[0]:
                # 可能是 x 和 y 座標連續存放
                half_len = test_y.shape[0] // 2
                test_y = np.column_stack((test_y[:half_len], test_y[half_len:]))
                print(f"修復 test_y 形狀為: {test_y.shape}")
            else:
                # 可能是只有單一維度的座標
                test_y = np.column_stack((test_y, np.zeros_like(test_y)))
                print(f"擴展 test_y 為二維座標: {test_y.shape}")
        
        # 4. 如果train_c/test_c確定是位置座標，則需要創建樓層標籤
        if coords_like_train_c:
            print("train_c 似乎是位置座標，而非樓層標籤")
            # 嘗試創建樓層標籤，預設為全0
            train_f = np.zeros(train_x.shape[0], dtype=int)
            
            # 如果配置中有樓層映射，嘗試使用它來創建更好的標籤
            if 'floor_mapping' in config and config['floor_mapping']:
                # 假設所有樓層都在一樓 (最常見的樓層)
                default_floor_id = 0
                for floor_name, floor_id in config['floor_mapping'].items():
                    if '1' in floor_name:  # 找到表示一樓的標籤
                        default_floor_id = floor_id
                        break
                train_f.fill(default_floor_id)
            
            print(f"創建樓層標籤 train_f: {train_f.shape}")
            np.save(os.path.join(data_dir, 'train_f.npy'), train_f)
        
        if coords_like_test_c:
            print("test_c 似乎是位置座標，而非樓層標籤")
            # 嘗試創建樓層標籤，預設為全0
            test_f = np.zeros(test_x.shape[0], dtype=int)
            
            # 如果配置中有樓層映射，嘗試使用它來創建更好的標籤
            if 'floor_mapping' in config and config['floor_mapping']:
                # 假設所有樓層都在一樓 (最常見的樓層)
                default_floor_id = 0
                for floor_name, floor_id in config['floor_mapping'].items():
                    if '1' in floor_name:  # 找到表示一樓的標籤
                        default_floor_id = floor_id
                        break
                test_f.fill(default_floor_id)
            
            print(f"創建樓層標籤 test_f: {test_f.shape}")
            np.save(os.path.join(data_dir, 'test_f.npy'), test_f)
        
        # 5. 保存修復後的數據
        np.save(os.path.join(data_dir, 'train_x.npy'), train_x)
        np.save(os.path.join(data_dir, 'train_y.npy'), train_y)
        np.save(os.path.join(data_dir, 'train_b.npy'), train_b)
        
        np.save(os.path.join(data_dir, 'test_x.npy'), test_x)
        np.save(os.path.join(data_dir, 'test_y.npy'), test_y)
        np.save(os.path.join(data_dir, 'test_b.npy'), test_b)
        
        # 如果沒有樓層標籤文件，則將train_c/test_c也保存為樓層標籤
        if not coords_like_train_c:
            train_f = train_c.reshape(-1)
            np.save(os.path.join(data_dir, 'train_f.npy'), train_f)
        
        if not coords_like_test_c:
            test_f = test_c.reshape(-1)
            np.save(os.path.join(data_dir, 'test_f.npy'), test_f)
        
        print("HADNN數據結構修復完成")
        return True
        
    except Exception as e:
        print(f"修復HADNN數據時發生錯誤: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    data_dir = "./hadnn_data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    repair_hadnn_data_structure(data_dir)
