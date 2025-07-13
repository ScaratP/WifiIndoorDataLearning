import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import scale, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os

class NTTUDataset:
    """
    NTTU WiFi 資料集類別，支援點分類和位置回歸
    """
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
        self.load_and_process_data()
        
    def load_and_process_data(self):
        """載入並處理資料"""
        # 讀取主要資料
        df = pd.read_csv(os.path.join(self.data_path, 'nttu_wifi_data.csv'))
        
        # 讀取標籤映射
        with open(os.path.join(self.data_path, 'label_mappings.json'), 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        self.building_mapping = mappings['building_mapping']
        self.floor_mapping = mappings['floor_mapping']
        self.point_mapping = mappings['point_mapping']
        
        # 提取 RSS 向量
        rss_columns = [col for col in df.columns if col.startswith('rss_')]
        self.rss_data = df[rss_columns].values.astype(np.float32)
        
        # 提取座標 (用於位置回歸)
        self.coordinates = df[['x', 'y']].values.astype(np.float32)
        
        # 提取標籤
        self.building_labels = df['building_id'].values.astype(np.float32)
        self.floor_labels = df['floor_id'].values.astype(np.float32)
        self.point_labels = df['point_id'].values.astype(np.float32)  # 新增點分類標籤
        
        # 計算統計資訊
        self.n_rss = len(rss_columns)
        self.n_buildings = len(self.building_mapping)
        self.n_floors = len(self.floor_mapping)
        self.n_points = len(self.point_mapping)  # 新增點數量
        
        # 座標標準化
        self.lo_mean = self.coordinates[:, 0].mean()
        self.lo_std = self.coordinates[:, 0].std()
        self.la_mean = self.coordinates[:, 1].mean()
        self.la_std = self.coordinates[:, 1].std()
        
        # 分割訓練/測試集
        self.split_data()
        
        # RSS 資料標準化
        self.normalize_rss()
        
        print(f"資料載入完成:")
        print(f"  RSS 特徵數: {self.n_rss}")
        print(f"  建築物數: {self.n_buildings}")
        print(f"  樓層數: {self.n_floors}")
        print(f"  點位數: {self.n_points}")
        print(f"  訓練樣本數: {self.num_examples}")
        print(f"  測試樣本數: {self.num_test_examples}")
        
    def split_data(self):
        """分割訓練和測試資料"""
        indices = np.arange(len(self.rss_data))
        
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.building_labels  # 按建築物分層
        )
        
        # 訓練集
        self.train_x = self.rss_data[train_idx]
        self.train_c = self.coordinates[train_idx].copy()
        self.train_b = self.building_labels[train_idx]
        self.train_y = self.floor_labels[train_idx]
        self.train_p = self.point_labels[train_idx]  # 新增點標籤
        
        # 測試集
        self.test_x = self.rss_data[test_idx]
        self.test_c = self.coordinates[test_idx].copy()
        self.test_b = self.building_labels[test_idx]
        self.test_y = self.floor_labels[test_idx]
        self.test_p = self.point_labels[test_idx]  # 新增點標籤
        
        # 標準化座標
        self.train_c[:, 0] = (self.train_c[:, 0] - self.lo_mean) / self.lo_std
        self.train_c[:, 1] = (self.train_c[:, 1] - self.la_mean) / self.la_std
        self.test_c[:, 0] = (self.test_c[:, 0] - self.lo_mean) / self.lo_std
        self.test_c[:, 1] = (self.test_c[:, 1] - self.la_mean) / self.la_std
        
        # 確保所有標籤都是有效的
        self.train_y = np.maximum(self.train_y, 0)
        self.test_y = np.maximum(self.test_y, 0)
        self.train_p = np.maximum(self.train_p, 0)
        self.test_p = np.maximum(self.test_p, 0)
        
        self.num_examples = len(self.train_x)
        self.num_test_examples = len(self.test_x)
        
    def normalize_rss(self):
        """標準化 RSS 資料，使用更穩健的方法"""
        # 檢查數據是否需要預先縮放
        max_abs_value = max(np.max(np.abs(self.train_x)), np.max(np.abs(self.test_x)))
        
        # 如果數值太大，先進行預縮放
        if max_abs_value > 1000:
            print("檢測到較大的 RSS 值，進行預縮放...")
            prescale_factor = 100.0
            self.train_x = self.train_x / prescale_factor
            self.test_x = self.test_x / prescale_factor
        
        # 嘗試使用更穩健的標準化方法
        try:
            # 先嘗試標準標準化
            scaler = StandardScaler(with_mean=True, with_std=True)
            self.train_x = scaler.fit_transform(self.train_x)
            self.test_x = scaler.transform(self.test_x)
        except (ValueError, Warning) as e:
            print(f"標準標準化遇到問題，嘗試 RobustScaler: {e}")
            try:
                # 如果標準標準化失敗，使用更穩健的縮放器
                robust_scaler = RobustScaler(quantile_range=(25.0, 75.0))
                self.train_x = robust_scaler.fit_transform(self.train_x)
                self.test_x = robust_scaler.transform(self.test_x)
                print("成功使用 RobustScaler 進行標準化")
            except Exception as e2:
                print(f"縮放遇到嚴重問題: {e2}")
                # 最後手段：使用簡單的最小-最大縮放
                print("使用簡單縮放，這可能影響模型性能")
                self.train_x = (self.train_x - np.min(self.train_x)) / (np.max(self.train_x) - np.min(self.train_x) + 1e-10)
                self.test_x = (self.test_x - np.min(self.test_x)) / (np.max(self.test_x) - np.min(self.test_x) + 1e-10)

def prepare_for_hadnn(data_path, output_dir):
    """
    為 HADNN 模型準備資料
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立資料集物件
    dataset = NTTUDataset(data_path)
    
    # 儲存處理後的資料
    np.save(os.path.join(output_dir, 'train_x.npy'), dataset.train_x)
    np.save(os.path.join(output_dir, 'train_c.npy'), dataset.train_c)
    np.save(os.path.join(output_dir, 'train_b.npy'), dataset.train_b)
    np.save(os.path.join(output_dir, 'train_y.npy'), dataset.train_y)
    np.save(os.path.join(output_dir, 'train_p.npy'), dataset.train_p)  # 新增點標籤
    
    np.save(os.path.join(output_dir, 'test_x.npy'), dataset.test_x)
    np.save(os.path.join(output_dir, 'test_c.npy'), dataset.test_c)
    np.save(os.path.join(output_dir, 'test_b.npy'), dataset.test_b)
    np.save(os.path.join(output_dir, 'test_y.npy'), dataset.test_y)
    np.save(os.path.join(output_dir, 'test_p.npy'), dataset.test_p)  # 新增點標籤
    
    # 儲存資料集配置
    config = {
        'n_rss': dataset.n_rss,
        'n_buildings': dataset.n_buildings,
        'n_floors': dataset.n_floors,
        'n_points': dataset.n_points,  # 新增點數量
        'num_examples': dataset.num_examples,
        'num_test_examples': dataset.num_test_examples,
        'lo_mean': float(dataset.lo_mean),
        'lo_std': float(dataset.lo_std),
        'la_mean': float(dataset.la_mean),
        'la_std': float(dataset.la_std),
        'building_mapping': dataset.building_mapping,
        'floor_mapping': dataset.floor_mapping,
        'point_mapping': dataset.point_mapping  # 新增點映射
    }
    
    with open(os.path.join(output_dir, 'dataset_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"HADNN 格式資料已儲存到: {output_dir}")
    
    return dataset

if __name__ == "__main__":
    data_path = "../processed_data"
    output_dir = "../hadnn_data"
    
    dataset = prepare_for_hadnn(data_path, output_dir)
