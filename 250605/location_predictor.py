import tensorflow as tf
import numpy as np
import os
import json
from collections import defaultdict
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class LocationPredictor:
    """為每個建築物和樓層組合訓練獨立的位置預測模型"""
    
    def __init__(self, hadnn_data_dir, output_dir):
        """
        初始化位置預測器
        
        參數:
            hadnn_data_dir (str): HADNN格式數據目錄
            output_dir (str): 輸出目錄
        """
        self.hadnn_data_dir = hadnn_data_dir
        self.output_dir = output_dir
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 載入配置
        self.load_config()
        
        # 載入數據
        self.load_data()
    
    def load_config(self):
        """載入HADNN數據配置"""
        config_path = os.path.join(self.hadnn_data_dir, 'dataset_config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("配置載入成功")
            
            # 提取重要參數
            self.n_rss = self.config.get('n_rss')
            self.n_buildings = self.config.get('n_buildings')
            self.n_floors = self.config.get('n_floors')
            self.lo_mean = self.config.get('lo_mean')
            self.lo_std = self.config.get('lo_std')
            self.la_mean = self.config.get('la_mean')
            self.la_std = self.config.get('la_std')
            
            print(f"RSS特徵數: {self.n_rss}")
            print(f"建築物數: {self.n_buildings}")
            print(f"樓層數: {self.n_floors}")
            
            return True
            
        except Exception as e:
            print(f"載入配置時發生錯誤: {e}")
            return False
    
    def load_data(self):
        """載入HADNN格式數據"""
        try:
            # 載入訓練資料
            self.train_x = np.load(os.path.join(self.hadnn_data_dir, 'train_x.npy'))
            self.train_y = np.load(os.path.join(self.hadnn_data_dir, 'train_y.npy'))
            self.train_b = np.load(os.path.join(self.hadnn_data_dir, 'train_b.npy')).squeeze()
            self.train_c = np.load(os.path.join(self.hadnn_data_dir, 'train_c.npy')).squeeze()
            
            # 載入測試資料
            self.test_x = np.load(os.path.join(self.hadnn_data_dir, 'test_x.npy'))
            self.test_y = np.load(os.path.join(self.hadnn_data_dir, 'test_y.npy'))
            self.test_b = np.load(os.path.join(self.hadnn_data_dir, 'test_b.npy')).squeeze()
            self.test_c = np.load(os.path.join(self.hadnn_data_dir, 'test_c.npy')).squeeze()
            
            print(f"訓練樣本數: {len(self.train_x)}")
            print(f"測試樣本數: {len(self.test_x)}")
            
            # 分離不同建築物樓層的數據
            self.building_floor_data = self._split_by_building_floor()
            
            return True
            
        except Exception as e:
            print(f"載入數據時發生錯誤: {e}")
            return False
    
    def _split_by_building_floor(self):
        """將數據按建築物和樓層分組"""
        building_floor_data = {}
        
        # 處理訓練資料
        for i in range(len(self.train_x)):
            building = int(self.train_b[i])
            floor = int(self.train_c[i])
            key = (building, floor)
            
            if key not in building_floor_data:
                building_floor_data[key] = {
                    'train_x': [], 'train_y': [],
                    'test_x': [], 'test_y': []
                }
            
            building_floor_data[key]['train_x'].append(self.train_x[i])
            building_floor_data[key]['train_y'].append(self.train_y[i])
        
        # 處理測試資料
        for i in range(len(self.test_x)):
            building = int(self.test_b[i])
            floor = int(self.test_c[i])
            key = (building, floor)
            
            if key not in building_floor_data:
                building_floor_data[key] = {
                    'train_x': [], 'train_y': [],
                    'test_x': [], 'test_y': []
                }
            
            building_floor_data[key]['test_x'].append(self.test_x[i])
            building_floor_data[key]['test_y'].append(self.test_y[i])
        
        # 轉換為NumPy數組
        for key in building_floor_data:
            building_floor_data[key]['train_x'] = np.array(building_floor_data[key]['train_x'])
            building_floor_data[key]['train_y'] = np.array(building_floor_data[key]['train_y'])
            building_floor_data[key]['test_x'] = np.array(building_floor_data[key]['test_x'])
            building_floor_data[key]['test_y'] = np.array(building_floor_data[key]['test_y'])
        
        # 顯示每個組合的樣本數
        print("\n建築物樓層組合樣本數:")
        for key, data in building_floor_data.items():
            building, floor = key
            print(f"建築物 {building}, 樓層 {floor}: 訓練 {len(data['train_x'])}, 測試 {len(data['test_x'])}")
        
        return building_floor_data
    
    def create_location_model(self, n_rss):
        """建立位置預測模型架構"""
        model = Sequential([
            Dense(64, input_shape=(n_rss,)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            Dense(32),
            BatchNormalization(),
            Activation('relu'),
            
            Dense(16),
            BatchNormalization(),
            Activation('relu'),
            
            Dense(2)  # 輸出x, y座標
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_location_models(self, batch_size=32, epochs=100, patience=20):
        """為每個建築物和樓層組合訓練位置預測模型"""
        models = {}
        results = {}
        
        for key, data in self.building_floor_data.items():
            building, floor = key
            model_name = f"building_{building}_floor_{floor}"
            
            print(f"\n訓練 {model_name} 模型...")
            
            # 跳過資料太少的組合
            if len(data['train_x']) < 10 or len(data['test_x']) < 5:
                print(f"跳過 {model_name}，資料太少")
                continue
            
            # 建立模型
            model = self.create_location_model(self.n_rss)
            
            # 回調函數
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
            ]
            
            # 訓練模型
            history = model.fit(
                data['train_x'], data['train_y'],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(data['test_x'], data['test_y']),
                callbacks=callbacks,
                verbose=1
            )
            
            # 評估模型
            evaluation = model.evaluate(data['test_x'], data['test_y'], verbose=0)
            mae = evaluation[1]  # 平均絕對誤差
            
            # 轉換回實際座標下的誤差（單位：米）
            mae_meters_x = mae * self.lo_std
            mae_meters_y = mae * self.la_std
            mae_meters_total = np.sqrt(mae_meters_x**2 + mae_meters_y**2)
            
            results[model_name] = {
                'mse': float(evaluation[0]),
                'mae': float(evaluation[1]),
                'mae_meters_total': float(mae_meters_total),
                'train_samples': int(len(data['train_x'])),
                'test_samples': int(len(data['test_x']))
            }
            
            print(f"{model_name} 測試結果:")
            print(f"  - MSE: {evaluation[0]:.4f}")
            print(f"  - MAE: {evaluation[1]:.4f}")
            print(f"  - 實際距離誤差: {mae_meters_total:.4f} 米")
            
            # 儲存模型
            model_path = os.path.join(self.output_dir, model_name)
            model.save(model_path)
            print(f"{model_name} 模型已儲存到 {model_path}")
            
            models[key] = model
        
        # 儲存結果
        results_path = os.path.join(self.output_dir, 'location_models_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n模型評估結果已儲存到 {results_path}")
        
        # 儲存模型配置
        config_path = os.path.join(self.output_dir, 'location_models_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'lo_mean': self.lo_mean,
                'lo_std': self.lo_std,
                'la_mean': self.la_mean,
                'la_std': self.la_std,
                'buildings': self.n_buildings,
                'floors': self.n_floors,
                'model_count': len(models)
            }, f, indent=4)
        
        return models, results
    
    def convert_to_tflite(self):
        """將所有位置預測模型轉換為TensorFlow Lite格式"""
        print("\n將模型轉換為TensorFlow Lite格式...")
        
        tflite_dir = os.path.join(self.output_dir, 'tflite')
        os.makedirs(tflite_dir, exist_ok=True)
        
        # 尋找所有模型文件夾
        model_dirs = [d for d in os.listdir(self.output_dir) 
                      if os.path.isdir(os.path.join(self.output_dir, d)) and d.startswith('building_')]
        
        for model_dir in model_dirs:
            model_path = os.path.join(self.output_dir, model_dir)
            tflite_path = os.path.join(tflite_dir, f"{model_dir}.tflite")
            
            try:
                # 載入模型
                model = tf.keras.models.load_model(model_path)
                
                # 建立轉換器
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                
                # 設定優化選項
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
                # 轉換模型
                tflite_model = converter.convert()
                
                # 儲存TensorFlow Lite模型
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                # 計算模型大小
                model_size = len(tflite_model) / (1024 * 1024)  # MB
                print(f"✓ {model_dir}: {model_size:.2f} MB")
                
            except Exception as e:
                print(f"✗ 轉換 {model_dir} 時發生錯誤: {e}")
        
        print(f"TensorFlow Lite模型已儲存到 {tflite_dir}")

if __name__ == "__main__":
    hadnn_data_dir = "./processed_data/hadnn"
    output_dir = "./location_models"
    
    # 建立位置預測器
    predictor = LocationPredictor(hadnn_data_dir, output_dir)
    
    # 訓練模型
    predictor.train_location_models(batch_size=32, epochs=100, patience=20)
    
    # 轉換為TensorFlow Lite格式
    predictor.convert_to_tflite()
