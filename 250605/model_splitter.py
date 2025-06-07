import tensorflow as tf
import os
import numpy as np
import json
import sys
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Add, Concatenate, Dropout
from tensorflow.keras.models import Model, load_model
import shutil

class HADNNModelSplitter:
    """從HADNN模型拆分出建築物和樓層分類模型"""
    
    def __init__(self, hadnn_model_path, output_dir, hadnn_version="HADNN2"):
        """
        初始化HADNN模型拆分器
        
        參數:
            hadnn_model_path (str): HADNN模型路徑
            output_dir (str): 輸出目錄
            hadnn_version (str): HADNN版本，例如 "HADNN2" 或 "HADNNh2"
        """
        self.hadnn_model_path = hadnn_model_path
        self.output_dir = output_dir
        self.hadnn_version = hadnn_version
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
    def load_hadnn_model(self):
        """載入HADNN模型"""
        print(f"載入HADNN模型 {self.hadnn_model_path}...")
        try:
            self.model = load_model(self.hadnn_model_path)
            print("模型載入成功")
            return True
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            return False
    
    def extract_building_model(self):
        """提取建築物分類模型"""
        print("提取建築物分類模型...")
        
        # 檢查是否為HADNN2或HADNNh2模型
        if "HADNN" not in self.hadnn_version or self.hadnn_version[-1] != "2":
            print("錯誤: 只能從HADNN2或HADNNh2模型提取建築物分類模型")
            return None
        
        try:
            # 獲取輸入層
            input_layer = self.model.input
            
            # 找出建築物分類輸出層
            building_output = None
            for i, output in enumerate(self.model.outputs):
                # HADNN模型中，第二個輸出通常是建築物分類
                if i == 1:
                    building_output = output
                    break
            
            if building_output is None:
                print("錯誤: 無法找到建築物分類輸出層")
                return None
            
            # 建立建築物分類模型
            building_model = Model(inputs=input_layer, outputs=building_output)
            
            # 儲存模型
            building_model_path = os.path.join(self.output_dir, f"{self.hadnn_version}_building_classifier")
            building_model.save(building_model_path)
            print(f"建築物分類模型已儲存到 {building_model_path}")
            
            return building_model
        
        except Exception as e:
            print(f"提取建築物模型時發生錯誤: {e}")
            return None
    
    def extract_floor_model(self):
        """提取樓層分類模型"""
        print("提取樓層分類模型...")
        
        try:
            # 獲取輸入層
            input_layer = self.model.input
            
            # 找出樓層分類輸出層
            floor_output = None
            
            # 對於HADNN2/HADNNh2，樓層分類是第三個輸出
            # 對於HADNN1/HADNNh1，樓層分類是第二個輸出
            output_index = 2 if self.hadnn_version[-1] == "2" else 1
            
            for i, output in enumerate(self.model.outputs):
                if i == output_index:
                    floor_output = output
                    break
            
            if floor_output is None:
                print("錯誤: 無法找到樓層分類輸出層")
                return None
            
            # 建立樓層分類模型
            floor_model = Model(inputs=input_layer, outputs=floor_output)
            
            # 儲存模型
            floor_model_path = os.path.join(self.output_dir, f"{self.hadnn_version}_floor_classifier")
            floor_model.save(floor_model_path)
            print(f"樓層分類模型已儲存到 {floor_model_path}")
            
            return floor_model
        
        except Exception as e:
            print(f"提取樓層模型時發生錯誤: {e}")
            return None
    
    def extract_location_model(self):
        """提取位置預測模型"""
        print("提取位置預測模型...")
        
        try:
            # 獲取輸入層
            input_layer = self.model.input
            
            # 位置預測是第一個輸出
            location_output = self.model.outputs[0]
            
            # 建立位置預測模型
            location_model = Model(inputs=input_layer, outputs=location_output)
            
            # 儲存模型
            location_model_path = os.path.join(self.output_dir, f"{self.hadnn_version}_location_predictor")
            location_model.save(location_model_path)
            print(f"位置預測模型已儲存到 {location_model_path}")
            
            return location_model
            
        except Exception as e:
            print(f"提取位置預測模型時發生錯誤: {e}")
            return None
    
    def create_hierarchical_model(self, config_path):
        """建立分級模型系統"""
        print("建立分級模型系統...")
        
        try:
            # 載入配置檔案
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            n_rss = config.get('n_rss')
            n_buildings = config.get('n_buildings')
            n_floors = config.get('n_floors')
            
            # 輸入層
            inputs = Input(shape=(n_rss,))
            
            # 基礎特徵提取層
            x = Dense(128)(inputs)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.3)(x)
            
            # 建築物分類分支
            building_x = Dense(64)(x)
            building_x = BatchNormalization()(building_x)
            building_x = Activation('relu')(building_x)
            building_x = Dense(n_buildings)(building_x)
            building_output = Activation('softmax', name='building_output')(building_x)
            
            # 樓層分類分支
            floor_x = Dense(64)(x)
            floor_x = BatchNormalization()(floor_x)
            floor_x = Activation('relu')(floor_x)
            floor_x = Dense(n_floors)(floor_x)
            floor_output = Activation('softmax', name='floor_output')(floor_x)
            
            # 位置預測分支
            location_x = Dense(64)(x)
            location_x = BatchNormalization()(location_x)
            location_x = Activation('relu')(location_x)
            location_x = Dense(32)(location_x)
            location_x = BatchNormalization()(location_x)
            location_x = Activation('relu')(location_x)
            location_output = Dense(2, name='location_output')(location_x)
            
            # 建立模型
            hierarchical_model = Model(inputs=inputs, outputs=[location_output, building_output, floor_output])
            
            # 儲存模型
            hierarchical_model_path = os.path.join(self.output_dir, "hierarchical_model")
            hierarchical_model.save(hierarchical_model_path)
            print(f"分級模型已儲存到 {hierarchical_model_path}")
            
            return hierarchical_model
            
        except Exception as e:
            print(f"建立分級模型時發生錯誤: {e}")
            return None
    
    def split_all(self, config_path=None):
        """執行所有拆分操作"""
        if not self.load_hadnn_model():
            return False
        
        building_model = None
        if self.hadnn_version[-1] == "2":
            building_model = self.extract_building_model()
        
        floor_model = self.extract_floor_model()
        location_model = self.extract_location_model()
        
        # 如果提供了配置路徑，則建立分級模型
        if config_path and os.path.exists(config_path):
            hierarchical_model = self.create_hierarchical_model(config_path)
        
        results = {
            'building_model': building_model is not None,
            'floor_model': floor_model is not None,
            'location_model': location_model is not None
        }
        
        print("模型拆分完成")
        return results

def create_building_floor_models_from_scratch(output_dir, config_path):
    """從零開始建立建築物和樓層分類模型"""
    print("從零開始建立建築物和樓層分類模型...")
    
    try:
        # 載入配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        n_rss = config.get('n_rss')
        n_buildings = config.get('n_buildings')
        n_floors = config.get('n_floors')
        
        # 建築物分類模型
        if n_buildings > 1:
            building_inputs = Input(shape=(n_rss,))
            x = Dense(128)(building_inputs)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(64)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(n_buildings)(x)
            building_outputs = Activation('softmax')(x)
            
            building_model = Model(inputs=building_inputs, outputs=building_outputs)
            building_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            building_model_path = os.path.join(output_dir, "building_classifier")
            building_model.save(building_model_path)
            print(f"建築物分類模型已儲存到 {building_model_path}")
        
        # 樓層分類模型
        floor_inputs = Input(shape=(n_rss,))
        x = Dense(128)(floor_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(n_floors)(x)
        floor_outputs = Activation('softmax')(x)
        
        floor_model = Model(inputs=floor_inputs, outputs=floor_outputs)
        floor_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        floor_model_path = os.path.join(output_dir, "floor_classifier")
        floor_model.save(floor_model_path)
        print(f"樓層分類模型已儲存到 {floor_model_path}")
        
        return True
    
    except Exception as e:
        print(f"從零建立模型時發生錯誤: {e}")
        return False

if __name__ == "__main__":
    # 範例使用方式
    hadnn_model_path = "../trained_models/HADNN2"
    output_dir = "./split_models"
    config_path = "./processed_data/hadnn/dataset_config.json"
    
    # 拆分現有HADNN模型
    splitter = HADNNModelSplitter(hadnn_model_path, output_dir, "HADNN2")
    results = splitter.split_all(config_path)
    
    # 從零建立模型
    create_building_floor_models_from_scratch(output_dir, config_path)
