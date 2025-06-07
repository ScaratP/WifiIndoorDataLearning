import tensorflow as tf
import numpy as np
import os
import pickle

class HADNNModelSplitter:
    def __init__(self, original_model_path):
        self.original_model_path = original_model_path
        self.original_model = None
        self.building_classifier = None
        self.floor_classifier = None
        
    def load_original_model(self):
        """載入原始 HADNN 模型"""
        if os.path.exists(self.original_model_path):
            self.original_model = tf.keras.models.load_model(self.original_model_path)
            print(f"成功載入原始模型: {self.original_model_path}")
            return True
        else:
            print(f"找不到原始模型: {self.original_model_path}")
            return False
    
    def extract_building_classifier(self, input_shape, n_buildings):
        """從原始模型中提取建築物分類器"""
        if self.original_model is None:
            print("請先載入原始模型")
            return None
        
        # 建立新的建築物分類器
        inputs = tf.keras.Input(shape=input_shape, name='wifi_input')
        
        # 複製原始模型的共享層
        x = inputs
        for i, layer in enumerate(self.original_model.layers[1:]):  # 跳過輸入層
            if 'building' in layer.name or i < 3:  # 假設前幾層是共享特徵提取層
                try:
                    if hasattr(layer, 'get_weights') and layer.get_weights():
                        new_layer = tf.keras.layers.Dense(
                            layer.units if hasattr(layer, 'units') else layer.output_shape[-1],
                            activation=layer.activation,
                            name=f"building_{layer.name}"
                        )
                        x = new_layer(x)
                        new_layer.set_weights(layer.get_weights())
                    else:
                        x = layer(x)
                except:
                    # 如果層無法直接複製，建立相似的層
                    if hasattr(layer, 'units'):
                        x = tf.keras.layers.Dense(layer.units, activation='relu')(x)
                    break
        
        # 添加建築物分類輸出層
        building_output = tf.keras.layers.Dense(n_buildings, activation='softmax', name='building_output')(x)
        
        self.building_classifier = tf.keras.Model(inputs=inputs, outputs=building_output, name='building_classifier')
        
        # 編譯模型
        self.building_classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.building_classifier
    
    def extract_floor_classifier(self, input_shape, n_building_floor_combinations):
        """從原始模型中提取樓層分類器"""
        if self.original_model is None:
            print("請先載入原始模型")
            return None
        
        # 建立新的樓層分類器（包含建築物資訊）
        wifi_input = tf.keras.Input(shape=input_shape, name='wifi_input')
        building_input = tf.keras.Input(shape=(1,), name='building_input')
        
        # WiFi 特徵提取
        x = wifi_input
        for i, layer in enumerate(self.original_model.layers[1:]):
            if 'floor' in layer.name or i < 3:
                try:
                    if hasattr(layer, 'get_weights') and layer.get_weights():
                        new_layer = tf.keras.layers.Dense(
                            layer.units if hasattr(layer, 'units') else layer.output_shape[-1],
                            activation=layer.activation,
                            name=f"floor_{layer.name}"
                        )
                        x = new_layer(x)
                        new_layer.set_weights(layer.get_weights())
                    else:
                        x = layer(x)
                except:
                    if hasattr(layer, 'units'):
                        x = tf.keras.layers.Dense(layer.units, activation='relu')(x)
                    break
        
        # 將建築物資訊編碼並融合
        building_embedded = tf.keras.layers.Embedding(10, 16, name='building_embedding')(building_input)
        building_embedded = tf.keras.layers.Flatten()(building_embedded)
        
        # 融合 WiFi 特徵和建築物資訊
        combined = tf.keras.layers.concatenate([x, building_embedded])
        combined = tf.keras.layers.Dense(128, activation='relu', name='combined_features')(combined)
        
        # 樓層分類輸出
        floor_output = tf.keras.layers.Dense(n_building_floor_combinations, activation='softmax', name='floor_output')(combined)
        
        self.floor_classifier = tf.keras.Model(
            inputs=[wifi_input, building_input], 
            outputs=floor_output, 
            name='floor_classifier'
        )
        
        # 編譯模型
        self.floor_classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.floor_classifier
    
    def create_position_predictor_template(self, input_shape):
        """建立位置預測器模板（用於每個建築樓層組合）"""
        inputs = tf.keras.Input(shape=input_shape, name='wifi_input')
        
        # 特徵提取層
        x = tf.keras.layers.Dense(512, activation='relu', name='feature_1')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu', name='feature_2')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='feature_3')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 位置預測輸出 (x, y 座標)
        position_output = tf.keras.layers.Dense(2, activation='linear', name='position_output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=position_output, name='position_predictor')
        
        # 編譯模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def save_split_models(self, save_dir):
        """儲存拆分後的模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.building_classifier:
            building_path = os.path.join(save_dir, 'building_classifier')
            self.building_classifier.save(building_path)
            print(f"建築物分類器已儲存至: {building_path}")
        
        if self.floor_classifier:
            floor_path = os.path.join(save_dir, 'floor_classifier')
            self.floor_classifier.save(floor_path)
            print(f"樓層分類器已儲存至: {floor_path}")
    
    def create_all_models(self, input_shape, n_buildings, n_building_floor_combinations):
        """建立所有拆分後的模型"""
        print("正在建立建築物分類器...")
        building_model = self.extract_building_classifier(input_shape, n_buildings)
        
        print("正在建立樓層分類器...")
        floor_model = self.extract_floor_classifier(input_shape, n_building_floor_combinations)
        
        print("正在建立位置預測器模板...")
        position_template = self.create_position_predictor_template(input_shape)
        
        return building_model, floor_model, position_template

if __name__ == "__main__":
    # 測試模型拆分器
    # 注意：這裡需要實際的 HADNN 模型路徑
    original_model_path = "../../../hadnn/data/pretrained/HADNN2"
    
    splitter = HADNNModelSplitter(original_model_path)
    
    # 假設的參數
    input_shape = (237,)  # 237 個 BSSID 特徵
    n_buildings = 2  # SE1, SE2
    n_building_floor_combinations = 2  # SE1_F1, SE2_F2
    
    if splitter.load_original_model():
        building_model, floor_model, position_template = splitter.create_all_models(
            input_shape, n_buildings, n_building_floor_combinations
        )
        
        print("模型拆分完成！")
        print(f"建築物分類器: {building_model.summary()}")
        print(f"樓層分類器: {floor_model.summary()}")
        print(f"位置預測器模板: {position_template.summary()}")
    else:
        print("由於無法載入原始模型，將建立新的模型架構")
        building_model, floor_model, position_template = splitter.create_all_models(
            input_shape, n_buildings, n_building_floor_combinations
        )
