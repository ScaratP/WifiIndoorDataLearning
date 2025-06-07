import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import json
import pickle
from data_processor import DataProcessor
from model_splitter import HADNNModelSplitter

class TrainingPipeline:
    def __init__(self, data_path, save_dir):
        self.data_path = data_path
        self.save_dir = save_dir
        self.processor = DataProcessor(data_path)
        self.models = {}
        self.results = {}
        
    def train_building_classifier(self, X, building_labels, test_size=0.2):
        """訓練建築物分類器"""
        print("開始訓練建築物分類器...")
        
        # 分割訓練測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, building_labels, test_size=test_size, random_state=42, stratify=building_labels
        )
        
        # 正規化特徵
        X_train_norm, X_test_norm = self.processor.normalize_features(X_train, X_test)
        
        # 建立模型
        n_buildings = len(np.unique(building_labels))
        input_shape = (X.shape[1],)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_buildings, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 訓練模型
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_test_norm, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 評估模型
        y_pred = model.predict(X_test_norm)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"建築物分類器準確度: {accuracy:.4f}")
        
        # 儲存模型和結果
        model_path = os.path.join(self.save_dir, 'models', 'building_classifier')
        model.save(model_path)
        
        self.models['building_classifier'] = model
        self.results['building_classifier'] = {
            'accuracy': accuracy,
            'history': history.history
        }
        
        return model, accuracy
    
    def train_floor_classifier(self, X, building_labels, floor_labels, test_size=0.2):
        """訓練樓層分類器"""
        print("開始訓練樓層分類器...")
        
        # 分割訓練測試集
        X_train, X_test, building_train, building_test, floor_train, floor_test = train_test_split(
            X, building_labels, floor_labels, test_size=test_size, random_state=42
        )
        
        # 正規化特徵
        X_train_norm, X_test_norm = self.processor.normalize_features(X_train, X_test)
        
        # 建立模型
        n_floor_combinations = len(np.unique(floor_labels))
        input_shape = (X.shape[1],)
        
        wifi_input = tf.keras.Input(shape=input_shape, name='wifi_input')
        building_input = tf.keras.Input(shape=(1,), name='building_input')
        
        # WiFi 特徵處理
        wifi_features = tf.keras.layers.Dense(512, activation='relu')(wifi_input)
        wifi_features = tf.keras.layers.Dropout(0.3)(wifi_features)
        wifi_features = tf.keras.layers.Dense(256, activation='relu')(wifi_features)
        wifi_features = tf.keras.layers.Dropout(0.3)(wifi_features)
        
        # 建築物資訊嵌入
        building_embedded = tf.keras.layers.Embedding(10, 16)(building_input)
        building_embedded = tf.keras.layers.Flatten()(building_embedded)
        
        # 融合特徵
        combined = tf.keras.layers.concatenate([wifi_features, building_embedded])
        combined = tf.keras.layers.Dense(128, activation='relu')(combined)
        combined = tf.keras.layers.Dropout(0.2)(combined)
        
        # 輸出層
        output = tf.keras.layers.Dense(n_floor_combinations, activation='softmax')(combined)
        
        model = tf.keras.Model(inputs=[wifi_input, building_input], outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 訓練模型
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            [X_train_norm, building_train], floor_train,
            validation_data=([X_test_norm, building_test], floor_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 評估模型
        y_pred = model.predict([X_test_norm, building_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(floor_test, y_pred_classes)
        
        print(f"樓層分類器準確度: {accuracy:.4f}")
        
        # 儲存模型和結果
        model_path = os.path.join(self.save_dir, 'models', 'floor_classifier')
        model.save(model_path)
        
        self.models['floor_classifier'] = model
        self.results['floor_classifier'] = {
            'accuracy': accuracy,
            'history': history.history
        }
        
        return model, accuracy
    
    def train_position_predictors(self, data_splits, test_size=0.2):
        """為每個建築樓層組合訓練位置預測器"""
        print("開始訓練位置預測器...")
        
        position_models = {}
        position_results = {}
        
        for location_key, data in data_splits.items():
            print(f"正在訓練 {location_key} 的位置預測器...")
            
            X = data['X']
            positions = data['positions']
            
            if len(X) < 10:  # 數據點太少，跳過
                print(f"  {location_key} 數據點不足 ({len(X)} 個)，跳過訓練")
                continue
            
            # 分割訓練測試集
            X_train, X_test, pos_train, pos_test = train_test_split(
                X, positions, test_size=min(test_size, 0.5), random_state=42
            )
            
            # 正規化特徵
            X_train_norm, X_test_norm = self.processor.normalize_features(X_train, X_test)
            
            # 建立模型
            input_shape = (X.shape[1],)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2, activation='linear')  # x, y 座標
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # 訓練模型
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
            ]
            
            history = model.fit(
                X_train_norm, pos_train,
                validation_data=(X_test_norm, pos_test),
                epochs=150,
                batch_size=16,
                callbacks=callbacks,
                verbose=0
            )
            
            # 評估模型
            pos_pred = model.predict(X_test_norm)
            mse = mean_squared_error(pos_test, pos_pred)
            
            # 計算平均定位誤差
            distances = np.sqrt(np.sum((pos_test - pos_pred) ** 2, axis=1))
            mean_distance_error = np.mean(distances)
            
            print(f"  {location_key} - MSE: {mse:.4f}, 平均定位誤差: {mean_distance_error:.4f}")
            
            # 儲存模型
            model_path = os.path.join(self.save_dir, 'models', 'position_predictors', location_key)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            
            position_models[location_key] = model
            position_results[location_key] = {
                'mse': mse,
                'mean_distance_error': mean_distance_error,
                'history': history.history,
                'n_samples': len(X)
            }
        
        self.models['position_predictors'] = position_models
        self.results['position_predictors'] = position_results
        
        return position_models, position_results
    
    def run_full_pipeline(self):
        """執行完整的訓練流程"""
        print("="*50)
        print("開始執行 HADNN 拆分模型訓練流程")
        print("="*50)
        
        # 1. 處理數據
        print("\n1. 處理數據...")
        processed_data = self.processor.process_all_data()
        
        # 2. 訓練建築物分類器
        print("\n2. 訓練建築物分類器...")
        building_model, building_acc = self.train_building_classifier(
            processed_data['X'], 
            processed_data['building_labels']
        )
        
        # 3. 訓練樓層分類器
        print("\n3. 訓練樓層分類器...")
        floor_model, floor_acc = self.train_floor_classifier(
            processed_data['X'],
            processed_data['building_labels'],
            processed_data['floor_labels']
        )
        
        # 4. 訓練位置預測器
        print("\n4. 訓練位置預測器...")
        position_models, position_results = self.train_position_predictors(
            processed_data['data_splits']
        )
        
        # 5. 儲存訓練結果
        print("\n5. 儲存訓練結果...")
        self.save_training_results(processed_data)
        
        print("\n" + "="*50)
        print("訓練流程完成！")
        print(f"建築物分類器準確度: {building_acc:.4f}")
        print(f"樓層分類器準確度: {floor_acc:.4f}")
        print(f"訓練了 {len(position_models)} 個位置預測器")
        print("="*50)
        
        return self.models, self.results
    
    def save_training_results(self, processed_data):
        """儲存訓練結果和元數據"""
        results_dir = os.path.join(self.save_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 儲存訓練結果
        with open(os.path.join(results_dir, 'training_results.json'), 'w') as f:
            # 將 numpy 類型轉換為 Python 原生類型以便 JSON 序列化
            json_results = {}
            for key, value in self.results.items():
                if key == 'position_predictors':
                    json_results[key] = {}
                    for loc, metrics in value.items():
                        json_results[key][loc] = {
                            'mse': float(metrics['mse']),
                            'mean_distance_error': float(metrics['mean_distance_error']),
                            'n_samples': int(metrics['n_samples'])
                        }
                else:
                    json_results[key] = {
                        'accuracy': float(value['accuracy'])
                    }
            
            json.dump(json_results, f, indent=2)
        
        # 儲存編碼器和元數據
        metadata = {
            'building_encoder': processed_data['building_encoder'],
            'floor_encoder': processed_data['floor_encoder'],
            'bssid_list': processed_data['bssid_list'],
            'target_networks': self.processor.target_networks
        }
        
        with open(os.path.join(results_dir, 'metadata.pickle'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"訓練結果已儲存到: {results_dir}")

if __name__ == "__main__":
    # 設定路徑
    data_path = "/workspace/points/scan13"
    save_dir = "../"
    
    # 建立必要的目錄
    os.makedirs(os.path.join(save_dir, 'models', 'building_classifier'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models', 'floor_classifier'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models', 'position_predictors'), exist_ok=True)
    
    # 執行訓練流程
    pipeline = TrainingPipeline(data_path, save_dir)
    models, results = pipeline.run_full_pipeline()
