import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error
import importlib

# 檢查是否安裝了 tabulate 套件
TABULATE_AVAILABLE = importlib.util.find_spec("tabulate") is not None

# 自定義格式化表格函數，在沒有 tabulate 時使用
def format_table(df):
    """
    格式化 DataFrame 為簡單的表格文字
    當系統中沒有 tabulate 套件時，用這個函數代替 to_markdown()
    """
    cols = df.columns
    header = " | ".join(str(c) for c in cols)
    separator = "-" * len(header)
    rows = []
    
    for _, row in df.iterrows():
        # 格式化每個單元格的值
        formatted_values = []
        for col in cols:
            val = row[col]
            # 處理不同類型的數值格式
            if isinstance(val, (int, np.integer)):
                formatted_values.append(f"{val}")
            elif isinstance(val, (float, np.floating)):
                formatted_values.append(f"{val:.4f}")
            else:
                formatted_values.append(str(val))
        
        rows.append(" | ".join(formatted_values))
    
    # 組合表頭和行
    return header + "\n" + separator + "\n" + "\n".join(rows)

class ModelComparison:
    """WiFi 室內定位模型比較類"""
    
    def __init__(self, data_dir='./hadnn_data', models_to_compare=None):
        """
        初始化模型比較器
        
        參數:
            data_dir: 測試資料目錄
            models_to_compare: 要比較的模型列表，每項包含 {name, path, type}
        """
        self.data_dir = data_dir
        
        # 預設模型列表
        if models_to_compare is None:
            models_to_compare = [
                {
                    'name': '基礎 HADNN',
                    'path': './hadnn_models/hadnn_model.h5',
                    'type': 'keras'
                },
                {
                    'name': '增強 HADNN',
                    'path': './enhanced_models/enhanced_hadnn_model.h5',
                    'type': 'keras'
                },
                {
                    'name': '混合模型',
                    'path': './hybrid_models',
                    'type': 'hybrid'
                }
            ]
        
        self.models_to_compare = models_to_compare
        self.results = {}  # 儲存評估結果
        self.test_data = None  # 測試資料
        
    def load_test_data(self):
        """載入測試資料"""
        print("載入測試資料...")
        try:
            self.test_x = np.load(os.path.join(self.data_dir, 'test_x.npy'))
            self.test_b = np.load(os.path.join(self.data_dir, 'test_b.npy'))
            self.test_c = np.load(os.path.join(self.data_dir, 'test_c.npy'))
            self.test_y = np.load(os.path.join(self.data_dir, 'test_y.npy'))
            
            # 確保標籤維度正確
            self.test_b = self.test_b.reshape(-1)
            
            # 確保座標是二維的
            if len(self.test_c.shape) == 1:
                if self.test_c.shape[0] % 2 == 0 and self.test_c.shape[0] // 2 == self.test_x.shape[0]:
                    # 兩個向量分別存儲 x 和 y
                    half_len = self.test_c.shape[0] // 2
                    self.test_c = np.column_stack((self.test_c[:half_len], self.test_c[half_len:]))
                else:
                    # 一個向量可能是樓層標籤
                    from_test_y = False
                    try:
                        # 嘗試載入樓層標籤
                        self.test_f = np.load(os.path.join(self.data_dir, 'test_f.npy'))
                        self.test_f = self.test_f.reshape(-1)
                    except:
                        # 如果找不到，使用 test_y 作為樓層標籤
                        self.test_f = self.test_y
                        from_test_y = True
                    
                    if not from_test_y:
                        # 如果 test_y 不是樓層標籤，則可能是座標
                        self.test_c = self.test_y.reshape(-1, 2) if len(self.test_y.shape) > 1 else self.test_c
            
            # 數據預處理
            from original_hadnn import advanced_data_preprocessing
            self.test_x_enhanced, _ = advanced_data_preprocessing(self.test_x)
            
            # 載入標籤映射
            config_path = os.path.join(self.data_dir, 'dataset_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'building_mapping': {},
                    'floor_mapping': {}
                }
                
            print(f"測試資料形狀: x={self.test_x.shape}, b={self.test_b.shape}, c={self.test_c.shape}")
            return True
                
        except Exception as e:
            print(f"載入測試資料失敗: {e}")
            return False
    
    def load_and_evaluate_model(self, model_info):
        """載入並評估單個模型"""
        name = model_info['name']
        path = model_info['path']
        model_type = model_info['type']
        
        print(f"\n評估模型: {name} ({path})")
        
        # 檢查檔案是否存在
        if not os.path.exists(path):
            print(f"錯誤: 找不到模型檔案 {path}")
            return None
        
        try:
            if model_type == 'keras':
                # 載入 Keras 模型
                # 從 model_improvements 導入自定義層
                from original_hadnn import AttentionLayer
                custom_objects = {'AttentionLayer': AttentionLayer}
                
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                
                # 進行預測
                predictions = model.predict(self.test_x_enhanced)
                
                # 解析預測結果
                if isinstance(predictions, list):
                    building_preds = np.argmax(predictions[0], axis=1)
                    floor_preds = np.argmax(predictions[1], axis=1) if len(predictions) > 1 else np.zeros_like(self.test_b)
                    position_preds = predictions[2] if len(predictions) > 2 else np.zeros((len(self.test_x), 2))
                else:
                    # 如果模型只有一個輸出，假設它是位置座標
                    building_preds = np.zeros_like(self.test_b)
                    floor_preds = np.zeros_like(self.test_b)
                    position_preds = predictions
            
            elif model_type == 'hybrid':
                # 載入混合模型
                from random_forest import HybridPositionModel
                
                try:
                    hybrid_model = HybridPositionModel.load(path)
                    
                    # 進行預測
                    building_output, floor_proba, position_preds = hybrid_model.predict(self.test_x_enhanced)
                    
                    # 解析預測結果
                    building_preds = np.argmax(building_output, axis=1)
                    floor_preds = np.argmax(floor_proba, axis=1)
                except Exception as e:
                    print(f"載入混合模型時出錯: {e}")
                    return None
            
            else:
                print(f"不支援的模型類型: {model_type}")
                return None
            
            # 計算評估指標
            building_accuracy = accuracy_score(self.test_b, building_preds)
            
            try:
                # 樓層準確率，檢查是否需要使用 test_f
                if hasattr(self, 'test_f'):
                    floor_accuracy = accuracy_score(self.test_f, floor_preds)
                else:
                    floor_accuracy = accuracy_score(self.test_y, floor_preds)
            except:
                floor_accuracy = 0
            
            # 位置誤差
            position_mse = mean_squared_error(self.test_c, position_preds)
            position_rmse = np.sqrt(position_mse)
            
            # 歐氏距離誤差
            euclidean_distances = np.sqrt(np.sum((self.test_c - position_preds)**2, axis=1))
            mean_error = np.mean(euclidean_distances)
            median_error = np.median(euclidean_distances)
            
            # 儲存結果
            result = {
                'building_accuracy': building_accuracy,
                'floor_accuracy': floor_accuracy,
                'position_mse': position_mse,
                'position_rmse': position_rmse,
                'mean_error': mean_error,
                'median_error': median_error,
                'predictions': {
                    'building': building_preds,
                    'floor': floor_preds,
                    'position': position_preds
                }
            }
            
            print(f"模型評估完成:")
            print(f"  建築物分類準確率: {building_accuracy:.2%}")
            print(f"  樓層分類準確率: {floor_accuracy:.2%}")
            print(f"  位置預測平均誤差: {mean_error:.4f}")
            print(f"  位置預測中位數誤差: {median_error:.4f}")
            
            return result
            
        except Exception as e:
            print(f"評估模型 {name} 時出錯: {e}")
            return None
    
    def compare_models(self):
        """比較所有模型"""
        # 載入測試資料
        if not self.load_test_data():
            print("無法進行模型比較，因為測試資料載入失敗")
            return False
        
        # 評估每個模型
        for model_info in self.models_to_compare:
            result = self.load_and_evaluate_model(model_info)
            if result:
                self.results[model_info['name']] = result
        
        # 顯示比較結果
        if self.results:
            self.display_comparison()
            return True
        else:
            print("沒有成功評估任何模型")
            return False
    
    def display_comparison(self):
        """顯示模型比較結果"""
        print("\n=== 模型比較結果 ===")
        
        # 創建比較表格
        data = []
        for name, result in self.results.items():
            data.append({
                '模型名稱': name,
                '建築物準確率': result['building_accuracy'] * 100,
                '樓層準確率': result['floor_accuracy'] * 100,
                '位置平均誤差': result['mean_error'],
                '位置中位數誤差': result['median_error']
            })
        
        df = pd.DataFrame(data)
        
        # 顯示表格
        print(df.to_string(index=False))
    
    def generate_comparison_report(self, output_dir='./model_comparison'):
        """生成詳細的比較報告"""
        if not self.results:
            print("沒有模型評估結果可供報告")
            return
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 準備資料
        data = []
        for name, result in self.results.items():
            data.append({
                'Model': name,
                'Building Accuracy (%)': result['building_accuracy'] * 100,
                'Floor Accuracy (%)': result['floor_accuracy'] * 100,
                'Mean Error (m)': result['mean_error'],
                'Median Error (m)': result['median_error'],
                'RMSE (m)': result['position_rmse']
            })
        
        df = pd.DataFrame(data)
        df_display = df.copy()
        
        # 找出每列最佳值的索引
        best_building = df['Building Accuracy (%)'].idxmax()
        best_floor = df['Floor Accuracy (%)'].idxmax()
        best_mean = df['Mean Error (m)'].idxmin()
        best_median = df['Median Error (m)'].idxmin()
        best_rmse = df['RMSE (m)'].idxmin()
        
        # 格式化數值 (改進的顯示方式)
        for col in df_display.columns:
            if col == 'Model':
                continue
            if 'Accuracy' in col:
                df_display[col] = df_display[col].map('{:.2f}%'.format)
            else:
                df_display[col] = df_display[col].map('{:.4f}'.format)
        
        # 寫入 Markdown 格式的報告
        with open(os.path.join(output_dir, 'model_comparison.md'), 'w') as f:
            f.write("# WiFi 室內定位模型比較報告\n\n")
            f.write("## 綜合性能比較\n\n")
            
            # 使用格式化函數避免 tabulate 依賴
            if TABULATE_AVAILABLE:
                try:
                    f.write(df_display.to_markdown(index=False))
                except Exception:
                    f.write(format_table(df_display))
            else:
                f.write(format_table(df_display))
            
            f.write("\n\n## 最佳表現分析\n\n")
            f.write(f"* **建築物分類**: {df.iloc[best_building]['Model']} 達到最高準確率 {df.iloc[best_building]['Building Accuracy (%)']:.2f}%\n")
            f.write(f"* **樓層分類**: {df.iloc[best_floor]['Model']} 達到最高準確率 {df.iloc[best_floor]['Floor Accuracy (%)']:.2f}%\n")
            f.write(f"* **位置預測平均誤差**: {df.iloc[best_mean]['Model']} 達到最低誤差 {df.iloc[best_mean]['Mean Error (m)']:.4f} 米\n")
            f.write(f"* **位置預測中位數誤差**: {df.iloc[best_median]['Model']} 達到最低誤差 {df.iloc[best_median]['Median Error (m)']:.4f} 米\n")
            f.write(f"* **位置預測 RMSE**: {df.iloc[best_rmse]['Model']} 達到最低 RMSE {df.iloc[best_rmse]['RMSE (m)']:.4f} 米\n")
            
            f.write("\n## 詳細評估指標\n\n")
            for name, result in self.results.items():
                f.write(f"### {name}\n\n")
                f.write(f"* 建築物分類準確率: {result['building_accuracy']:.2%}\n")
                f.write(f"* 樓層分類準確率: {result['floor_accuracy']:.2%}\n")
                f.write(f"* 位置預測平均誤差: {result['mean_error']:.4f} 米\n")
                f.write(f"* 位置預測中位數誤差: {result['median_error']:.4f} 米\n")
                f.write(f"* 位置預測 RMSE: {result['position_rmse']:.4f} 米\n\n")
        
        # 生成圖表
        self.generate_comparison_charts(output_dir)
        
        print(f"比較報告已生成至: {output_dir}")
    
    def generate_comparison_charts(self, output_dir):
        """生成比較圖表"""
        # 準備資料
        names = list(self.results.keys())
        building_acc = [self.results[name]['building_accuracy'] * 100 for name in names]
        floor_acc = [self.results[name]['floor_accuracy'] * 100 for name in names]
        mean_errors = [self.results[name]['mean_error'] for name in names]
        
        # 設置圖表樣式
        plt.style.use('ggplot')
        
        # 1. 準確率對比圖
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, building_acc, width, label='建築物分類')
        plt.bar(x + width/2, floor_acc, width, label='樓層分類')
        
        plt.xlabel('模型')
        plt.ylabel('準確率 (%)')
        plt.title('不同模型的分類準確率對比')
        plt.xticks(x, names)
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'classification_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 位置誤差對比圖
        plt.figure(figsize=(10, 6))
        
        plt.bar(names, mean_errors, color='skyblue')
        
        plt.xlabel('模型')
        plt.ylabel('平均誤差 (米)')
        plt.title('不同模型的位置預測平均誤差對比')
        plt.xticks(rotation=15)
        
        plt.savefig(os.path.join(output_dir, 'position_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 誤差分布圖
        plt.figure(figsize=(12, 8))
        
        for i, name in enumerate(names):
            errors = np.sqrt(np.sum((self.test_c - self.results[name]['predictions']['position'])**2, axis=1))
            plt.hist(errors, bins=30, alpha=0.7, label=name)
        
        plt.xlabel('誤差 (米)')
        plt.ylabel('樣本數')
        plt.title('不同模型的位置預測誤差分布')
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 檢查是否有 tabulate 套件，如果沒有則顯示警告
    if not TABULATE_AVAILABLE:
        print("警告: 缺少 'tabulate' 套件。將使用簡單格式化輸出表格。")
        print("提示: 使用 pip install tabulate 安裝此套件以獲得更好的表格顯示效果。")
    
    # 創建比較器
    comparer = ModelComparison()
    
    # 執行比較
    if comparer.compare_models():
        # 生成報告
        comparer.generate_comparison_report()

if __name__ == "__main__":
    main()
