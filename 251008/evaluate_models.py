import os
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from mlp import MLPPositionModel, AttentionLayer, NpEncoder  # 導入 NpEncoder 類
from hadnn_n_random_forest import IntegratedPositionModel

# 添加中文字體支援
def setup_chinese_font():
    """設置中文字體支援"""
    import matplotlib
    from matplotlib.font_manager import fontManager
    
    # 檢查可用字型
    def get_available_chinese_font():
        chinese_fonts = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DFKai-SB', 
            'Arial Unicode MS', 'Noto Sans CJK TC', 'Noto Sans TC', 
            'Noto Sans CJK JP', 'Noto Sans CJK SC'
        ]
        
        available_fonts = [font.name for font in fontManager.ttflist]
        print("檢查可用中文字體...")
        
        # 尋找可用的中文字型
        for font in chinese_fonts:
            if font in available_fonts:
                print(f"使用中文字體: {font}")
                return font
        
        print("找不到預設中文字體，將使用系統默認字體")
        return None
    
    # 設定中文字型
    chinese_font = get_available_chinese_font()
    if chinese_font:
        matplotlib.rcParams['font.family'] = chinese_font
    else:
        # 備用方案：使用支援 Unicode 的字型，並確保 Matplotlib 可以顯示中文
        print("使用備用字體配置以支援中文")
        plt.rcParams['axes.unicode_minus'] = False
        
        # 嘗試使用多種通用字體
        if hasattr(plt, 'style') and 'font.sans-serif' in plt.rcParams:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

def load_test_data():
    """載入測試數據"""
    hadnn_data_dir = "./hadnn_data"
    
    test_x = np.load(os.path.join(hadnn_data_dir, 'test_x.npy'))
    test_b = np.load(os.path.join(hadnn_data_dir, 'test_b.npy'))
    test_y = np.load(os.path.join(hadnn_data_dir, 'test_y.npy'))
    test_c = np.load(os.path.join(hadnn_data_dir, 'test_c.npy'))
    
    # 載入測試點名稱 (如果檔案存在)
    test_names_path = os.path.join(hadnn_data_dir, 'test_names.json')
    try:
        if os.path.exists(test_names_path):
            with open(test_names_path, 'r', encoding='utf-8') as f:
                test_names = json.load(f)
        else:
            # 如果檔案不存在，創建默認的測試點名稱
            print(f"警告: 找不到 {test_names_path} 檔案，將使用默認名稱。")
            test_names = [f"Sample_{i}" for i in range(len(test_x))]
    except Exception as e:
        print(f"載入測試點名稱時發生錯誤: {e}，將使用默認名稱。")
        test_names = [f"Sample_{i}" for i in range(len(test_x))]
    
    # 載入配置
    config_path = os.path.join(hadnn_data_dir, 'dataset_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"載入配置檔案時發生錯誤: {e}，將使用空配置。")
        config = {
            'building_mapping': {},
            'floor_mapping': {},
            'point_mapping': {}
        }
    
    return test_x, test_b, test_y, test_c, test_names, config

def evaluate_model(model, model_name, test_x, test_b, test_y, test_c):
    """評估單個模型"""
    print(f"\n=== 評估 {model_name} 模型 ===")
    
    try:
        # 獲取預測結果
        if hasattr(model, 'predict'):
            # 深度學習模型
            predictions = model.predict(test_x)
            building_pred = np.argmax(predictions[0], axis=1)
            floor_pred = np.argmax(predictions[1], axis=1)
            position_pred = predictions[2]
        else:
            # TensorFlow模型
            predictions = model(test_x)
            building_pred = np.argmax(predictions[0], axis=1)
            floor_pred = np.argmax(predictions[1], axis=1)
            position_pred = predictions[2].numpy()
        
        # 計算準確率
        building_accuracy = np.mean(building_pred == test_b) * 100
        floor_accuracy = np.mean(floor_pred == test_y) * 100
        
        # 計算位置誤差
        position_errors = np.sqrt(np.sum((test_c - position_pred) ** 2, axis=1))
        mean_error = np.mean(position_errors)
        median_error = np.median(position_errors)
        std_error = np.std(position_errors)
        
        # 新增：計算條件位置誤差（只針對建築物和樓層都預測正確的樣本）
        correct_building = (building_pred == test_b)
        correct_floor = (floor_pred == test_y)
        correct_both = correct_building & correct_floor
        
        if np.any(correct_both):
            conditional_errors = position_errors[correct_both]
            conditional_mean_error = np.mean(conditional_errors)
            conditional_median_error = np.median(conditional_errors)
            conditional_count = np.sum(correct_both)
        else:
            conditional_mean_error = float('inf')
            conditional_median_error = float('inf')
            conditional_count = 0
        
        results = {
            'model_name': model_name,
            'building_accuracy': building_accuracy,
            'floor_accuracy': floor_accuracy,
            'mean_position_error': mean_error,
            'median_position_error': median_error,
            'std_position_error': std_error,
            'conditional_mean_position_error': conditional_mean_error,
            'conditional_median_position_error': conditional_median_error,
            'conditional_correct_count': conditional_count,
            'position_errors': position_errors.tolist()
        }
        
        print(f"建築物分類準確率: {building_accuracy:.2f}%")
        print(f"樓層分類準確率: {floor_accuracy:.2f}%")
        print(f"整體位置預測平均誤差: {mean_error:.4f}")
        print(f"條件位置預測平均誤差: {conditional_mean_error:.4f}")
        print(f"條件正確樣本數: {conditional_count}/{len(position_errors)} ({conditional_count/len(position_errors)*100:.1f}%)")
        print(f"位置誤差統計: 中位數={median_error:.4f}, 標準差={std_error:.4f}")
        
        return results
        
    except Exception as e:
        print(f"評估 {model_name} 時發生錯誤: {e}")
        return None

def load_all_models():
    """載入所有可用的模型"""
    models_dir = "./models"
    models = {}
    
    # 載入MLP模型
    try:
        mlp_model = MLPPositionModel.load(models_dir)
        models['MLP'] = mlp_model.model
        print("✓ 成功載入MLP模型")
    except Exception as e:
        print(f"✗ 載入MLP模型失敗: {e}")
    
    # 載入HADNN模型
    try:
        custom_objects = {'AttentionLayer': AttentionLayer}
        hadnn_path = os.path.join(models_dir, 'original_hadnn.h5')
        if os.path.exists(hadnn_path):
            hadnn_model = tf.keras.models.load_model(hadnn_path, custom_objects=custom_objects)
            models['Original HADNN'] = hadnn_model
            print("✓ 成功載入Original HADNN模型")
    except Exception as e:
        print(f"✗ 載入Original HADNN模型失敗: {e}")
    
    # 載入整合模型
    try:
        integrated_path = os.path.join(models_dir, 'hadnn_n_random_forest.h5')
        if os.path.exists(integrated_path):
            integrated_model = tf.keras.models.load_model(integrated_path, custom_objects={'AttentionLayer': AttentionLayer})
            models['HADNN + Random Forest'] = integrated_model
            print("✓ 成功載入HADNN + Random Forest模型")
    except Exception as e:
        print(f"✗ 載入HADNN + Random Forest模型失敗: {e}")
    
    return models

def plot_comparison_results(all_results, save_dir="./results"):
    """繪製比較結果圖表"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    model_names = [r['model_name'] for r in all_results if r is not None]
    building_accs = [r['building_accuracy'] for r in all_results if r is not None]
    floor_accs = [r['floor_accuracy'] for r in all_results if r is not None]
    mean_errors = [r['mean_position_error'] for r in all_results if r is not None]
    
    try:
        # 繪製準確率比較圖
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 使用英文標籤代替中文標籤以避免字體問題
        # 建築物分類準確率
        ax1.bar(model_names, building_accs)
        ax1.set_title('Building Classification Accuracy')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 樓層分類準確率
        ax2.bar(model_names, floor_accs)
        ax2.set_title('Floor Classification Accuracy')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 位置預測誤差
        ax3.bar(model_names, mean_errors)
        ax3.set_title('Position Prediction Mean Error')
        ax3.set_ylabel('Error')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"繪製比較圖表時發生錯誤: {e}")
    finally:
        plt.close()  # 確保圖形被關閉，即使發生例外也會執行
    
    print(f"比較圖表已保存至: {os.path.join(save_dir, 'model_comparison.png')}")

def main():
    """主函數"""
    print("=== 開始模型評估 ===")
    
    # 設置中文字體支援
    setup_chinese_font()
    
    # 載入測試數據
    test_x, test_b, test_y, test_c, test_names, config = load_test_data()
    print(f"測試數據載入完成: {len(test_x)} 個樣本")
    
    # 載入所有模型
    models = load_all_models()
    
    if not models:
        print("錯誤: 沒有可用的模型")
        return
    
    # 評估所有模型
    all_results = []
    for model_name, model in models.items():
        result = evaluate_model(model, model_name, test_x, test_b, test_y, test_c)
        if result:
            all_results.append(result)
    
    # 保存結果
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NpEncoder)  # 使用 NpEncoder 處理 NumPy 類型
    
    # 繪製比較圖表
    if len(all_results) > 1:
        plot_comparison_results(all_results, results_dir)
    
    # 顯示總結
    print(f"\n{'='*50}")
    print("=== 評估結果總結 ===")
    for result in all_results:
        print(f"{result['model_name']}:")
        print(f"  建築物準確率: {result['building_accuracy']:.2f}%")
        print(f"  樓層準確率: {result['floor_accuracy']:.2f}%")
        print(f"  位置誤差: {result['mean_position_error']:.4f}")
        print()
    
    print(f"詳細結果已保存至: {results_dir}")

if __name__ == "__main__":
    main()