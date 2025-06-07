import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model_improvements import AttentionLayer

def convert_saved_models_to_tflite():
    """將已訓練的 Keras 模型轉換為 TensorFlow Lite 格式"""
    
    # 修改來源與輸出資料夾
    trained_models_dir = 'enhanced_models'
    tflite_output_dir = 'tflite_models'
    
    # 建立輸出資料夾
    os.makedirs(tflite_output_dir, exist_ok=True)
    
    # 載入資料集資訊
    try:
        with open('hadnn_data\dataset_config.json', 'r') as f:
            dataset_config = json.load(f)
        print("✓ 成功載入資料集配置")
    except Exception as e:
        print(f"✗ 載入資料集配置失敗: {e}")
        print("  將使用預設配置")
        dataset_config = {
            'n_rss': 0, 'n_buildings': 0, 'n_floors': 0,
            'lo_mean': 0, 'lo_std': 1, 'la_mean': 0, 'la_std': 1,
            'building_mapping': {}, 'floor_mapping': {}
        }
    
    # 載入一些訓練資料作為代表性資料集
    try:
        sample_data = np.load('hadnn_data\\train_x.npy')
        print(f"✓ 載入校正資料: {len(sample_data)} 個樣本")
        
        # 為了避免記憶體過載，只使用前10000個樣本
        if len(sample_data) > 10000:
            sample_data = sample_data[:10000]
        
        # 產生代表性資料集函數
        def representative_dataset():
            for i in range(len(sample_data)):
                yield [sample_data[i:i+1].astype(np.float32)]
                
    except Exception as e:
        print(f"✗ 載入校正資料失敗: {e}")
        print("  將跳過量化優化")
        representative_dataset = None
    
    converted_models = {}
    # 找出所有可轉換的模型（資料夾或 .h5 檔案）
    model_candidates = []
    for entry in os.listdir(trained_models_dir):
        full_path = os.path.join(trained_models_dir, entry)
        if os.path.isdir(full_path):
            model_candidates.append((entry, full_path))
        elif entry.lower().endswith('.h5'):
            # 以檔名（不含副檔名）作為模型名稱
            model_name = os.path.splitext(entry)[0]
            model_candidates.append((model_name, full_path))

    if not model_candidates:
        print("✗ 找不到任何已訓練模型，請確保模型已儲存在 enhanced_models 目錄中")
        return

    print(f"\n找到 {len(model_candidates)} 個待轉換模型")
    
    # 遍歷所有已訓練的模型
    for model_name, model_path in tqdm(model_candidates, desc="轉換模型"):
        try:
            print(f"\n=== 轉換 {model_name} ===")
            # 載入 Keras 模型，加入 custom_objects 參數
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'AttentionLayer': AttentionLayer})
            print(f"✓ 載入模型: {model_path}")
            
            # 建立轉換器
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # 設定優化選項
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 嘗試啟用浮點16量化，這可以減少模型大小
            converter.target_spec.supported_types = [tf.float16]
            
            # 如果有代表性資料集，使用它來優化模型
            if representative_dataset:
                converter.representative_dataset = representative_dataset
                print("✓ 使用代表性資料集進行優化")
            
            # 增加轉換選項，嘗試減少記憶體使用
            converter.experimental_new_converter = True
            
            # 轉換模型
            print("⟳ 正在轉換模型，這可能需要一點時間...")
            tflite_model = converter.convert()
            
            # 儲存 TFLite 模型
            tflite_filename = f'{model_name.lower()}.tflite'
            tflite_path = os.path.join(tflite_output_dir, tflite_filename)
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # 計算模型大小
            model_size = len(tflite_model) / (1024 * 1024)  # MB
            
            print(f"✓ 轉換完成: {tflite_path}")
            print(f"  模型大小: {model_size:.2f} MB")
            
            # 釋放記憶體
            tf.keras.backend.clear_session()
            
            converted_models[model_name] = {
                'filename': tflite_filename,
                'size_mb': round(model_size, 2),
                'input_shape': [dim if dim is not None else -1 for dim in model.input_shape],
                'output_shape': [
                    [dim if dim is not None else -1 for dim in output.shape] 
                    for output in model.outputs
                ] if isinstance(model.outputs, list) else 
                [dim if dim is not None else -1 for dim in model.output_shape]
            }
            
        except Exception as e:
            print(f"✗ 轉換 {model_name} 失敗: {str(e)}")
    
    # 建立模型資訊檔案
    model_info = {
        'dataset_info': {
            'n_rss': dataset_config['n_rss'],
            'n_buildings': dataset_config['n_buildings'],
            'n_floors': dataset_config['n_floors'],
            'lo_mean': dataset_config['lo_mean'],
            'lo_std': dataset_config['lo_std'],
            'la_mean': dataset_config['la_mean'],
            'la_std': dataset_config['la_std'],
            'building_mapping': dataset_config['building_mapping'],
            'floor_mapping': dataset_config['floor_mapping']
        },
        'models': converted_models
    }
    
    # 儲存模型資訊
    with open(os.path.join(tflite_output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 轉換完成 ===")
    print(f"轉換了 {len(converted_models)} 個模型")
    print(f"檔案儲存位置: {tflite_output_dir}")
    print("可用模型:")
    for name, info in converted_models.items():
        print(f"  - {name}: {info['filename']} ({info['size_mb']} MB)")

if __name__ == "__main__":
    # 設定 TensorFlow 記憶體管理
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 限制 GPU 記憶體使用
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 記憶體設置失敗: {e}")
    
    print("開始轉換 Keras 模型為 TensorFlow Lite 格式...")
    convert_saved_models_to_tflite()
