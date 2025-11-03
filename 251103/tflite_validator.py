import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

def verify_tflite_accuracy(keras_model, tflite_path, test_x, test_b, test_y, test_c, tolerance=1e-3):
    """
    驗證 TFLite 模型與原始 Keras 模型的準確性差異
    
    參數:
        keras_model: 原始 Keras 模型
        tflite_path: TFLite 模型檔案路徑
        test_x, test_b, test_y, test_c: 測試資料
        tolerance: 允許的數值誤差容忍度
    """
    print("=" * 50)
    print("TFLite 模型準確性驗證")
    print("=" * 50)
    
    # 使用原始 Keras 模型進行預測
    print("執行 Keras 模型預測...")
    keras_predictions = keras_model.predict(test_x, verbose=0)
    
    # 載入 TFLite 模型
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite 模型資訊:")
    print(f"  輸入形狀: {input_details[0]['shape']}, 類型: {input_details[0]['dtype']}")
    for i, detail in enumerate(output_details):
        output_name = detail.get('name', f'output_{i}')
        print(f"  輸出 {i} ({output_name}): 形狀={detail['shape']}, 類型={detail['dtype']}")
    
    # Keras 模型輸出分析
    print(f"\nKeras 模型輸出分析:")
    keras_building = np.argmax(keras_predictions[0], axis=1)
    keras_floor = np.argmax(keras_predictions[1], axis=1)
    keras_position = keras_predictions[2]
    
    print(f"  建築物分類準確率: {np.mean(keras_building == test_b) * 100:.2f}%")
    print(f"  樓層分類準確率: {np.mean(keras_floor == test_y) * 100:.2f}%")
    print(f"  位置預測誤差: {np.mean(np.sqrt(np.sum((test_c - keras_position)**2, axis=1))):.4f}")
    
    # TFLite 模型預測
    print(f"\n執行 TFLite 模型預測...")
    tflite_predictions = []
    for i in range(len(output_details)):
        tflite_predictions.append([])
    
    sample_count = min(test_x.shape[0], 100)  # 限制樣本數以加快驗證
    for i in range(sample_count):
        input_data = test_x[i:i+1].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        for j, output_detail in enumerate(output_details):
            output = interpreter.get_tensor(output_detail['index'])
            tflite_predictions[j].append(output[0])
    
    # 轉換為 numpy 陣列
    for i in range(len(tflite_predictions)):
        tflite_predictions[i] = np.array(tflite_predictions[i])
    
    # 智能輸出映射
    print(f"\n智能輸出映射分析:")
    
    # 分析每個 TFLite 輸出的特徵
    mapping_candidates = {}
    
    for tflite_idx, tflite_pred in enumerate(tflite_predictions):
        for keras_idx, keras_pred in enumerate(keras_predictions):
            if tflite_pred.shape == keras_pred[:sample_count].shape:
                # 計算數值相似性
                diff = np.mean(np.abs(tflite_pred - keras_pred[:sample_count]))
                print(f"  TFLite[{tflite_idx}] vs Keras[{keras_idx}]: 形狀匹配, 平均差異={diff:.6f}")
                
                if keras_idx not in mapping_candidates:
                    mapping_candidates[keras_idx] = []
                mapping_candidates[keras_idx].append((tflite_idx, diff))
    
    # 選擇最佳映射
    final_mapping = {}
    used_tflite_indices = set()
    
    for keras_idx in sorted(mapping_candidates.keys()):
        candidates = sorted(mapping_candidates[keras_idx], key=lambda x: x[1])  # 按差異排序
        for tflite_idx, diff in candidates:
            if tflite_idx not in used_tflite_indices:
                final_mapping[keras_idx] = tflite_idx
                used_tflite_indices.add(tflite_idx)
                print(f"  最佳映射: Keras[{keras_idx}] -> TFLite[{tflite_idx}] (差異: {diff:.6f})")
                break
    
    # 驗證分類準確率
    print(f"\n分類準確率比較:")
    
    if 0 in final_mapping and 1 in final_mapping:
        # 建築物分類
        tflite_building = np.argmax(tflite_predictions[final_mapping[0]], axis=1)
        building_match = np.mean(keras_building[:sample_count] == tflite_building) * 100
        tflite_building_acc = np.mean(tflite_building == test_b[:sample_count]) * 100
        keras_building_acc = np.mean(keras_building[:sample_count] == test_b[:sample_count]) * 100
        
        print(f"  建築物分類:")
        print(f"    Keras 準確率: {keras_building_acc:.2f}%")
        print(f"    TFLite 準確率: {tflite_building_acc:.2f}%")
        print(f"    預測一致性: {building_match:.2f}%")
        
        # 樓層分類
        tflite_floor = np.argmax(tflite_predictions[final_mapping[1]], axis=1)
        floor_match = np.mean(keras_floor[:sample_count] == tflite_floor) * 100
        tflite_floor_acc = np.mean(tflite_floor == test_y[:sample_count]) * 100
        keras_floor_acc = np.mean(keras_floor[:sample_count] == test_y[:sample_count]) * 100
        
        print(f"  樓層分類:")
        print(f"    Keras 準確率: {keras_floor_acc:.2f}%")
        print(f"    TFLite 準確率: {tflite_floor_acc:.2f}%")
        print(f"    預測一致性: {floor_match:.2f}%")
        
        # 判斷轉換是否成功
        success = (
            building_match >= 95.0 and  # 建築物預測一致性 >= 95%
            floor_match >= 95.0 and     # 樓層預測一致性 >= 95%
            abs(keras_building_acc - tflite_building_acc) <= 5.0 and  # 建築物準確率差異 <= 5%
            abs(keras_floor_acc - tflite_floor_acc) <= 5.0            # 樓層準確率差異 <= 5%
        )
        
        if success:
            print("\n✅ TFLite 轉換驗證成功")
        else:
            print("\n⚠️  TFLite 轉換存在問題，建議檢查:")
            if building_match < 95.0:
                print(f"    - 建築物預測一致性過低: {building_match:.1f}%")
            if floor_match < 95.0:
                print(f"    - 樓層預測一致性過低: {floor_match:.1f}%")
            if abs(keras_building_acc - tflite_building_acc) > 5.0:
                print(f"    - 建築物準確率差異過大: {abs(keras_building_acc - tflite_building_acc):.1f}%")
            if abs(keras_floor_acc - tflite_floor_acc) > 5.0:
                print(f"    - 樓層準確率差異過大: {abs(keras_floor_acc - tflite_floor_acc):.1f}%")
        
        return success
    else:
        print("⚠️  無法找到足夠的輸出映射進行驗證")
        return False

def compare_model_sizes(h5_path, tflite_path):
    """比較 H5 和 TFLite 模型的檔案大小"""
    import os
    
    h5_size = os.path.getsize(h5_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    
    print(f"\n模型檔案大小比較:")
    print(f"  H5 模型: {h5_size:.2f} MB")
    print(f"  TFLite 模型: {tflite_size:.2f} MB")
    print(f"  壓縮比: {h5_size/tflite_size:.1f}x")

def diagnose_tflite_conversion(keras_model, tflite_path, test_x, test_labels):
    """
    診斷 TFLite 轉換問題的詳細工具
    
    參數:
        keras_model: 原始 Keras 模型
        tflite_path: TFLite 模型路徑
        test_x: 測試輸入
        test_labels: 測試標籤 (tuple of arrays)
    """
    print("=" * 60)
    print("TFLite 轉換問題診斷")
    print("=" * 60)
    
    # 比較模型權重
    keras_weights = keras_model.get_weights()
    print(f"Keras 模型權重數量: {len(keras_weights)}")
    
    # 載入 TFLite 模型
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 檢查量化情況
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n量化檢查:")
    print(f"  輸入量化: {input_details[0]['dtype']}")
    for i, output_detail in enumerate(output_details):
        print(f"  輸出 {i} 量化: {output_detail['dtype']}")
    
    # 逐層比較（如果可能）
    try:
        tensor_details = interpreter.get_tensor_details()
        print(f"\nTFLite 模型張量數量: {len(tensor_details)}")
        
        # 檢查是否有量化張量
        quantized_tensors = [t for t in tensor_details if t['dtype'] != np.float32]
        if quantized_tensors:
            print(f"警告: 發現 {len(quantized_tensors)} 個量化張量")
            for tensor in quantized_tensors[:5]:  # 只顯示前5個
                print(f"  - {tensor['name']}: {tensor['dtype']}")
        else:
            print("✅ 所有張量都是 float32 類型")
            
    except Exception as e:
        print(f"張量詳細資訊獲取失敗: {e}")
    
    # 性能測試
    import time
    
    # Keras 模型推論時間
    start_time = time.time()
    keras_pred = keras_model.predict(test_x[:10], verbose=0)
    keras_time = time.time() - start_time
    
    # TFLite 模型推論時間
    start_time = time.time()
    for i in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_x[i:i+1].astype(np.float32))
        interpreter.invoke()
    tflite_time = time.time() - start_time
    
    print(f"\n性能比較 (10個樣本):")
    print(f"  Keras 推論時間: {keras_time:.4f}s")
    print(f"  TFLite 推論時間: {tflite_time:.4f}s")
    print(f"  速度提升: {keras_time/tflite_time:.2f}x")

if __name__ == "__main__":
    print("TFLite 驗證工具")
    print("請在模型訓練腳本中調用相關函數進行驗證")
