import os
import sys
import subprocess
import time

def run_preprocessing():
    """運行數據預處理"""
    print("=== 步驟 1: 數據預處理 ===")
    
    # 切換到預處理目錄
    preprocess_dir = "./preprocess"
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    # 運行預處理腳本
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join(preprocess_dir, "data_preprocess.py")
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("數據預處理完成")
            print(result.stdout)
        else:
            print("數據預處理失敗:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("數據預處理超時")
        return False
    except Exception as e:
        print(f"運行預處理時發生錯誤: {e}")
        return False
    
    return True

def run_model_training(model_script, model_name):
    """運行模型訓練"""
    print(f"=== 訓練 {model_name} 模型 ===")
    
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 
            model_script
        ], capture_output=True, text=True, timeout=1800)  # 30分鐘超時
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"{model_name} 模型訓練完成 (耗時: {training_time:.2f}秒)")
            print("訓練輸出:")
            print(result.stdout)
            return True
        else:
            print(f"{model_name} 模型訓練失敗:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"{model_name} 模型訓練超時")
        return False
    except Exception as e:
        print(f"訓練 {model_name} 時發生錯誤: {e}")
        return False

def check_data_exists():
    """檢查預處理數據是否存在"""
    hadnn_data_dir = "./hadnn_data"
    required_files = [
        'train_x.npy', 'train_b.npy', 'train_y.npy', 'train_c.npy',
        'test_x.npy', 'test_b.npy', 'test_y.npy', 'test_c.npy',
        'dataset_config.json'
    ]
    
    if not os.path.exists(hadnn_data_dir):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(hadnn_data_dir, file)):
            print(f"缺少數據文件: {file}")
            return False
    
    return True

def main():
    """主函數"""
    print("=== 開始完整的模型訓練流程 ===")
    
    # 檢查輸入文件
    json_file = "./se_all_13_filtered.json"
    if not os.path.exists(json_file):
        print(f"錯誤: 找不到輸入文件 {json_file}")
        return
    
    # 步驟1: 數據預處理
    if not check_data_exists():
        print("數據不存在，開始預處理...")
        if not run_preprocessing():
            print("數據預處理失敗，終止訓練")
            return
    else:
        print("數據已存在，跳過預處理步驟")
    
    # 創建模型輸出目錄
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # 步驟2: 訓練所有模型
    models_to_train = [
        ("mlp.py", "MLP"),
        ("original_hadnn.py", "Original HADNN"),
        ("hadnn_n_random_forest.py", "HADNN + Random Forest")
    ]
    
    successful_models = []
    failed_models = []
    
    for script, name in models_to_train:
        if os.path.exists(script):
            print(f"\n{'='*50}")
            success = run_model_training(script, name)
            if success:
                successful_models.append(name)
            else:
                failed_models.append(name)
        else:
            print(f"警告: 找不到模型腳本 {script}")
            failed_models.append(name)
    
    # 訓練結果總結
    print(f"\n{'='*50}")
    print("=== 訓練結果總結 ===")
    print(f"成功訓練的模型 ({len(successful_models)}):")
    for model in successful_models:
        print(f"  ✓ {model}")
    
    if failed_models:
        print(f"\n失敗的模型 ({len(failed_models)}):")
        for model in failed_models:
            print(f"  ✗ {model}")
    
    print(f"\n所有模型文件保存在: {models_dir}")
    print("訓練完成!")

if __name__ == "__main__":
    main()
