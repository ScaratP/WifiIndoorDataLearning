import os
import tensorflow as tf
import json
import numpy as np
from pathlib import Path

def load_model_with_custom_layers(model_path, custom_objects=None):
    """
    使用自定義層載入模型的通用函數
    
    Args:
        model_path: 模型檔案路徑
        custom_objects: 自定義層字典
    
    Returns:
        載入後的模型
    """
    if custom_objects is None:
        # 嘗試自動導入自定義層
        try:
            from custom_layers import get_custom_objects
            custom_objects = get_custom_objects()
        except ImportError:
            # 如果沒有 custom_layers 模組，使用空字典
            custom_objects = {}
    
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
        print(f"成功載入模型：{model_path}")
        return model
    except Exception as e:
        print(f"模型載入失敗：{str(e)}")
        raise

def save_model_summary(model, output_path):
    """將模型摘要保存到文件"""
    with open(output_path, 'w') as f:
        # 先保存模型的基本信息
        f.write("模型類別: " + model.__class__.__name__ + "\n")
        f.write("模型架構:\n")
        
        # 獲取模型摘要
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        
        # 獲取各層参數量
        f.write("\n層參數統計:\n")
        total_params = 0
        for layer in model.layers:
            params = layer.count_params()
            total_params += params
            f.write(f"{layer.name}: {params:,} 參數\n")
        
        f.write(f"\n總參數量: {total_params:,}\n")

def convert_model_to_tflite(model, output_path, quantize=False):
    """將 Keras 模型轉換為 TFLite 格式"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"已将模型轉換為 TFLite 格式並保存至: {output_path}")
    print(f"TFLite 模型大小: {Path(output_path).stat().st_size / 1024:.2f} KB")
    
    return output_path
