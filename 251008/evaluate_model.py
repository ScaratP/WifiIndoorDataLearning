import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 設定 matplotlib 支援中文顯示
import matplotlib
from matplotlib.font_manager import fontManager

# 檢查可用字型
def get_available_chinese_font():
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DFKai-SB','Arial Unicode MS', 'Noto Sans CJK TC', 'Noto Sans TC','Noto Sans CJK JP', 'Noto Sans CJK SC']
    
    available_fonts = [font.name for font in fontManager.ttflist]
    print("Available fonts:", available_fonts[:10], f"...and {len(available_fonts)-10} more")
    
    # 尋找可用的中文字型
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"Using Chinese font: {font}")
            return font
    
    print("No preferred Chinese fonts found, using default")
    return None

# 設定中文字型
chinese_font = get_available_chinese_font()
if chinese_font:
    matplotlib.rcParams['font.family'] = chinese_font
else:
    # 備用方案：使用支援 Unicode 的字型，並確保 Matplotlib 可以顯示中文
    print("Using fallback font configuration for Chinese support")
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用不需要特定字型的方法
    if hasattr(plt, 'style') and 'font.sans-serif' in plt.rcParams:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

# Import AttentionLayer class
from original_hadnn import AttentionLayer

def load_model_and_data(model_path, hadnn_data_dir):
    """Load model and test data"""
    # Load model and register custom layer
    print(f"Loading model: {model_path}")
    
    # Use custom object scope to solve the unknown layer AttentionLayer problem
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading error: {e}")
            raise
    
    # Load test data
    print(f"Loading test data: {hadnn_data_dir}")
    test_x = np.load(os.path.join(hadnn_data_dir, 'test_x.npy'))
    test_b = np.load(os.path.join(hadnn_data_dir, 'test_b.npy'))
    test_c = np.load(os.path.join(hadnn_data_dir, 'test_c.npy'))
    test_y = np.load(os.path.join(hadnn_data_dir, 'test_y.npy'))
    
    # Ensure labels have correct shape
    test_b = test_b.reshape(-1)  # Building labels
    
    # Fix floor label shape issues
    if test_c.shape[0] != test_x.shape[0]:
        print(f"Warning: Floor label shape mismatch, trying to fix. test_c shape: {test_c.shape}, test_x shape: {test_x.shape}")
        if len(test_c.shape) > 1 and test_c.shape[0] == test_x.shape[0]:
            # If test_c is position coordinate, not floor label, try to load correct floor label
            try:
                print("Trying to load floor labels 'test_f.npy'...")
                test_f_path = os.path.join(hadnn_data_dir, 'test_f.npy')
                if os.path.exists(test_f_path):
                    test_f = np.load(test_f_path)
                    test_f = test_f.reshape(-1)
                    print(f"Successfully loaded floor labels, shape: {test_f.shape}")
                    # Update test_c to floor labels
                    test_c = test_f
                else:
                    # If floor label file not found, try to get from test_y
                    print("Floor label file not found, trying to get from test_y...")
                    test_c = np.zeros(test_x.shape[0], dtype=int)
                    print(f"Created default floor labels, shape: {test_c.shape}")
            except Exception as e:
                print(f"Failed to load floor labels: {e}")
        elif test_c.shape[0] == test_x.shape[0] * 2:
            # Data might be duplicated, take the first half
            test_c = test_c[:test_x.shape[0]]
            print(f"Fixed floor label shape to: {test_c.shape}")
    
    if len(test_c.shape) > 1:
        # If test_c is still multi-dimensional (possibly coordinates rather than floor labels), create default floor labels
        print("Warning: test_c appears to be coordinates rather than floor labels, creating default floor labels")
        test_c = np.zeros(test_x.shape[0], dtype=int)
    else:
        test_c = test_c.reshape(-1)  # Ensure floor labels are one-dimensional
    
    # Ensure location data is two-dimensional
    if len(test_y.shape) == 1:
        print("Location data is one-dimensional, trying to reshape to 2D coordinates")
        # If test_y is one-dimensional but length matches sample count, could be single coordinate values
        if test_y.shape[0] == test_x.shape[0]:
            # If there's only one coordinate value (e.g., only x-coordinates), expand to 2D
            test_y = np.column_stack((test_y, np.zeros_like(test_y)))
            print(f"Expanded location data to 2D coordinates: {test_y.shape}")
        else:
            # Check if can be reshaped to (n, 2) shape
            if test_y.shape[0] % 2 == 0 and test_y.shape[0] // 2 == test_x.shape[0]:
                # x and y coordinates might be stored sequentially
                half_len = test_y.shape[0] // 2
                test_y = np.column_stack((test_y[:half_len], test_y[half_len:]))
                print(f"Reshaped location data to: {test_y.shape}")
            else:
                print("Warning: Cannot reshape location data to 2D coordinates, using zero matrix instead")
                test_y = np.zeros((test_x.shape[0], 2))
    
    # Final confirmation that all data dimensions match
    if test_c.shape[0] != test_x.shape[0] or test_b.shape[0] != test_x.shape[0] or test_y.shape[0] != test_x.shape[0]:
        print("Warning: Data shapes don't match, making final adjustments")
        min_samples = min(test_x.shape[0], test_b.shape[0], test_c.shape[0], test_y.shape[0])
        test_x = test_x[:min_samples]
        test_b = test_b[:min_samples]
        test_c = test_c[:min_samples]
        test_y = test_y[:min_samples]
        print(f"Adjusted data shapes - X: {test_x.shape}, Building: {test_b.shape}, Floor: {test_c.shape}, Position: {test_y.shape}")
    
    # Load label mappings (if available)
    label_mapping_path = os.path.join(hadnn_data_dir, 'dataset_config.json')
    print(f"Loading label mappings: {label_mapping_path}")
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            building_mapping = config.get('building_mapping', {})
            floor_mapping = config.get('floor_mapping', {})
            print(f"Found {len(building_mapping)} building label mappings and {len(floor_mapping)} floor label mappings")
    else:
        print("Label mapping file not found")
        building_mapping = {}
        floor_mapping = {}
    
    # Try to load test point names (if available)
    test_names_path = os.path.join(hadnn_data_dir, 'test_names.npy')
    if os.path.exists(test_names_path):
        test_names = np.load(test_names_path, allow_pickle=True)
        print(f"Loaded {len(test_names)} test sample names")
    else:
        print("Test sample names not found")
        test_names = None
    
    # Output data shapes
    print(f"Test data shapes - X: {test_x.shape}, Building: {test_b.shape}, Floor: {test_c.shape}, Position: {test_y.shape}")
    
    return model, test_x, test_b, test_c, test_y, building_mapping, floor_mapping, test_names

def evaluate_and_visualize(model, test_x, test_b, test_c, test_y, building_mapping, floor_mapping, output_dir):
    """Evaluate model and visualize results"""
    # Get model predictions
    predictions = model.predict(test_x)
    
    # Check prediction results structure
    if len(predictions) < 2:
        print("Warning: Model output less than expected, might not be HADNN model format")
        # If there's only one output, assume it's position prediction
        position_preds = predictions[0] if isinstance(predictions, list) else predictions
        # Create virtual classification outputs
        building_preds = np.zeros(len(test_b), dtype=int)
        floor_preds = np.zeros(len(test_c), dtype=int)
    else:
        # Normally parse three outputs of HADNN model
        if isinstance(predictions, list) and len(predictions) >= 3:
            building_preds = np.argmax(predictions[0], axis=1)
            floor_preds = np.argmax(predictions[1], axis=1)
            position_preds = predictions[2]
        else:
            # Handle non-standard output structure
            print(f"Non-standard model output structure, output type: {type(predictions)}")
            if isinstance(predictions, list):
                print(f"Output count: {len(predictions)}")
                # Assume first output is building, second is floor
                building_preds = np.argmax(predictions[0], axis=1) if len(predictions) > 0 else np.zeros(len(test_b), dtype=int)
                floor_preds = np.argmax(predictions[1], axis=1) if len(predictions) > 1 else np.zeros(len(test_c), dtype=int)
                position_preds = predictions[2] if len(predictions) > 2 else np.zeros((len(test_x), 2))
            else:
                # Completely unable to parse, use default values
                building_preds = np.zeros(len(test_b), dtype=int)
                floor_preds = np.zeros(len(test_c), dtype=int)
                position_preds = np.zeros((len(test_x), 2))
    
    # Ensure prediction results and test labels have the same sample count and correct data types
    if len(floor_preds) != len(test_c):
        print(f"Warning: Floor prediction results and test labels have mismatched sample count, adjusting...")
        min_samples = min(len(floor_preds), len(test_c))
        floor_preds = floor_preds[:min_samples]
        test_c = test_c[:min_samples]
        print(f"After adjustment: Floor prediction shape: {floor_preds.shape}, Test label shape: {test_c.shape}")
    
    # Calculate position errors
    position_errors = np.sqrt(np.sum((test_y - position_preds) ** 2, axis=1))
    
    # Ensure labels are integer type, otherwise confusion matrix calculation will fail
    test_b = test_b.astype(int)
    test_c = test_c.astype(int)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Building classification evaluation
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(test_b, building_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Building Classification Confusion Matrix')
    plt.xlabel('Predicted Building')
    plt.ylabel('Actual Building')
    
    # If there are building label mappings, add them to the axes
    if building_mapping:
        rev_mapping = {v: k for k, v in building_mapping.items()}
        building_labels = [rev_mapping.get(i, str(i)) for i in range(len(cm))]
        plt.xticks(np.arange(len(building_labels)) + 0.5, building_labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(building_labels)) + 0.5, building_labels)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'building_confusion_matrix.png'))
    plt.close()
    
    # 2. Floor classification evaluation
    plt.figure(figsize=(12, 10))
    try:
        cm = confusion_matrix(test_c, floor_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Floor Classification Confusion Matrix')
        plt.xlabel('Predicted Floor')
        plt.ylabel('Actual Floor')
        
        # If there are floor label mappings, add them to the axes
        if floor_mapping:
            rev_mapping = {v: k for k, v in floor_mapping.items()}
            floor_labels = [rev_mapping.get(i, str(i)) for i in range(len(cm))]
            plt.xticks(np.arange(len(floor_labels)) + 0.5, floor_labels, rotation=45, ha='right')
            plt.yticks(np.arange(len(floor_labels)) + 0.5, floor_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'floor_confusion_matrix.png'))
    except Exception as e:
        print(f"Error generating floor confusion matrix: {e}")
        # Create a blank figure and save
        plt.clf()
        plt.title('Floor Classification Confusion Matrix (Failed to Generate)')
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'floor_confusion_matrix.png'))
    plt.close()
    
    # 3. Position error analysis
    plt.figure(figsize=(10, 6))
    plt.hist(position_errors, bins=20, alpha=0.7)
    plt.axvline(np.mean(position_errors), color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Position Prediction Error Distribution (Mean: {np.mean(position_errors):.2f})')
    plt.xlabel('Euclidean Distance Error')
    plt.ylabel('Sample Count')
    plt.savefig(os.path.join(output_dir, 'position_error_distribution.png'))
    plt.close()
    
    # 4. Error vs building relationship
    plt.figure(figsize=(12, 6))
    building_error = {}
    for i in range(len(test_b)):
        b = test_b[i]
        if b not in building_error:
            building_error[b] = []
        building_error[b].append(position_errors[i])
    
    buildings = sorted(building_error.keys())
    error_means = [np.mean(building_error[b]) for b in buildings]
    error_stds = [np.std(building_error[b]) for b in buildings]
    
    plt.bar(range(len(buildings)), error_means, yerr=error_stds, alpha=0.7)
    plt.title('Average Position Error by Building')
    plt.xlabel('Building')
    plt.ylabel('Average Error')
    
    # If there are building label mappings, add them to the axes
    if building_mapping:
        rev_mapping = {v: k for k, v in building_mapping.items()}
        building_labels = [rev_mapping.get(b, str(b)) for b in buildings]
        plt.xticks(range(len(buildings)), building_labels, rotation=45, ha='right')
    else:
        plt.xticks(range(len(buildings)), buildings)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'building_position_errors.png'))
    plt.close()
    
    # 5. Generate detailed evaluation report
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("=== HADNN Model Evaluation Report ===\n\n")
        
        # Building classification report
        f.write("Building Classification Report:\n")
        try:
            f.write(classification_report(test_b, building_preds))
            f.write("\nBuilding Classification Accuracy: {:.2%}\n\n".format(np.mean(test_b == building_preds)))
        except Exception as e:
            f.write(f"Failed to generate building classification report: {e}\n\n")
        
        # Floor classification report
        f.write("Floor Classification Report:\n")
        try:
            f.write(classification_report(test_c, floor_preds))
            f.write("\nFloor Classification Accuracy: {:.2%}\n\n".format(np.mean(test_c == floor_preds)))
        except Exception as e:
            f.write(f"Failed to generate floor classification report: {e}\n\n")
        
        # Position prediction report
        f.write("Position Prediction Report:\n")
        f.write("Mean Error (Euclidean Distance): {:.4f}\n".format(np.mean(position_errors)))
        f.write("Error Standard Deviation: {:.4f}\n".format(np.std(position_errors)))
        f.write("Maximum Error: {:.4f}\n".format(np.max(position_errors)))
        f.write("Minimum Error: {:.4f}\n".format(np.min(position_errors)))
        f.write("Median Error: {:.4f}\n".format(np.median(position_errors)))
        
        # Position error after correct building and floor classification
        correct_building_floor = (test_b == building_preds) & (test_c == floor_preds)
        if np.any(correct_building_floor):
            correct_errors = position_errors[correct_building_floor]
            f.write("\nPosition Error after Correct Building and Floor Classification:\n")
            f.write("Sample Count: {}\n".format(len(correct_errors)))
            f.write("Average Error: {:.4f}\n".format(np.mean(correct_errors)))
        
        # Combined score (weighted average)
        f.write("\nCombined Model Score:\n")
        building_score = np.mean(test_b == building_preds) * 100
        floor_score = np.mean(test_c == floor_preds) * 100
        position_score = np.mean(position_errors)
        final_score = position_score / ((building_score * floor_score) / 10000)
        f.write("Final Score (lower is better): {:.4f}\n".format(final_score))
    
    # Return evaluation metrics
    return {
        'building_accuracy': np.mean(test_b == building_preds),
        'floor_accuracy': np.mean(test_c == floor_preds),
        'position_mse': np.mean(position_errors),
        'final_score': final_score
    }

def analyze_misclassified_samples(model, test_x, test_b, test_c, test_names, building_mapping, floor_mapping, output_dir):
    """Detailed analysis of misclassified samples"""
    # Get model predictions
    predictions = model.predict(test_x)
    building_preds = np.argmax(predictions[0], axis=1)
    floor_preds = np.argmax(predictions[1], axis=1)
    
    # Find misclassified building samples
    building_errors = np.where(test_b != building_preds)[0]
    building_error_report = []
    
    rev_building_mapping = {v: k for k, v in building_mapping.items()} if building_mapping else {}
    rev_floor_mapping = {v: k for k, v in floor_mapping.items()} if floor_mapping else {}
    
    for idx in building_errors:
        true_b = test_b[idx]
        pred_b = building_preds[idx]
        true_b_name = rev_building_mapping.get(true_b, str(true_b))
        pred_b_name = rev_building_mapping.get(pred_b, str(pred_b))
        
        sample_name = test_names[idx] if test_names is not None else f"Sample {idx}"
        
        # Get prediction probability distribution for this sample
        building_probs = predictions[0][idx]
        top_buildings = np.argsort(building_probs)[::-1][:3]  # Top 3 most likely buildings
        prob_info = []
        for b in top_buildings:
            b_name = rev_building_mapping.get(b, str(b))
            prob_info.append(f"{b_name}: {building_probs[b]:.4f}")
        
        error_info = {
            'sample': sample_name,
            'true_building': true_b_name,
            'pred_building': pred_b_name,
            'top_probs': prob_info,
            'confidence': building_probs[pred_b],
            'correct_prob': building_probs[true_b]
        }
        building_error_report.append(error_info)
    
    # Write error report to file
    with open(os.path.join(output_dir, 'building_error_analysis.txt'), 'w') as f:
        f.write(f"=== Building Classification Error Analysis ({len(building_errors)}/{len(test_b)} Samples) ===\n\n")
        
        for error in building_error_report:
            f.write(f"Sample: {error['sample']}\n")
            f.write(f"True Building: {error['true_building']}, Predicted Building: {error['pred_building']}\n")
            f.write(f"Prediction Confidence: {error['confidence']:.4f}, Correct Building Probability: {error['correct_prob']:.4f}\n")
            f.write(f"Top 3 Likely Buildings: {', '.join(error['top_probs'])}\n")
            f.write("\n")
        
        # Add some overall analysis
        f.write("=== Overall Analysis ===\n")
        f.write(f"Error Rate: {len(building_errors) / len(test_b):.2%}\n")
        avg_confidence = np.mean([e['confidence'] for e in building_error_report])
        f.write(f"Average Error Confidence: {avg_confidence:.4f}\n")
    
    return building_error_report

if __name__ == "__main__":
    print("=== Starting Model Evaluation ===")
    
    # Configure paths
    hadnn_data_dir = "./hadnn_data"  # Ensure this is the correct path
    model_dir = "./enhanced_models"  # Ensure this is the correct path
    model_path = os.path.join(model_dir, 'enhanced_hadnn_model.h5')
    output_dir = os.path.join(model_dir, 'evaluation')
    
    # Check paths
    print(f"Data directory: {os.path.abspath(hadnn_data_dir)}")
    print(f"Model directory: {os.path.abspath(model_dir)}")
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Confirm file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
    else:
        # Load model and data
        model, test_x, test_b, test_c, test_y, building_mapping, floor_mapping, test_names = load_model_and_data(
            model_path, hadnn_data_dir
        )
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_and_visualize(
            model, test_x, test_b, test_c, test_y, building_mapping, floor_mapping, output_dir
        )
        
        # Analyze error samples
        if test_names is not None:
            print("Analyzing misclassified samples...")
            analyze_misclassified_samples(
                model, test_x, test_b, test_c, test_names, building_mapping, floor_mapping, output_dir
            )
        
        print(f"Evaluation complete. Results saved in: {output_dir}")
        print("\nModel Performance Summary:")
        print(f"Building Classification Accuracy: {metrics['building_accuracy']:.2%}")
        print(f"Floor Classification Accuracy: {metrics['floor_accuracy']:.2%}")
        print(f"Position Prediction MSE: {metrics['position_mse']:.4f}")
        print(f"Final Score: {metrics['final_score']:.4f}")
