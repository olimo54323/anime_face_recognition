#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the anime face recognition model.
This script evaluates a trained model and generates performance metrics.
"""

import os
import sys
import numpy as np
import cv2
import random
import pandas as pd
import pickle
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import glob

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define default paths
DEFAULT_DATASET_PATH = 'dataset'
DEFAULT_MODEL_PATH = 'models/anime_face_recognition_model.h5'
DEFAULT_CLASS_NAMES_PATH = 'models/class_names.pkl'
DEFAULT_RESULTS_PATH = 'models/evaluation_results'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate anime face recognition model')
    
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default=DEFAULT_DATASET_PATH,
        help=f'Path to the dataset (default: {DEFAULT_DATASET_PATH})'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help=f'Path to the trained model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--class_names_path', 
        type=str, 
        default=DEFAULT_CLASS_NAMES_PATH,
        help=f'Path to class names file (default: {DEFAULT_CLASS_NAMES_PATH})'
    )
    
    parser.add_argument(
        '--results_path', 
        type=str, 
        default=DEFAULT_RESULTS_PATH,
        help=f'Path to save evaluation results (default: {DEFAULT_RESULTS_PATH})'
    )
    
    parser.add_argument(
        '--image_size', 
        type=int, 
        default=128,
        help='Image size (width and height) for model input (default: 128)'
    )
    
    parser.add_argument(
        '--test_split', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--samples_per_class', 
        type=int, 
        default=None,
        help='Maximum number of samples per class for testing (default: None, use all)'
    )
    
    parser.add_argument(
        '--confusion_matrix', 
        action='store_true',
        help='Generate confusion matrix visualization'
    )
    
    parser.add_argument(
        '--max_classes_vis', 
        type=int, 
        default=20,
        help='Maximum number of classes to show in visualizations (default: 20)'
    )
    
    parser.add_argument(
        '--analyze_failures', 
        action='store_true',
        help='Analyze and visualize failure cases'
    )
    
    return parser.parse_args()

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess an image for model input.
    
    Args:
        image: Input image (can be path or numpy array)
        target_size: Size to resize the image to (height, width)
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image if path is provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        img = image.copy()
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to range [0, 1]
    img = img / 255.0
    
    return img

def load_model_and_classes(model_path, class_names_path):
    """
    Load the trained model and class names.
    
    Args:
        model_path: Path to the model file
        class_names_path: Path to the class names file
        
    Returns:
        model, class_names
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found at {class_names_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Load class names
    with open(class_names_path, 'rb') as f:
        class_names = pickle.load(f)
    
    print(f"Model loaded successfully")
    print(f"Number of classes: {len(class_names)}")
    
    return model, class_names

def load_test_dataset(dataset_path, class_names, test_split=0.2, samples_per_class=None, target_size=(128, 128)):
    """
    Load a test dataset from the anime face dataset.
    
    Args:
        dataset_path: Path to the dataset
        class_names: List of character names
        test_split: Fraction of images to use for testing
        samples_per_class: Optional limit on samples per class
        target_size: Size to resize images to
        
    Returns:
        X_test, y_test
    """
    X_test = []
    y_test = []
    class_counts = {}
    
    print(f"Loading test dataset...")
    
    for i, character in enumerate(class_names):
        char_path = os.path.join(dataset_path, character)
        
        # Skip if character folder doesn't exist
        if not os.path.exists(char_path):
            print(f"Warning: Character folder '{character}' not found, skipping")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(char_path, ext)))
            image_files.extend(glob.glob(os.path.join(char_path, ext.upper())))
        
        # Skip if no images found
        if not image_files:
            print(f"Warning: No images found for '{character}', skipping")
            continue
        
        # Determine which images to use for testing
        random.shuffle(image_files)
        test_count = max(1, int(len(image_files) * test_split))
        
        # Limit samples per class if specified
        if samples_per_class is not None:
            test_count = min(test_count, samples_per_class)
        
        test_images = image_files[:test_count]
        class_counts[character] = len(test_images)
        
        print(f"Processing {character}: {len(test_images)} test images")
        
        # Load and preprocess test images
        for image_path in test_images:
            try:
                if not os.path.exists(image_path):
                    continue
                
                # Preprocess image
                image = preprocess_image(image_path, target_size)
                X_test.append(image)
                y_test.append(i)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    # Convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Test dataset loaded: {len(X_test)} images across {len(class_counts)} characters")
    return X_test, y_test, class_counts

def calculate_metrics(model, X_test, y_test, class_names):
    """
    Calculate evaluation metrics.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        class_names: List of class names
        
    Returns:
        metrics, y_pred, y_pred_probs
    """
    # Make predictions
    start_time = time.time()
    y_pred_probs = model.predict(X_test)
    end_time = time.time()
    
    # Convert to class indices
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate inference time
    total_time = end_time - start_time
    avg_inference_time = total_time / len(X_test)
    
    # Calculate top-K accuracy
    top3_correct = 0
    top5_correct = 0
    
    for i, probs in enumerate(y_pred_probs):
        top_indices = np.argsort(probs)[::-1]
        if y_test[i] in top_indices[:3]:
            top3_correct += 1
        if y_test[i] in top_indices[:5]:
            top5_correct += 1
    
    top3_accuracy = top3_correct / len(y_test)
    top5_accuracy = top5_correct / len(y_test)
    
    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total prediction time: {total_time:.2f} seconds")
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms per image")
    print(f"Images per second: {1/avg_inference_time:.2f}")
    
    # Get per-class metrics
    class_report = classification_report(y_test, y_pred, 
                                         target_names=class_names, 
                                         output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate additional metrics
    metrics = {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_accuracy),
        'top5_accuracy': float(top5_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'avg_inference_time_ms': float(avg_inference_time * 1000),
        'images_per_second': float(1/avg_inference_time),
        'class_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, y_pred, y_pred_probs

def plot_confusion_matrix(y_test, y_pred, class_names, results_path, max_classes=20):
    """
    Plot confusion matrix.
    
    Args:
        y_test, y_pred: True and predicted labels
        class_names: List of class names
        results_path: Path to save results
        max_classes: Maximum number of classes to show
        
    Returns:
        Path to saved figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # If too many classes, select a subset for visualization
    if len(class_names) > max_classes:
        # Find classes with most samples
        class_counts = {}
        for i in range(len(class_names)):
            class_counts[i] = np.sum(y_test == i)
        
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:max_classes]
        top_indices = [idx for idx, _ in top_classes]
        
        # Filter test and pred to only include top classes
        mask = np.isin(y_test, top_indices)
        filtered_y_test = y_test[mask]
        filtered_y_pred = y_pred[mask]
        
        # Create index mapping for contiguous indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(top_indices)}
        
        # Map indices
        new_y_test = np.array([index_map[idx] for idx in filtered_y_test])
        new_y_pred = np.array([index_map[idx] for idx in filtered_y_pred])
        
        # Recalculate confusion matrix with subset
        cm = confusion_matrix(new_y_test, new_y_pred)
        
        # Get class names for display
        display_class_names = [class_names[idx] for idx in top_indices]
    else:
        display_class_names = class_names
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=display_class_names,
        yticklabels=display_class_names
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(results_path, exist_ok=True)
    cm_path = os.path.join(results_path, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {cm_path}")
    return cm_path

def analyze_class_performance(metrics, class_names, results_path):
    """
    Analyze and visualize per-class performance.
    
    Args:
        metrics: Evaluation metrics
        class_names: List of class names
        results_path: Path to save results
        
    Returns:
        Dataframe with per-class metrics
    """
    # Extract per-class metrics from classification report
    class_report = metrics['class_report']
    
    # Create dataframe for visualization
    class_metrics = []
    for i, class_name in enumerate(class_names):
        if class_name in class_report:
            class_metrics.append({
                'class': class_name,
                'precision': class_report[class_name]['precision'],
                'recall': class_report[class_name]['recall'],
                'f1-score': class_report[class_name]['f1-score'],
                'support': class_report[class_name]['support']
            })
    
    df = pd.DataFrame(class_metrics)
    
    # Sort by F1 score
    df = df.sort_values('f1-score', ascending=False)
    
    # Plot top and bottom performers
    plt.figure(figsize=(15, 8))
    
    # Top performers
    plt.subplot(1, 2, 1)
    top_df = df.head(10)
    sns.barplot(x='f1-score', y='class', data=top_df, palette='viridis')
    plt.title('Top 10 Character Classes by F1 Score')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Bottom performers
    plt.subplot(1, 2, 2)
    bottom_df = df.tail(10).sort_values('f1-score')
    sns.barplot(x='f1-score', y='class', data=bottom_df, palette='viridis')
    plt.title('Bottom 10 Character Classes by F1 Score')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(results_path, exist_ok=True)
    perf_path = os.path.join(results_path, 'class_performance.png')
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class performance visualization saved to {perf_path}")
    
    # Save metrics to CSV
    csv_path = os.path.join(results_path, 'class_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Class metrics saved to {csv_path}")
    
    return df

def analyze_failure_cases(X_test, y_test, y_pred, y_pred_probs, class_names, results_path, num_samples=5):
    """
    Analyze and visualize misclassified examples.
    
    Args:
        X_test, y_test: Test data
        y_pred, y_pred_probs: Predictions
        class_names: List of class names
        results_path: Path to save results
        num_samples: Number of samples to visualize
    """
    # Find misclassified examples
    misclassified = np.where(y_test != y_pred)[0]
    
    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return
    
    # Select random samples from misclassified
    samples = random.sample(list(misclassified), min(num_samples, len(misclassified)))
    
    # Visualize
    plt.figure(figsize=(15, 4*len(samples)))
    
    for i, idx in enumerate(samples):
        # Get image and predictions
        img = X_test[idx]
        true_class = y_test[idx]
        pred_class = y_pred[idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(y_pred_probs[idx])[-3:][::-1]
        top_probs = y_pred_probs[idx][top_indices]
        
        # Display image
        plt.subplot(len(samples), 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}")
        plt.axis('off')
        
        # Display bar chart of top predictions
        plt.subplot(len(samples), 2, 2*i+2)
        bars = plt.barh([class_names[idx] for idx in top_indices], top_probs)
        
        # Mark true class in the predictions
        for j, idx in enumerate(top_indices):
            if idx == true_class:
                bars[j].set_color('green')
            else:
                bars[j].set_color('red')
        
        plt.xlim(0, 1)
        plt.title('Top 3 Predictions')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(results_path, exist_ok=True)
    fail_path = os.path.join(results_path, 'failure_cases.png')
    plt.savefig(fail_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Failure cases visualization saved to {fail_path}")

def analyze_prediction_confidence(y_test, y_pred, y_pred_probs, results_path):
    """
    Analyze prediction confidence distributions.
    
    Args:
        y_test, y_pred: True and predicted labels
        y_pred_probs: Prediction probabilities
        results_path: Path to save results
    """
    # Get confidence of predicted class for each sample
    confidences = [prob[pred] for prob, pred in zip(y_pred_probs, y_pred)]
    
    # Separate confidences for correct and incorrect predictions
    correct_mask = (y_test == y_pred)
    correct_confidences = [conf for conf, is_correct in zip(confidences, correct_mask) if is_correct]
    incorrect_confidences = [conf for conf, is_correct in zip(confidences, correct_mask) if not is_correct]
    
    # Plot distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct')
    plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot ROC-like curve for confidence thresholds
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0, 1, 100)
    accuracy_at_threshold = []
    retention_at_threshold = []
    
    for threshold in thresholds:
        # Count samples above threshold
        above_threshold = [conf >= threshold for conf in confidences]
        if sum(above_threshold) == 0:
            accuracy_at_threshold.append(0)
        else:
            # Calculate accuracy for samples above threshold
            correct_at_threshold = sum([is_correct and above for is_correct, above 
                                      in zip(correct_mask, above_threshold)])
            accuracy_at_threshold.append(correct_at_threshold / sum(above_threshold))
        
        # Calculate retention (percentage of samples kept)
        retention_at_threshold.append(sum(above_threshold) / len(confidences))
    
    plt.plot(retention_at_threshold, accuracy_at_threshold)
    plt.title('Accuracy vs. Retention Rate at Different Confidence Thresholds')
    plt.xlabel('Retention Rate')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(results_path, exist_ok=True)
    conf_path = os.path.join(results_path, 'confidence_analysis.png')
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confidence analysis saved to {conf_path}")
    
    # Print statistics
    print(f"Mean confidence for correct predictions: {np.mean(correct_confidences):.4f}")
    print(f"Mean confidence for incorrect predictions: {np.mean(incorrect_confidences):.4f}")
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0
    best_accuracy = 0
    best_retention = 0
    
    for threshold, acc, ret in zip(thresholds, accuracy_at_threshold, retention_at_threshold):
        if ret > 0:
            f1 = 2 * acc * ret / (acc + ret) if (acc + ret) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_accuracy = acc
                best_retention = ret
    
    print(f"\nOptimal confidence threshold: {best_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Retention rate: {best_retention:.4f} (keeping {int(best_retention * len(confidences))} samples)")
    
    # Return confidence analysis results
    return {
        'thresholds': thresholds.tolist(),
        'accuracy': accuracy_at_threshold,
        'retention': retention_at_threshold,
        'optimal_threshold': best_threshold,
        'optimal_accuracy': best_accuracy,
        'optimal_retention': best_retention,
        'mean_confidence_correct': float(np.mean(correct_confidences)),
        'mean_confidence_incorrect': float(np.mean(incorrect_confidences))
    }

def generate_summary_report(metrics, results_path, args):
    """
    Generate a summary report of evaluation results.
    
    Args:
        metrics: Evaluation metrics
        results_path: Path to save results
        args: Command-line arguments
    """
    summary = {
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'image_size': args.image_size,
        'metrics': {
            'accuracy': metrics['accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'avg_inference_time_ms': metrics['avg_inference_time_ms'],
            'images_per_second': metrics['images_per_second']
        }
    }
    
    # Save summary to JSON
    os.makedirs(results_path, exist_ok=True)
    summary_path = os.path.join(results_path, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary report saved to {summary_path}")
    
    # Create a text report
    report = [
        "===== ANIME FACE RECOGNITION MODEL EVALUATION =====",
        f"Date: {summary['date']}",
        f"Model: {summary['model_path']}",
        f"Dataset: {summary['dataset_path']}",
        f"Image size: {summary['image_size']}x{summary['image_size']}",
        "",
        "PERFORMANCE METRICS:",
        f"  Accuracy: {summary['metrics']['accuracy']:.4f}",
        f"  Top-3 Accuracy: {summary['metrics']['top3_accuracy']:.4f}",
        f"  Top-5 Accuracy: {summary['metrics']['top5_accuracy']:.4f}",
        f"  Precision: {summary['metrics']['precision']:.4f}",
        f"  Recall: {summary['metrics']['recall']:.4f}",
        f"  F1 Score: {summary['metrics']['f1_score']:.4f}",
        "",
        "SPEED METRICS:",
        f"  Average inference time: {summary['metrics']['avg_inference_time_ms']:.2f} ms per image",
        f"  Images per second: {summary['metrics']['images_per_second']:.2f}",
        "",
        "RECOMMENDATIONS:",
    ]
    
    # Add recommendations based on metrics
    if metrics['accuracy'] < 0.7:
        report.append("  • Consider collecting more training data, especially for poorly performing classes")
        report.append("  • Try using more aggressive data augmentation")
        report.append("  • Experiment with fine-tuning more layers of the base model")
    
    if metrics['top3_accuracy'] - metrics['accuracy'] > 0.2:
        report.append("  • The model has good recall in top-3, consider implementing a confidence threshold")
        report.append("  • For user experience, show top-3 predictions instead of just the top one")
    
    if metrics['avg_inference_time_ms'] > 50:
        report.append("  • Consider model optimization techniques like quantization for faster inference")
    
    # Save text report
    report_path = os.path.join(results_path, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Evaluation report saved to {report_path}")
    
    return summary

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load model and class names
    model, class_names = load_model_and_classes(args.model_path, args.class_names_path)
    
    # Load test dataset
    X_test, y_test, class_counts = load_test_dataset(
        args.dataset_path,
        class_names,
        test_split=args.test_split,
        samples_per_class=args.samples_per_class,
        target_size=(args.image_size, args.image_size)
    )
    
    # Calculate metrics
    metrics, y_pred, y_pred_probs = calculate_metrics(model, X_test, y_test, class_names)
    
    # Create results directory
    os.makedirs(args.results_path, exist_ok=True)
    
    # Save all metrics to JSON
    metrics_path = os.path.join(args.results_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=4)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Plot confusion matrix if requested
    if args.confusion_matrix:
        plot_confusion_matrix(y_test, y_pred, class_names, args.results_path, args.max_classes_vis)
    
    # Analyze class performance
    class_metrics_df = analyze_class_performance(metrics, class_names, args.results_path)
    
    # Analyze failure cases if requested
    if args.analyze_failures:
        analyze_failure_cases(X_test, y_test, y_pred, y_pred_probs, class_names, args.results_path)
    
    # Analyze prediction confidence
    confidence_analysis = analyze_prediction_confidence(y_test, y_pred, y_pred_probs, args.results_path)
    
    # Save confidence analysis
    conf_analysis_path = os.path.join(args.results_path, 'confidence_analysis.json')
    with open(conf_analysis_path, 'w') as f:
        json.dump(confidence_analysis, f, indent=4)
    
    # Generate summary report
    summary = generate_summary_report(metrics, args.results_path, args)
    
    print("\nModel evaluation completed!")
    print(f"All results saved to {args.results_path}")

if __name__ == "__main__":
    main()