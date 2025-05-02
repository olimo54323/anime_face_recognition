# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import tensorflow as tf

def calculate_accuracy(model, X_test, y_test):
    """
    Calculate the accuracy of the model on test data.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        
    Returns:
        Accuracy score
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return accuracy_score(y_test, y_pred)


def calculate_precision_recall(model, X_test, y_test, average='weighted'):
    """
    Calculate precision and recall for the model on test data.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        average: Method to calculate average ('micro', 'macro', 'weighted')
        
    Returns:
        precision, recall
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    return precision, recall


def calculate_f1_score(model, X_test, y_test, average='weighted'):
    """
    Calculate F1 score for the model on test data.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        average: Method to calculate average ('micro', 'macro', 'weighted')
        
    Returns:
        F1 score
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return f1_score(y_test, y_pred, average=average, zero_division=0)


def plot_confusion_matrix(model, X_test, y_test, class_names, figsize=(12, 10)):
    """
    Generate and display confusion matrix.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        class_names: List of class names
        figsize: Size of the figure
        
    Returns:
        Confusion matrix figure
    """
    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Use a heatmap for better visualization
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt.gcf()


def print_classification_report(model, X_test, y_test, class_names):
    """
    Print detailed classification report.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        class_names: List of class names
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    return report


def evaluate_model(model, X_test, y_test, class_names, batch_size=32):
    """
    Perform a comprehensive evaluation of the model.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        class_names: List of class names
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Evaluate with Keras
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    
    # Calculate predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate other metrics
    precision, recall = calculate_precision_recall(model, X_test, y_test)
    f1 = calculate_f1_score(model, X_test, y_test)
    
    # Create report
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Return metrics
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report
    }