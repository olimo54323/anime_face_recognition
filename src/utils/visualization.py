# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import random
import math
import os


def plot_training_history(history, figsize=(12, 5)):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: History object from model.fit()
        figsize: Size of the figure
        
    Returns:
        Figure with plots
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_predictions(model, X_test, y_test, class_names, num_images=5, figsize=(15, 10)):
    """
    Visualize model predictions on sample test images.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels for test images
        class_names: List of class names
        num_images: Number of images to display
        figsize: Size of the figure
        
    Returns:
        Figure with images and predictions
    """
    # Get random indices
    indices = random.sample(range(len(X_test)), min(num_images, len(X_test)))
    
    # Make predictions
    predictions = model.predict(X_test[indices])
    pred_classes = np.argmax(predictions, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    for i, idx in enumerate(indices):
        # Create subplot
        plt.subplot(1, num_images, i + 1)
        
        # Display image
        img = X_test[idx]
        if img.shape[-1] == 1:  # Convert grayscale to RGB for display
            img = np.repeat(img, 3, axis=-1)
        plt.imshow(img)
        
        # Get true and predicted class
        true_class = y_test[idx]
        pred_class = pred_classes[i]
        
        # Set title with true and predicted class
        title_color = 'green' if true_class == pred_class else 'red'
        plt.title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}", 
                 color=title_color)
        
        plt.axis('off')
    
    plt.tight_layout()
    return fig


def display_dataset_samples(dataset_path, class_names, samples_per_class=3, figsize=(15, 15)):
    """
    Display sample images from each class in the dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        class_names: List of class names
        samples_per_class: Number of samples to display per class
        figsize: Size of the figure
        
    Returns:
        Figure with sample images
    """
    # Calculate grid dimensions
    num_classes = len(class_names)
    grid_cols = samples_per_class
    grid_rows = min(10, num_classes)  # Limit to 10 rows max
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Counter for subplot position
    count = 1
    
    # Iterate through classes
    for i, character in enumerate(class_names[:grid_rows]):
        # Get path to character folder
        char_path = os.path.join(dataset_path, character)
        
        # Skip if folder doesn't exist
        if not os.path.exists(char_path):
            continue
        
        # Get all image files in the folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            image_files.extend(glob.glob(os.path.join(char_path, ext)))
            image_files.extend(glob.glob(os.path.join(char_path, ext.upper())))
        
        # Skip if no images found
        if not image_files:
            continue
        
        # Select random samples
        samples = random.sample(image_files, min(samples_per_class, len(image_files)))
        
        # Display each sample
        for sample in samples:
            plt.subplot(grid_rows, grid_cols, count)
            img = cv2.imread(sample)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(character)
            plt.axis('off')
            count += 1
    
    plt.tight_layout()
    return fig


def plot_class_distribution(dataset_path, class_names, figsize=(15, 8)):
    """
    Plot distribution of images across classes.
    
    Args:
        dataset_path: Path to the dataset directory
        class_names: List of class names
        figsize: Size of the figure
        
    Returns:
        Bar chart figure
    """
    # Count images per class
    class_counts = []
    
    for character in class_names:
        char_path = os.path.join(dataset_path, character)
        
        if not os.path.exists(char_path):
            class_counts.append(0)
            continue
        
        # Count all images
        count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            count += len(glob.glob(os.path.join(char_path, ext)))
            count += len(glob.glob(os.path.join(char_path, ext.upper())))
        
        class_counts.append(count)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get indices sorted by count (descending)
    sorted_indices = np.argsort(class_counts)[::-1]
    
    # Plot bar chart with top 30 classes
    top_n = 30
    sorted_indices = sorted_indices[:top_n]
    
    # Create bar chart
    plt.bar(
        range(len(sorted_indices)),
        [class_counts[i] for i in sorted_indices],
        color='skyblue'
    )
    
    plt.xlabel('Character')
    plt.ylabel('Number of Images')
    plt.title(f'Distribution of Images Across Top {top_n} Classes')
    
    plt.xticks(
        range(len(sorted_indices)),
        [class_names[i] for i in sorted_indices],
        rotation=90
    )
    
    plt.tight_layout()
    return plt.gcf()


def visualize_image_preprocessing(image_path, target_size=(128, 128)):
    """
    Visualize the stages of image preprocessing.
    
    Args:
        image_path: Path to the input image
        target_size: Size to resize the image to
        
    Returns:
        Figure with original, preprocessed, and intermediate images
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (BGR)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Resized image
    resized = cv2.resize(img, target_size)
    axes[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Resized to {target_size}')
    axes[1].axis('off')
    
    # Normalized image
    normalized = resized / 255.0
    axes[2].imshow(cv2.cvtColor(normalized.astype(np.float32), cv2.COLOR_BGR2RGB))
    axes[2].set_title('Normalized [0, 1]')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig