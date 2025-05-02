#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a model for anime face recognition.
This script trains a CNN model using the MobileNetV2 architecture
for recognizing anime character faces.
"""

import os
import sys
import numpy as np
import cv2
import random
import pandas as pd
import glob
import pickle
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define default paths
DEFAULT_DATASET_PATH = 'dataset'
DEFAULT_MODEL_PATH = 'models/anime_face_recognition_model.h5'
DEFAULT_CLASS_NAMES_PATH = 'models/class_names.pkl'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train anime face recognition model')
    
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
        help=f'Path to save the model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--class_names_path', 
        type=str, 
        default=DEFAULT_CLASS_NAMES_PATH,
        help=f'Path to save class names (default: {DEFAULT_CLASS_NAMES_PATH})'
    )
    
    parser.add_argument(
        '--min_images', 
        type=int, 
        default=15,
        help='Minimum number of images per character (default: 15)'
    )
    
    parser.add_argument(
        '--max_classes', 
        type=int, 
        default=50,
        help='Maximum number of character classes to include (default: 50, 0 for all)'
    )
    
    parser.add_argument(
        '--image_size', 
        type=int, 
        default=128,
        help='Image size (width and height) for model input (default: 128)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=30,
        help='Number of training epochs (default: 30)'
    )
    
    parser.add_argument(
        '--no_augmentation', 
        action='store_false',
        dest='use_augmentation',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--test_split', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.set_defaults(use_augmentation=True)
    
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

def find_character_folders(dataset_path, verbose=False):
    """
    Find character folders in the dataset.
    
    Args:
        dataset_path: Path to the dataset
        verbose: Whether to print verbose output
        
    Returns:
        List of character folders
    """
    if verbose:
        print("Looking for character folders...")
    
    # First try: direct folders in the dataset path
    character_folders = []
    
    # Check if the provided path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
        return []
    
    # Get all directories in the dataset path
    try:
        potential_folders = [f for f in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, f))]
        
        # Check if these folders contain images (then they are character folders)
        for folder in potential_folders:
            folder_path = os.path.join(dataset_path, folder)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if image_files:
                character_folders.append(folder)
    except Exception as e:
        print(f"Error while listing directories: {e}")
    
    # If found character folders
    if character_folders:
        if verbose:
            print(f"Found {len(character_folders)} character folders directly in {dataset_path}")
        return character_folders
    
    # Second try: check if there's a 'dataset' subfolder
    dataset_subdir = os.path.join(dataset_path, 'dataset')
    if os.path.exists(dataset_subdir) and os.path.isdir(dataset_subdir):
        if verbose:
            print(f"Checking 'dataset' subfolder: {dataset_subdir}")
        
        try:
            potential_folders = [f for f in os.listdir(dataset_subdir) 
                               if os.path.isdir(os.path.join(dataset_subdir, f))]
            
            # Check if these folders contain images
            for folder in potential_folders:
                folder_path = os.path.join(dataset_subdir, folder)
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                    image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
                
                if image_files:
                    # Return full paths for these folders
                    character_folders.append(os.path.join('dataset', folder))
        except Exception as e:
            print(f"Error while checking dataset subfolder: {e}")
        
        if character_folders:
            if verbose:
                print(f"Found {len(character_folders)} character folders in 'dataset' subfolder")
            return character_folders
    
    # Third try: check all subdirectories
    if verbose:
        print("Checking all subdirectories for potential character folders...")
    
    try:
        for root, dirs, files in os.walk(dataset_path):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                # Skip if already checked
                if dir_path == dataset_subdir:
                    continue
                
                # Check if this directory contains images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(glob.glob(os.path.join(dir_path, ext)))
                    image_files.extend(glob.glob(os.path.join(dir_path, ext.upper())))
                
                if image_files:
                    # Get relative path from dataset_path
                    rel_path = os.path.relpath(dir_path, dataset_path)
                    character_folders.append(rel_path)
                    if verbose:
                        print(f"Found character folder: {rel_path} with {len(image_files)} images")
    except Exception as e:
        print(f"Error while walking directories: {e}")
    
    if character_folders:
        if verbose:
            print(f"Found {len(character_folders)} total character folders")
        return character_folders
    
    print("No character folders found in any location")
    return []

def load_dataset(dataset_path, min_images_per_class=15, max_classes=None, target_size=(128, 128), verbose=False):
    """
    Load and preprocess the dataset.
    
    Args:
        dataset_path: Path to the dataset
        min_images_per_class: Minimum number of images required per character
        max_classes: Maximum number of character classes to include (0 for all)
        target_size: Size to resize images to
        verbose: Whether to print verbose output
        
    Returns:
        X, y, class_names
    """
    print(f"Loading dataset from {dataset_path} (filtering classes with < {min_images_per_class} images)...")
    
    # Find all character folders
    character_folders = find_character_folders(dataset_path, verbose)
    
    if not character_folders:
        print("Error: No character folders found")
        return np.array([]), np.array([]), []
    
    # Count images per character
    character_counts = {}
    
    for character in character_folders:
        char_path = os.path.join(dataset_path, character)
        
        # Check if the path exists as is
        if not os.path.exists(char_path):
            print(f"Warning: Path {char_path} does not exist, skipping")
            continue
        
        image_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(char_path, ext)))
            image_files.extend(glob.glob(os.path.join(char_path, ext.upper())))
        
        character_counts[character] = len(image_files)
    
    # Filter characters with enough images
    valid_characters = [char for char, count in character_counts.items() 
                       if count >= min_images_per_class]
    
    print(f"Found {len(valid_characters)} characters with at least {min_images_per_class} images")
    
    # If no valid characters, return empty arrays
    if not valid_characters:
        print("Error: No character folders with sufficient images found")
        print("Character counts:")
        for char, count in character_counts.items():
            print(f"  {char}: {count} images")
        return np.array([]), np.array([]), []
    
    # Limit number of classes if specified
    if max_classes is not None and max_classes > 0 and max_classes < len(valid_characters):
        # Sort by number of images (descending) and take top N
        valid_characters = sorted(valid_characters, 
                                 key=lambda x: character_counts[x], 
                                 reverse=True)[:max_classes]
        print(f"Using top {max_classes} characters with most images")
    
    # Load and preprocess images
    X = []
    y = []
    class_names = []
    
    for i, character in enumerate(valid_characters):
        class_names.append(character)
        char_path = os.path.join(dataset_path, character)
        
        print(f"Processing {character} ({i+1}/{len(valid_characters)}): {character_counts[character]} images")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(char_path, ext)))
            image_files.extend(glob.glob(os.path.join(char_path, ext.upper())))
        
        # Process each image
        for image_path in image_files:
            try:
                if not os.path.exists(image_path):
                    continue
                
                # Preprocess image
                image = preprocess_image(image_path, target_size)
                X.append(image)
                y.append(i)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {len(X)} images across {len(class_names)} characters")
    return X, y, class_names

def create_model(num_classes, input_shape=(128, 128, 3)):
    """
    Create a CNN model for anime face recognition using MobileNetV2 as the base.
    
    Args:
        num_classes: Number of character classes to recognize
        input_shape: Input shape of images (height, width, channels)
        
    Returns:
        A compiled Keras model
    """
    # Use MobileNetV2 as base model (lightweight and efficient)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generator():
    """Create data augmentation pipeline"""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def train_model(X_train, y_train, X_val, y_val, num_classes, args):
    """
    Train the model on the dataset.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        num_classes: Number of classes
        args: Command-line arguments
        
    Returns:
        Trained model and training history
    """
    # Create model
    input_shape = (args.image_size, args.image_size, 3)
    model = create_model(num_classes, input_shape)
    
    # Print model summary
    model.summary()
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        ModelCheckpoint(
            args.model_path,
            save_best_only=True,
            monitor='val_accuracy'
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor='val_accuracy'
        )
    ]
    
    # Initialize training params
    train_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_data': (X_val, y_val),
        'callbacks': callbacks
    }
    
    # Use data augmentation if specified
    if args.use_augmentation:
        print("Using data augmentation")
        datagen = create_data_generator()
        datagen.fit(X_train)
        
        # Train model with data augmentation
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=args.batch_size),
            steps_per_epoch=len(X_train) // args.batch_size,
            **train_params
        )
    else:
        # Train model without augmentation
        history = model.fit(
            X_train, y_train,
            **train_params
        )
    
    return model, history

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        class_names: List of class names
        
    Returns:
        Evaluation metrics
    """
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate top-3 accuracy
    top3_correct = 0
    y_pred_probs = model.predict(X_test)
    
    for i, probs in enumerate(y_pred_probs):
        top_indices = np.argsort(probs)[-3:][::-1]
        if y_test[i] in top_indices:
            top3_correct += 1
    
    top3_accuracy = top3_correct / len(y_test)
    print(f"Top-3 accuracy: {top3_accuracy:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'top3_accuracy': top3_accuracy
    }

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load dataset
    X, y, class_names = load_dataset(
        args.dataset_path,
        min_images_per_class=args.min_images,
        max_classes=args.max_classes,
        target_size=(args.image_size, args.image_size),
        verbose=args.verbose
    )
    
    # Check if dataset is empty
    if len(X) == 0 or len(class_names) == 0:
        print("Error: Empty dataset. Cannot train model.")
        return
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, len(class_names), args)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, class_names)
    
    # Save class names
    os.makedirs(os.path.dirname(args.class_names_path), exist_ok=True)
    with open(args.class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    
    print(f"\nModel saved to {args.model_path}")
    print(f"Class names saved to {args.class_names_path}")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Number of training images: {len(X_train)}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Data augmentation: {'Yes' if args.use_augmentation else 'No'}")
    print(f"  Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.4f}")
    
    print("\nModel training completed!")

if __name__ == "__main__":
    main()