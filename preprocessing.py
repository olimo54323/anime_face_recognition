import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224), model_type='resnet50'):
    """
    Preprocess image for anime face recognition
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
        model_type (str): Type of model being used for preprocessing
    
    Returns:
        np.array: Preprocessed image ready for prediction
    """
    try:
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Resize image to target size
        image = cv2.resize(image, target_size)
        
        # Convert BGR to RGB (OpenCV loads in BGR, but TensorFlow expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Apply preprocessing based on model type
        if model_type.lower() in ['resnet50', 'resnet', 'imagenet']:
            # Use TensorFlow's built-in ResNet50 preprocessing
            # This applies ImageNet normalization automatically
            image = preprocess_input(image)
        else:
            # Default normalization for custom models
            image = image / 255.0
        
        return image
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def preprocess_image_pil(image_path, target_size=(224, 224)):
    """
    Alternative preprocessing using PIL (sometimes more reliable)
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
    
    Returns:
        np.array: Preprocessed image ready for prediction
    """
    try:
        # Open image with PIL
        image = Image.open(image_path)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image = np.array(image, dtype=np.float32)
        
        # Apply ResNet50 preprocessing
        image = preprocess_input(image)
        
        return image
        
    except Exception as e:
        raise Exception(f"Error preprocessing image with PIL: {str(e)}")

def validate_image(image_path):
    """
    Validate if the image file is readable and has valid format
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Try OpenCV first
        image = cv2.imread(image_path)
        if image is not None:
            return True
        
        # Try PIL as backup
        with Image.open(image_path) as img:
            img.verify()
            return True
            
    except Exception:
        return False

def get_image_info(image_path):
    """
    Get basic information about the image
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'valid': True
            }
    except Exception:
        return {'valid': False}

def preprocess_for_prediction(image_path, target_size=(224, 224)):
    """
    Complete preprocessing pipeline for prediction
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
    
    Returns:
        np.array: Preprocessed image with batch dimension
    """
    try:
        # First try OpenCV method
        try:
            image = preprocess_image(image_path, target_size, 'resnet50')
        except:
            # Fallback to PIL method
            image = preprocess_image_pil(image_path, target_size)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        raise Exception(f"Failed to preprocess image for prediction: {str(e)}")