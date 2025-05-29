import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

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
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB (OpenCV loads in BGR, but models expect RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        image = cv2.resize(image, target_size)
        
        # Convert to float and normalize
        image = image.astype(np.float32)
        
        image = preprocess_input(image)
        
        return image
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def validate_image(image_path):
    """
    Validate if the image file is readable and has valid format
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        image = cv2.imread(image_path)
        return image is not None
    except:
        return False