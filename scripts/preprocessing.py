import cv2
import numpy as np

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
        
        # Apply ResNet50-style preprocessing (same as ImageNet)
        if model_type.lower() in ['resnet50', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1']:
            # ImageNet preprocessing: 
            # Convert from RGB [0,255] to BGR [-103.939, 116.779, 123.68]
            image = image[..., ::-1]  # RGB to BGR
            mean = [103.939, 116.779, 123.68]
            image[..., 0] -= mean[0]  # B
            image[..., 1] -= mean[1]  # G  
            image[..., 2] -= mean[2]  # R
        else:
            # Default normalization for custom models
            image = image / 255.0
        
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