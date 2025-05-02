import cv2
import numpy as np

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

def detect_anime_face(image):
    """
    Detect anime face in an image using an anime-specific face detector.
    This is a simplified placeholder - actual implementation would use 
    a trained anime face detector.
    
    Args:
        image: Input image (can be path or numpy array)
        
    Returns:
        List of (x, y, w, h) tuples for detected faces
    """
    # Load image if path is provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        img = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use a custom cascade classifier for anime faces
    # Note: This is a placeholder - you'll need a trained cascade for anime faces
    try:
        # Try to use an anime face cascade if available
        cascade = cv2.CascadeClassifier('models/lbpcascade_animeface.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    except:
        # Fallback to a standard face detector
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces