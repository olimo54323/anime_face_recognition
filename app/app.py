# Import required libraries
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Path configurations
UPLOAD_FOLDER = 'app/static/uploads'
MODEL_PATH = 'models/anime_face_recognition_model.h5'
CLASS_NAMES_PATH = 'models/class_names.pkl'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_model():
    """Load the trained model and class names"""
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    logger.info(f"Attempting to load class names from: {CLASS_NAMES_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")
    
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    
    logger.info(f"Model and class names loaded successfully. Model recognizes {len(class_names)} characters.")
    
    return model, class_names

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the image
        target_size: Size to resize the image to (width, height)
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to range [0, 1]
    img = img / 255.0
    
    return img

# Load model at startup
try:
    model, class_names = load_model()
    logger.info(f"Model loaded. Recognized characters: {len(class_names)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    class_names = []

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html', characters=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Preprocess image
            processed_img = preprocess_image(file_path)
            processed_img = np.expand_dims(processed_img, axis=0)
            
            # Make prediction
            predictions = model.predict(processed_img)[0]
            top_idx = np.argsort(predictions)[-3:][::-1]
            
            # Prepare results
            top_predictions = [
                {
                    'character': class_names[idx],
                    'probability': float(predictions[idx]),
                    'percentage': f"{float(predictions[idx]) * 100:.2f}%"
                }
                for idx in top_idx
            ]
            
            return jsonify({
                'success': True,
                'top_predictions': top_predictions
            })
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({
                'error': f"Error processing image: {str(e)}"
            })
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)