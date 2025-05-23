# Import required libraries
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import logging
from werkzeug.utils import secure_filename

# Importuj moduły specyficzne dla projektu
from scripts.preprocessing import preprocess_image

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

# Konfiguracja modelu (domyślnie mobilenetv2)
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'mobilenetv2')
MODEL_PATH = MODEL_PATH.replace('.h5', f'_{MODEL_TYPE}.h5')
CLASS_NAMES_PATH = CLASS_NAMES_PATH.replace('.pkl', f'_{MODEL_TYPE}.pkl')

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
    logger.info(f"Model type: {MODEL_TYPE}")
    
    return model, class_names

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    # Przekaż typ modelu do szablonu
    return render_template('index.html', characters=class_names, model_type=MODEL_TYPE)

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
            # Determine image size based on model type
            target_size = (240, 240) if MODEL_TYPE == 'efficientnetb1' else (224, 224)
            
            # Preprocess image
            processed_img = preprocess_image(
                file_path, 
                model_type=MODEL_TYPE, 
                target_size=target_size
            )
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
                'model_type': MODEL_TYPE,
                'top_predictions': top_predictions
            })
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({
                'error': f"Error processing image: {str(e)}"
            })
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    model_dir = os.path.dirname(MODEL_PATH)
    
    available_models = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.h5'):
                # Extract model type from filename
                model_name = file.replace('anime_face_recognition_model_', '').replace('.h5', '')
                available_models.append(model_name)
    
    return jsonify({
        'success': True,
        'current_model': MODEL_TYPE,
        'available_models': available_models
    })

@app.route('/model/<model_type>', methods=['POST'])
def switch_model(model_type):
    """Switch to a different model"""
    global model, class_names, MODEL_TYPE
    
    # Validate model type
    if model_type not in ['mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'resnet50']:
        return jsonify({'error': f'Invalid model type: {model_type}'})
    
    # Update paths
    new_model_path = MODEL_PATH.replace(MODEL_TYPE, model_type)
    new_class_names_path = CLASS_NAMES_PATH.replace(MODEL_TYPE, model_type)
    
    # Check if files exist
    if not os.path.exists(new_model_path):
        return jsonify({'error': f'Model file not found: {new_model_path}'})
    
    if not os.path.exists(new_class_names_path):
        return jsonify({'error': f'Class names file not found: {new_class_names_path}'})
    
    try:
        # Load new model
        new_model = tf.keras.models.load_model(new_model_path)
        
        # Load new class names
        with open(new_class_names_path, 'rb') as f:
            new_class_names = pickle.load(f)
        
        # Update global variables
        model = new_model
        class_names = new_class_names
        MODEL_TYPE = model_type
        
        logger.info(f"Switched to model: {MODEL_TYPE}")
        
        return jsonify({
            'success': True,
            'model_type': MODEL_TYPE,
            'num_classes': len(class_names)
        })
    
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({'error': f"Error switching model: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)