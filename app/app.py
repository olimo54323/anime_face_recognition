# Import required libraries
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import logging
from werkzeug.utils import secure_filename
import glob

# Import project-specific modules
from scripts.preprocessing import preprocess_image, validate_image

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
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for current model
current_model = None
current_class_names = []
current_model_name = None

def get_available_models():
    """Get list of available models from models folder"""
    available_models = []
    
    # Look for .h5 and .keras files
    for extension in ['*.h5', '*.keras']:
        model_files = glob.glob(os.path.join(MODELS_FOLDER, extension))
        for model_file in model_files:
            # Extract model name without extension
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            
            # Check if corresponding pickle file exists
            pickle_file = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")
            if os.path.exists(pickle_file):
                available_models.append(model_name)
    
    return list(set(available_models))  # Remove duplicates

def load_model(model_name):
    """Load the specified model and its class names"""
    global current_model, current_class_names, current_model_name
    
    logger.info(f"Attempting to load model: {model_name}")
    
    # Try both .h5 and .keras extensions
    model_path = None
    for extension in ['.h5', '.keras']:
        potential_path = os.path.join(MODELS_FOLDER, f"{model_name}{extension}")
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Model file not found for: {model_name}")
    
    pickle_path = os.path.join(MODELS_FOLDER, f"{model_name}.pkl")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Class names file not found: {pickle_path}")
    
    # Load the model
    current_model = tf.keras.models.load_model(model_path)
    
    # Load class names
    with open(pickle_path, 'rb') as f:
        current_class_names = pickle.load(f)
    
    current_model_name = model_name
    
    logger.info(f"Model loaded successfully. Recognizes {len(current_class_names)} characters.")
    logger.info(f"Characters: {current_class_names}")
    
    return current_model, current_class_names

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load default model at startup
try:
    available_models = get_available_models()
    if available_models:
        default_model = available_models[0]
        load_model(default_model)
        logger.info(f"Default model '{default_model}' loaded successfully")
    else:
        logger.warning("No models found in models folder")
except Exception as e:
    logger.error(f"Error loading default model: {e}")

@app.route('/')
def home():
    """Render home page"""
    available_models = get_available_models()
    return render_template('index.html', 
                         characters=current_class_names, 
                         current_model=current_model_name,
                         available_models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction with top 3 results"""
    if current_model is None:
        return jsonify({'error': 'No model loaded. Please check server logs.'})
    
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
            # Validate image
            if not validate_image(file_path):
                return jsonify({'error': 'Invalid or corrupted image file'})
            
            # Preprocess image
            processed_img = preprocess_image(file_path, target_size=(224, 224))
            processed_img = np.expand_dims(processed_img, axis=0)
            
            # Make prediction
            predictions = current_model.predict(processed_img)[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]  # Get top 3 in descending order
            
            # Prepare results
            top_predictions = []
            for idx in top_3_indices:
                character_name = current_class_names[idx]
                probability = float(predictions[idx])
                percentage = f"{probability * 100:.2f}%"
                
                top_predictions.append({
                    'character': character_name,
                    'probability': probability,
                    'percentage': percentage
                })
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            return jsonify({
                'success': True,
                'model_name': current_model_name,
                'top_predictions': top_predictions
            })
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Clean up uploaded file in case of error
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({
                'error': f"Error processing image: {str(e)}"
            })
    
    return jsonify({'error': 'Invalid file format. Please upload PNG, JPG, or JPEG file.'})

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    available_models = get_available_models()
    
    return jsonify({
        'success': True,
        'current_model': current_model_name,
        'available_models': available_models,
        'total_models': len(available_models)
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    data = request.get_json()
    
    if not data or 'model_name' not in data:
        return jsonify({'error': 'Model name not provided'})
    
    model_name = data['model_name']
    available_models = get_available_models()
    
    if model_name not in available_models:
        return jsonify({'error': f'Model {model_name} not found. Available models: {available_models}'})
    
    try:
        # Load the new model
        load_model(model_name)
        
        logger.info(f"Successfully switched to model: {model_name}")
        
        return jsonify({
            'success': True,
            'model_name': current_model_name,
            'num_classes': len(current_class_names),
            'characters': current_class_names
        })
    
    except Exception as e:
        logger.error(f"Error switching to model {model_name}: {e}")
        return jsonify({'error': f"Error loading model {model_name}: {str(e)}"})

@app.route('/model_info')
def model_info():
    """Get current model information"""
    if current_model is None:
        return jsonify({'error': 'No model loaded'})
    
    return jsonify({
        'success': True,
        'model_name': current_model_name,
        'num_classes': len(current_class_names),
        'characters': current_class_names,
        'model_summary': str(current_model.summary())
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)