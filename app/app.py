# Setup paths first
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required libraries
from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import pickle
import logging
from werkzeug.utils import secure_filename
import glob

# Force TensorFlow import with specific settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Import project-specific modules
try:
    from scripts.preprocessing import preprocess_image, validate_image
except ImportError:
    # Fallback import path
    sys.path.append('/app')
    from scripts.preprocessing import preprocess_image, validate_image

# Lazy import for TensorFlow - only when needed
tf = None

def get_tensorflow():
    """Lazy import TensorFlow only when needed with error handling"""
    global tf
    if tf is None:
        try:
            # Set TensorFlow to use only CPU and reduce warnings
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            import tensorflow as tf_module
            
            # Configure TensorFlow
            tf_module.config.threading.set_intra_op_parallelism_threads(1)
            tf_module.config.threading.set_inter_op_parallelism_threads(1)
            
            tf = tf_module
            logger.info(f"TensorFlow loaded successfully, version: {tf.version.VERSION}")
        except Exception as e:
            logger.error(f"Failed to import TensorFlow: {e}")
            raise ImportError(f"Could not import TensorFlow: {e}")
    return tf

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
            
            # Check for different pickle file naming patterns
            pickle_patterns = [
                f"{model_name}.pkl",                    # exact match
                f"{model_name}_class_names.pkl",        # with _class_names suffix
            ]
            
            pickle_found = False
            for pattern in pickle_patterns:
                pickle_file = os.path.join(MODELS_FOLDER, pattern)
                if os.path.exists(pickle_file):
                    pickle_found = True
                    break
            
            if pickle_found:
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
    
    # Try different pickle file naming patterns
    pickle_patterns = [
        f"{model_name}.pkl",                    # exact match
        f"{model_name}_class_names.pkl",        # with _class_names suffix
    ]
    
    pickle_path = None
    for pattern in pickle_patterns:
        potential_path = os.path.join(MODELS_FOLDER, pattern)
        if os.path.exists(potential_path):
            pickle_path = potential_path
            break
    
    if pickle_path is None:
        raise FileNotFoundError(f"Class names file not found. Tried: {pickle_patterns}")
    
    # Load TensorFlow and the model with different strategies
    tf = get_tensorflow()
    
    logger.info(f"potential path: {potential_path}")
    logger.info(f"potential path: {model_path}")
    try:
        # Strategy 1: Try direct loading
        current_model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded with keras.models.load_model")
    except Exception as e1:
        logger.warning(f"Direct loading failed: {e1}")
        try:
            # Strategy 2: Load with compile=False
            current_model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Model loaded with compile=False")
        except Exception as e2:
            logger.warning(f"Loading with compile=False failed: {e2}")
            try:
                # Strategy 3: Load as SavedModel if it's a directory
                if os.path.isdir(model_path):
                    current_model = tf.saved_model.load(model_path)
                    logger.info(f"Model loaded as SavedModel")
                else:
                    raise Exception("All loading strategies failed")
            except Exception as e3:
                logger.error(f"All model loading strategies failed: {e1}, {e2}, {e3}")
                raise Exception(f"Could not load model: {model_path}")
    
    # Load class names
    with open(pickle_path, 'rb') as f:
        current_class_names = pickle.load(f)
    
    current_model_name = model_name
    
    logger.info(f"Model loaded successfully from: {model_path}")
    logger.info(f"Class names loaded from: {pickle_path}")
    logger.info(f"Recognizes {len(current_class_names)} characters.")
    logger.info(f"Characters: {current_class_names}")
    
    return current_model, current_class_names

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Disable auto-loading of models at startup to avoid import errors
def safe_load_default_model():
    """Safely try to load default model without crashing the app"""
    try:
        available_models = get_available_models()
        if available_models:
            default_model = available_models[0]
            load_model(default_model)
            logger.info(f"Default model '{default_model}' loaded successfully")
        else:
            logger.warning("No models found in models folder")
    except Exception as e:
        logger.error(f"Could not load default model (this is OK, load manually): {e}")

# Try to load default model, but don't crash if it fails
# safe_load_default_model()  # Commented out for now

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