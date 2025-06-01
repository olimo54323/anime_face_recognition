import os
import numpy as np
import cv2
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'anime-face-recognition-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for model and classes
model = None
class_names = []
model_loaded = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_anime_model():
    """Load model and class names"""
    global model, class_names, model_loaded
    
    try:
        # Model file paths
        model_dir = "models"
        model_paths = [
            os.path.join(model_dir, "anime_face_recognition_model.h5"),
            os.path.join(model_dir, "anime_face_recognition_model.keras")
        ]
        
        # Try to load model
        model_loaded_successfully = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"Loading model from: {model_path}")
                    model = load_model(model_path, compile=False)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    model_loaded_successfully = True
                    print(f"Model loaded successfully from: {model_path}")
                    break
                except Exception as e:
                    print(f"Error loading model from {model_path}: {str(e)}")
                    continue
        
        if not model_loaded_successfully:
            print("ERROR: Cannot load model!")
            model = None
            class_names = []
            model_loaded = False
            return False
        
        # Load class names
        class_names_paths = [
            os.path.join(model_dir, "anime_face_recognition_model.pkl"),
            "anime_face_recognition_model.pkl"
        ]
        
        class_names_loaded = False
        for class_path in class_names_paths:
            if os.path.exists(class_path):
                try:
                    with open(class_path, 'rb') as f:
                        class_names = pickle.load(f)
                    print(f"Class names loaded from: {class_path}")
                    print(f"Available classes: {class_names}")
                    class_names_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading class names from {class_path}: {str(e)}")
                    continue
        
        if not class_names_loaded:
            print("Cannot load class names - using default")
            num_classes = model.output_shape[-1] if model else 5
            class_names = [f"character_{i+1}" for i in range(num_classes)]
        
        # Check if class_names has enough elements for model
        if model and len(class_names) < model.output_shape[-1]:
            print(f"WARNING: Model has {model.output_shape[-1]} classes, but class_names has only {len(class_names)}")
            original_classes = class_names.copy()
            while len(class_names) < model.output_shape[-1]:
                class_names.append(f"unknown_class_{len(class_names)}")
            print(f"Extended class_names to {len(class_names)} elements")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"Critical error loading model: {str(e)}")
        model_loaded = False
        return False

def predict_anime_character(image_path):
    """Predict anime character from image"""
    global model, class_names, model_loaded
    
    if not model_loaded or model is None:
        return None, "Model is unavailable. Prediction impossible."
    
    if not class_names:
        return None, "Class names are unavailable. Prediction impossible."
    
    try:
        # Load and prepare image (with preprocessing!)
        test_img = cv2.imread(image_path)
        if test_img is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        test_img = cv2.resize(test_img, (224, 224))
        test_img = test_img.astype(np.float32)
        test_img = preprocess_input(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        
        # Prediction
        result = model.predict(test_img, verbose=0)
        
        # Top 3
        top_3_indices = np.argsort(result[0])[::-1][:3]
        top_3_predictions = []
        
        for i, idx in enumerate(top_3_indices):
            if idx < len(class_names):
                character_name = class_names[idx]
            else:
                character_name = f"unknown_{idx}"
            
            confidence = float(result[0][idx]) * 100
            top_3_predictions.append({
                'rank': i + 1,
                'character': character_name,
                'confidence': round(confidence, 2)
            })
        
        return top_3_predictions, None
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        return None, error_msg

def image_to_base64(image_path):
    """Convert image to base64 for HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        return encoded_string
    except:
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         class_names=class_names, 
                         model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and analyze file"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Perform prediction
            predictions, error = predict_anime_character(file_path)
            
            if error:
                flash(f'Error during analysis: {error}')
                return redirect(url_for('index'))
            
            # Convert image to base64
            image_base64 = image_to_base64(file_path)
            
            return render_template('results.html', 
                                 predictions=predictions,
                                 image_base64=image_base64,
                                 filename=filename,
                                 class_names=class_names)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Allowed: PNG, JPG, JPEG, GIF, BMP')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting Anime Face Recognition application...")
    
    # Load model at startup
    print("Loading model...")
    try:
        success = load_anime_model()
        
        if success and model_loaded and model is not None:
            print(f"SUCCESS: Model loaded successfully! Available classes: {len(class_names)}")
            for i, name in enumerate(class_names):
                print(f"  {i+1}. {name}")
        else:
            print("WARNING: Model not loaded - prediction will be unavailable")
    except Exception as e:
        print(f"ERROR during loading: {e}")
    
    print("Server starting on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)