# Import required libraries
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import sys
from werkzeug.utils import secure_filename

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from src
from src.data.preprocessing import preprocess_image, detect_anime_face

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/anime_face_recognition_model.h5'
CLASS_NAMES_PATH = 'models/class_names.pkl'
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and class names
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    print(f"Loaded model with {len(class_names)} character classes")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    class_names = []

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
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
            # Detect faces
            img = cv2.imread(file_path)
            faces = detect_anime_face(img)
            
            results = []
            
            if len(faces) == 0:
                # If no faces detected, process the whole image
                processed_img = preprocess_image(file_path)
                processed_img = np.expand_dims(processed_img, axis=0)
                
                # Make prediction
                predictions = model.predict(processed_img)[0]
                top_idx = np.argsort(predictions)[-3:][::-1]
                
                # Prepare results
                results = [
                    {
                        'character': class_names[idx],
                        'probability': float(predictions[idx]),
                        'percentage': f"{float(predictions[idx]) * 100:.2f}%"
                    }
                    for idx in top_idx
                ]
            else:
                # Process each detected face
                for i, (x, y, w, h) in enumerate(faces):
                    # Crop the face
                    face_img = img[y:y+h, x:x+w]
                    
                    # Preprocess face
                    processed_face = preprocess_image(face_img)
                    processed_face = np.expand_dims(processed_face, axis=0)
                    
                    # Make prediction
                    predictions = model.predict(processed_face)[0]
                    top_idx = np.argsort(predictions)[-3:][::-1]
                    
                    # Prepare results for this face
                    face_results = [
                        {
                            'character': class_names[idx],
                            'probability': float(predictions[idx]),
                            'percentage': f"{float(predictions[idx]) * 100:.2f}%"
                        }
                        for idx in top_idx
                    ]
                    
                    results.append({
                        'face_id': i,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'predictions': face_results
                    })
            
            return jsonify({
                'success': True,
                'results': results
            })
        
        except Exception as e:
            return jsonify({
                'error': f"Error processing image: {str(e)}"
            })
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')