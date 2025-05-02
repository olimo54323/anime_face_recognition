# Import required libraries
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import glob
import kagglehub
from werkzeug.utils import secure_filename
import shutil
import zipfile

# Initialize Flask app
app = Flask(__name__)

# Path configurations
UPLOAD_FOLDER = 'app/static/uploads'
MODEL_PATH = 'models/anime_face_recognition_model.h5'
CLASS_NAMES_PATH = 'models/class_names.pkl'
DATASET_PATH = 'dataset'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def download_dataset():
    """Download the dataset from Kaggle if it doesn't exist locally"""
    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        print("Downloading dataset from Kaggle...")
        # Download dataset
        dataset_path = kagglehub.dataset_download("thedevastator/anime-face-dataset-by-character-name")
        
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Jeśli dataset_path istnieje i jest plikiem, spróbuj go rozpakować
        if os.path.isfile(dataset_path):
            print(f"Downloaded dataset is a file: {dataset_path}")
            # Sprawdź czy to plik ZIP
            if dataset_path.endswith('.zip'):
                print("Extracting files...")
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    # Wydrukuj kilka pierwszych plików w ZIP, aby zrozumieć strukturę
                    print("ZIP contents (first 10 files):")
                    for i, file_info in enumerate(zip_ref.infolist()[:10]):
                        print(f"- {file_info.filename}")
                    
                    # Utwórz katalog docelowy, jeśli nie istnieje
                    if not os.path.exists(DATASET_PATH):
                        os.makedirs(DATASET_PATH)
                    
                    # Rozpakuj
                    zip_ref.extractall(DATASET_PATH)
        
        print(f"Dataset downloaded to {DATASET_PATH}")
    else:
        print(f"Dataset already exists at {DATASET_PATH}")
    
    # Sprawdź strukturę katalogu
    print("\nDataset structure:")
    character_folders = [f for f in os.listdir(DATASET_PATH) 
                       if os.path.isdir(os.path.join(DATASET_PATH, f))]
    print(f"Found {len(character_folders)} character folders")
    
    # Jeśli nie znaleziono żadnych katalogów postaci, sprawdź, czy istnieje struktura zagnieżdżona
    if len(character_folders) == 0:
        print("No character folders found, checking for nested structure...")
        subdirs = [f for f in os.listdir(DATASET_PATH) 
                 if os.path.isdir(os.path.join(DATASET_PATH, f))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(DATASET_PATH, subdir)
            sub_character_folders = [f for f in os.listdir(subdir_path) 
                                  if os.path.isdir(os.path.join(subdir_path, f))]
            
            print(f"Found {len(sub_character_folders)} character folders in {subdir}")
            
            if len(sub_character_folders) > 0:
                print(f"Moving character folders from {subdir} to main dataset directory")
                # Przenieś katalogi postaci do głównego katalogu danych
                for char_folder in sub_character_folders:
                    src = os.path.join(subdir_path, char_folder)
                    dst = os.path.join(DATASET_PATH, char_folder)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                    else:
                        print(f"Folder {char_folder} already exists, merging...")
                        # Sklonuj pliki z src do dst
                        for file in os.listdir(src):
                            src_file = os.path.join(src, file)
                            dst_file = os.path.join(dst, file)
                            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
    
    # Sprawdź ponownie katalogi postaci
    character_folders = [f for f in os.listdir(DATASET_PATH) 
                       if os.path.isdir(os.path.join(DATASET_PATH, f))]
    print(f"Final count: {len(character_folders)} character folders")
    
    # Przykładowo pokaż kilka pierwszych katalogów i liczbę obrazów w każdym
    for char_folder in character_folders[:5]:
        folder_path = os.path.join(DATASET_PATH, char_folder)
        image_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"- {char_folder}: {len(image_files)} images")

# Check if model exists, if not, train it
def check_and_train_model():
    # Download dataset if needed
    download_dataset()
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        print("Model not found. Training new model...")
        train_model()
    
    # Load the model and class names
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    
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

def train_model():
    """Train a new model on the dataset"""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    
    # Load dataset
    X = []
    y = []
    class_names = []
    
    # Get all character folders (classes)
    character_folders = [f for f in os.listdir(DATASET_PATH) 
                       if os.path.isdir(os.path.join(DATASET_PATH, f))]
    
    print(f"Found {len(character_folders)} character folders")
    
    # Wyświetl informacje o każdym folderze
    for i, character in enumerate(character_folders):
        class_names.append(character)
        character_path = os.path.join(DATASET_PATH, character)
        
        print(f"Processing {character} ({i+1}/{len(character_folders)})")
        
        # Znajdź wszystkie pliki obrazów w folderze postaci
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(character_path, ext)))
            image_files.extend(glob.glob(os.path.join(character_path, ext.upper())))
        
        print(f"  Found {len(image_files)} images")
        
        # Jeśli nie znaleziono obrazów, kontynuuj z następną postacią
        if len(image_files) == 0:
            continue
        
        # Iteruj przez wszystkie obrazy w folderze postaci
        for image_path in image_files:
            try:
                # Sprawdź czy plik faktycznie istnieje
                if not os.path.exists(image_path):
                    print(f"  Warning: File does not exist: {image_path}")
                    continue
                
                # Przetwórz obraz
                image = preprocess_image(image_path)
                X.append(image)
                y.append(i)
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {len(X)} images across {len(class_names)} characters")
    
    # Sprawdź, czy mamy wystarczająco dużo danych do treningu
    if len(X) < 10 or len(np.unique(y)) < 2:
        raise ValueError(f"Not enough data for training. Found {len(X)} images across {len(np.unique(y))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create the model using transfer learning
    num_classes = len(class_names)
    
    # Use MobileNetV2 as base model (lightweight and efficient)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(128, 128, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=3, 
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save(MODEL_PATH)
    
    # Save class names
    with open(CLASS_NAMES_PATH, 'wb') as f:
        pickle.dump(class_names, f)
    
    print(f"Model trained and saved. Recognized characters: {len(class_names)}")
    return model, class_names

# Load or train model
try:
    model, class_names = check_and_train_model()
    print(f"Model loaded. Recognized characters: {len(class_names)}")
except Exception as e:
    print(f"Error loading/training model: {e}")
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
            return jsonify({
                'error': f"Error processing image: {str(e)}"
            })
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')