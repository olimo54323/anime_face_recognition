import os
import numpy as np
import cv2
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Utworz folder uploads jesli nie istnieje
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dozwolone rozszerzenia plikow
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Globalne zmienne dla modelu i klas
model = None
class_names = []
model_loaded = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_anime_model():
    """Zaladuj model i nazwy klas"""
    global model, class_names, model_loaded
    
    try:
        # Sciezki do plikow modelu
        model_dir = "models"
        model_paths = [
            os.path.join(model_dir, "anime_face_recognition_model.h5"),
            os.path.join(model_dir, "anime_face_recognition_model.keras")
        ]
        
        # Sprobuj zaladowac model
        model_loaded_successfully = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"Ladowanie modelu z: {model_path}")
                    model = load_model(model_path, compile=False)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    model_loaded_successfully = True
                    print(f"Model zaladowany pomyslnie z: {model_path}")
                    break
                except Exception as e:
                    print(f"Blad ladowania modelu z {model_path}: {str(e)}")
                    continue
        
        if not model_loaded_successfully:
            print("Nie mozna zaladowac modelu - tworze model testowy")
            # Stworz prosty model testowy
            model = create_dummy_model()
            class_names = ["test_character_1", "test_character_2", "test_character_3"]
            model_loaded = True
            return True
        
        # Zaladuj nazwy klas
        class_names_paths = [
            os.path.join(model_dir, "anime_face_recognition_model_class_names.pkl"),
            "anime_face_recognition_model_class_names.pkl"
        ]
        
        class_names_loaded = False
        for class_path in class_names_paths:
            if os.path.exists(class_path):
                try:
                    with open(class_path, 'rb') as f:
                        class_names = pickle.load(f)
                    print(f"Nazwy klas zaladowane z: {class_path}")
                    print(f"Dostepne klasy: {class_names}")
                    class_names_loaded = True
                    break
                except Exception as e:
                    print(f"Blad ladowania nazw klas z {class_path}: {str(e)}")
                    continue
        
        if not class_names_loaded:
            print("Nie mozna zaladowac nazw klas - uzywam domyslnych")
            # Sprobuj wywnioskować z modelu
            try:
                num_classes = model.output_shape[-1]
                class_names = [f"character_{i+1}" for i in range(num_classes)]
            except:
                class_names = ["character_1", "character_2", "character_3", "character_4", "character_5"]
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"Krytyczny blad ladowania modelu: {str(e)}")
        return False

def create_dummy_model():
    """Stworz prosty model testowy"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 klasy testowe
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image_for_prediction(image_path, target_size=(224, 224)):
    """Przetwarzanie obrazu do predykcji"""
    try:
        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nie mozna wczytac obrazu z {image_path}")
        
        # Zmien rozmiar
        image = cv2.resize(image, target_size)
        
        # Konwertuj BGR do RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalizacja dla ResNet50
        image = image.astype(np.float32)
        image = preprocess_input(image)
        
        # Dodaj wymiar batch
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        print(f"Blad przetwarzania obrazu: {str(e)}")
        return None

def predict_anime_character(image_path):
    """Przewiduj postac anime z obrazu"""
    global model, class_names, model_loaded
    
    if not model_loaded or model is None:
        return None, "Model nie jest zaladowany"
    
    try:
        # Przetwórz obraz
        processed_image = preprocess_image_for_prediction(image_path)
        if processed_image is None:
            return None, "Nie mozna przetworzyc obrazu"
        
        # Wykonaj predykcje
        predictions = model.predict(processed_image)
        
        # Pobierz top 3 predykcji
        top_3_indices = np.argsort(predictions[0])[::-1][:3]
        top_3_predictions = []
        
        for i, idx in enumerate(top_3_indices):
            character_name = class_names[idx] if idx < len(class_names) else f"unknown_{idx}"
            confidence = float(predictions[0][idx]) * 100
            top_3_predictions.append({
                'rank': i + 1,
                'character': character_name,
                'confidence': round(confidence, 2)
            })
        
        return top_3_predictions, None
        
    except Exception as e:
        error_msg = f"Blad podczas predykcji: {str(e)}"
        print(error_msg)
        return None, error_msg

def image_to_base64(image_path):
    """Konwertuj obraz do base64 dla wyswietlenia w HTML"""
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        return encoded_string
    except:
        return None

@app.route('/')
def index():
    """Strona glowna"""
    return render_template('index.html', 
                         class_names=class_names, 
                         model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload i analiza pliku"""
    if 'file' not in request.files:
        flash('Nie wybrano pliku')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Nie wybrano pliku')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Wykonaj predykcje
            predictions, error = predict_anime_character(file_path)
            
            if error:
                flash(f'Blad podczas analizy: {error}')
                return redirect(url_for('index'))
            
            # Konwertuj obraz do base64
            image_base64 = image_to_base64(file_path)
            
            return render_template('results.html', 
                                 predictions=predictions,
                                 image_base64=image_base64,
                                 filename=filename,
                                 class_names=class_names)
            
        except Exception as e:
            flash(f'Blad podczas przetwarzania pliku: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Nieprawidlowy format pliku. Dozwolone: PNG, JPG, JPEG, GIF, BMP')
        return redirect(url_for('index'))

@app.route('/api/classes')
def get_classes():
    """API endpoint zwracajacy dostepne klasy"""
    return jsonify({
        'classes': class_names,
        'model_loaded': model_loaded,
        'num_classes': len(class_names)
    })

@app.errorhandler(413)
def too_large(e):
    flash('Plik jest za duzy. Maksymalny rozmiar to 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Uruchamianie aplikacji Anime Face Recognition...")
    
    # Zaladuj model przy starcie
    print("Ladowanie modelu...")
    success = load_anime_model()
    
    if success:
        print(f"Model zaladowany pomyslnie! Dostepne klasy: {len(class_names)}")
        for i, name in enumerate(class_names):
            print(f"  {i+1}. {name}")
    else:
        print("UWAGA: Nie udalo sie zaladowac modelu. Aplikacja bedzie dzialac w trybie demo.")
    
    print("Serwer uruchamia sie na http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)