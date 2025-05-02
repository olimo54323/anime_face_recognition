# Anime face recognition

An app for recognizing anime characters' faces using machine learning.

## Description

Project uses machine learning and computer vision techniques to recognize anime characters based on their faces. The model was trained on a dataset containing images of anime characters' faces.

## Functions

- Anime character face recognition
- Returning match probability for top-3 characters
- Web interface for easy testing
## Technologies

- TensorFlow/Keras - Model building and training
- Flask - Web app backend
- OpenCV - Image processing
- Docker - Application containerization

## Installation

### Requirements
- Python 3.8+
- Docker (optional)

### Local installation

```bash
# Cloning the repository
git clone https://github.com/username/anime-face-recognition.git
cd anime-face-recognition

# Installing dependencies
pip install -r requirements.txt

# Running the application
python app/app.py
```

### Docker instalation

```bash
# Budowanie obrazu Docker
docker build -t anime-face-recognition .

# Uruchomienie kontenera
docker run -p 5000:5000 anime-face-recognition
```

## Authors
- Grzegorz Urbański
- Wiktor Kaszuba