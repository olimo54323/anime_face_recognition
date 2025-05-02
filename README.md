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

### Kaggle API configuration
1. Sign up for [Kaggle](https://www.kaggle.com) (if you don't have an account yet)
2. Go to your account settings (click on your profile picture)
3. Scroll to the "API" section and click "Create New API Token"
4. Download the kaggle.json file
5. Place this file in the project root (it will be copied to the container)

### Local installation

```bash
# Cloning the repository
git clone https://github.com/olimo54323/anime_face_recognition.git
cd anime-face-recognition 

# Installing dependencies
pip install -r requirements.txt

# Running the application
python app/app.py
```

### Docker instalation (recommended)

```bash
# Cloning the repository
git clone https://github.com/olimo54323/anime_face_recognition.git
cd anime-face-recognition 

# Building Docker image
docker build -t anime-face-recognition .

# Running the container
docker run -p 5000:5000 anime-face-recognition
```

## Authors
- Grzegorz Urbański
- Wiktor Kaszuba