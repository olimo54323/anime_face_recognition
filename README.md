# Anime face recognition

An app for recognizing anime characters' faces using machine learning.

## Description

Project uses machine learning and computer vision techniques to recognize anime characters based on their faces. The model was trained on a dataset containing images of anime characters' faces.

## Functions

- Anime character face recognition
- Web interface for easy testing
## Technologies

- TensorFlow/Keras - Model building and training, image preprocessing
- Flask - Web app backend
- cv2 - Image tool

## Installation

### Requirements
- Python 3.8+
- google colab/jupiter
- docker

```bash
# Cloning the repository
git clone https://github.com/olimo54323/anime_face_recognition.git
cd anime-face-recognition 

# Building Docker image
docker build -t anime-face-recognition .

# run docker container
docker run -d --name anime-app -p 5000:5000 anime-face-recognition

# (diagnostic) check logs
docker logs anime-app
```

open:

http://localhost:5000


## First model (small number of classes: 5) conclusion:
![training plots](img/training_plots.png)

![sample predictions](img/sample_predictions.png)

![confusion matrix](img/confusion_matrix.png)

![raport 1](img/raport.png)

![raport 2](img/raport1.png)

## Authors
- Grzegorz Urbański
- Wiktor Kaszuba