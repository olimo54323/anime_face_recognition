# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

def create_model(num_classes, input_shape=(128, 128, 3)):
    """
    Create a CNN model for anime face recognition using MobileNetV2 as the base.
    
    Args:
        num_classes: Number of character classes to recognize
        input_shape: Input shape of images (height, width, channels)
        
    Returns:
        A compiled Keras model
    """
    # Use MobileNetV2 as base model (lightweight and efficient)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
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
    
    return model