import tensorflow as tf
from tensorflow.keras import layers, models

def create_spatial_cnn(input_shape=(64, 64, 4)):
    """
    Creates a Convolutional Neural Network (CNN) to process 
    multi-spectral Sentinel-2 image patches (e.g., RGB + NIR bands).
    
    Args:
        input_shape (tuple): Shape of image patches (height, width, channels).
        
    Returns:
        tf.keras.Model: Feature extractor model returning spatial vectors.
    """
    inputs = layers.Input(shape=input_shape, name="spatial_input")
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D(name="spatial_feature_layer")(x)
    
    # Dense projection for spatial feature vector
    outputs = layers.Dense(64, activation='relu', name="spatial_vector")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Spatial_CNN")
    return model

if __name__ == "__main__":
    cnn_model = create_spatial_cnn()
    cnn_model.summary()