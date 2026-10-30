import tensorflow as tf
from tensorflow.keras import layers, models

def create_temporal_lstm(input_shape=(30, 3)):
    """
    Creates a Long Short-Term Memory (LSTM) network to process 
    daily meteorological time-series data (Temperature, Rainfall, Humidity).
    
    Args:
        input_shape (tuple): Shape of time-series (timesteps, features).
        
    Returns:
        tf.keras.Model: Feature extractor model returning temporal vectors.
    """
    inputs = layers.Input(shape=input_shape, name="temporal_input")
    
    # LSTM Layers for sequence memory modeling
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False, name="temporal_feature_layer")(x)
    
    # Dense projection for temporal feature vector
    outputs = layers.Dense(32, activation='relu', name="temporal_vector")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Temporal_LSTM")
    return model

if __name__ == "__main__":
    lstm_model = create_temporal_lstm()
    lstm_model.summary()