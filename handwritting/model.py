import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(num_classes):
    """Build a CNN model for handwritten character recognition."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_crnn_model(num_classes):
    """Build a CRNN model for sequence recognition (extendable to full words/sentences)."""
    # This is a simplified CRNN; for full sequences, you'd need CTC loss and more complex setup
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Reshape((7, 7*64)),  # Reshape for RNN
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model):
    """Compile the model with optimizer, loss, and metrics."""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds, epochs=10):
    """Train the model."""
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history

def evaluate_model(model, test_ds):
    """Evaluate the model on test data."""
    loss, accuracy = model.evaluate(test_ds)
    return loss, accuracy

def predict(model, image):
    """Predict on a single image."""
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return tf.argmax(prediction, axis=1).numpy()[0]
