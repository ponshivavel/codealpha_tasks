import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2

def load_mnist():
    """Load MNIST dataset for digits."""
    (train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)
    return train_ds, test_ds, ds_info

def load_emnist():
    """Load EMNIST dataset for characters."""
    (train_ds, test_ds), ds_info = tfds.load('emnist/byclass', split=['train', 'test'], as_supervised=True, with_info=True)
    return train_ds, test_ds, ds_info

def preprocess_image(image, label):
    """Preprocess images: normalize and resize if needed."""
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)  # Add channel dimension
    return image, label

def create_data_pipeline(dataset, batch_size=32, shuffle=True):
    """Create data pipeline with preprocessing."""
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_and_preprocess_data(dataset_name='mnist', batch_size=32):
    """Load and preprocess dataset."""
    if dataset_name == 'mnist':
        train_ds, test_ds, ds_info = load_mnist()
    elif dataset_name == 'emnist':
        train_ds, test_ds, ds_info = load_emnist()
    else:
        raise ValueError("Dataset must be 'mnist' or 'emnist'")

    train_ds = create_data_pipeline(train_ds, batch_size)
    test_ds = create_data_pipeline(test_ds, batch_size, shuffle=False)
    return train_ds, test_ds, ds_info
