import argparse
from utils import load_and_preprocess_data
from model import build_cnn_model, build_crnn_model, compile_model, train_model, evaluate_model, predict
import tensorflow as tf
import numpy as np
import cv2

def main(dataset='mnist', model_type='cnn', epochs=10, batch_size=32):
    """Main function to run the handwritten character recognition."""
    print(f"Loading {dataset} dataset...")
    train_ds, test_ds, ds_info = load_and_preprocess_data(dataset, batch_size)
    
    num_classes = ds_info.features['label'].num_classes
    print(f"Number of classes: {num_classes}")
    
    if model_type == 'cnn':
        model = build_cnn_model(num_classes)
    elif model_type == 'crnn':
        model = build_crnn_model(num_classes)
    else:
        raise ValueError("Model type must be 'cnn' or 'crnn'")
    
    model = compile_model(model)
    print("Model compiled.")
    
    print("Training model...")
    history = train_model(model, train_ds, test_ds, epochs)
    
    print("Evaluating model...")
    loss, accuracy = evaluate_model(model, test_ds)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    
    # Save the model
    model.save(f'{dataset}_{model_type}_model.h5')
    print(f"Model saved as {dataset}_{model_type}_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handwritten Character Recognition')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'crnn'], help='Model type')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    main(args.dataset, args.model, args.epochs, args.batch_size)
