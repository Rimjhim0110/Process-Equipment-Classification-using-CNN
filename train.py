import numpy as np
from preprocessing import load_data
from model import CNNmodel

def train_model(data_file):
    """
    Train a Convolutional Neural Network (CNN) model for symbol classification

    Parameters:
    data_file (str): Path to the CSV data file

    Returns:
    model (Keras model): Trained CNN model
    """
    X_train, X_test, y_train, y_test, val_X, val_y = load_data(data_file)
    
    num_classes = y_train.shape[1]
    model = CNNmodel(num_classes)
    
    results = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(val_X, val_y))
    score = model.evaluate(X_train, y_train, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])

    return model
