import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def encode_classes(y_train, y_test):
    """
    Encode symbol classes using LabelEncoder.
    """
    l = LabelEncoder()
    l.fit(y_train)
    y_train = l.transform(y_train)
    y_test = l.transform(y_test)
    return y_train, y_test

    
def load_data(data_file):
    """
    Load and preprocess data from a CSV file, and split it into training and test sets.
    """
    df = pd.read_csv(data_file, names=[i for i in range(1, 10002)])
    
    # Remove symbol classes with less than 7 instances
    df = df[~df.iloc[:, -1].isin(['Ultrasonic Flow Meter', 'Barred Tee','Temporary Strainer', 'Control Valve Angle Choke','Line Blindspacer','Vessel','Valve Gate Through Conduit','Deluge','Control Valve'])]

    X = df.values[:, :-1]
    y = df.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    X_train = X_train.reshape(X_train.shape[0],100,100,1)
    X_test = X_test.reshape(X_test.shape[0],100,100,1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    y_train, y_test = encode_classes(y_train,y_test) 

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Validation set
    for _ in range(5): 
        idx = np.random.permutation(len(X_train))

    X_train = X_train[idx]
    y_train = y_train[idx]

    val_size = 0.10
    count = int(val_size * len(X_train))

    val_X = X_train[:count,:]
    val_y = y_train[:count,:]

    return X_train, X_test, y_train, y_test, val_X, val_y