import matplotlib.pyplot as plt

def visualize(results):
    """
    Visualize the loss and accuracy curves during training and validation

    Parameters:
    results (Keras history object): Contains accuracy and loss information from model training and validation
    """
    fig, ax = plt.subplots(1,2, figsize=(10, 5))

    ax[0].plot(results.history['loss'], color='b', label="Training loss")
    ax[0].plot(results.history['val_loss'], color='r', label="Validation loss", axes=ax[0])

    ax[0].set_xlabel('EPOCHS', fontsize=10, color='black')
    ax[0].set_ylabel('LOSS', fontsize=10, color='black')
    ax[0].legend()

    ax[1].plot(results.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(results.history['val_accuracy'], color='r', label="Validation accuracy")

    ax[1].set_xlabel('EPOCHS', fontsize=10, color='black')
    ax[1].set_ylabel('ACCURACY', fontsize=10, color='black')
    ax[1].legend()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print the test accuracy

    Parameters:
    model (Keras model): Trained model
    X_test (ndarray): Test input data
    y_test (ndarray): Test labels
    """
    y_pred = model.predict(X_test)
    accuracy = sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / X_test.shape[0]
    print('Test Accuracy:', accuracy)
