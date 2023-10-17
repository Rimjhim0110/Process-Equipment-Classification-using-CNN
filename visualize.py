import matplotlib.pyplot as plt

def visualize_loss(results):
    """
    Function to visualize loss
    Plots the training and validation loss over epochs

    Parameters
    ----------
    results: A Keras history object containing loss information from model training
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results.history['loss'], color='b', label="Training loss")
    ax.plot(results.history['val_loss'], color='r', label="Validation loss")

    ax.set_xlabel('EPOCHS', fontsize=10, color='black')
    ax.set_ylabel('LOSS', fontsize=10, color='black')
    ax.legend()

    plt.title('Training and Validation Loss')
    plt.show()


def visualize_accuracy(results):
    """
    Function to visualize accuracy
    Plots the training and validation accuracy over epochs

    Parameters
    ----------
    results: A Keras history object containing accuracy information from model training
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results.history['accuracy'], color='b', label="Training accuracy")
    ax.plot(results.history['val_accuracy'], color='r', label="Validation accuracy")

    ax.set_xlabel('EPOCHS', fontsize=10, color='black')
    ax.set_ylabel('ACCURACY', fontsize=10, color='black')
    ax.legend()

    plt.title('Training and Validation Accuracy')
    plt.show()