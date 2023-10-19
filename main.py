from train import train_model
from eval import visualize, evaluate_model
  
if __name__ == "__main__":
    data_file = '/content/drive/MyDrive/Symbols_pixel.csv'

    # Train the model
    model = train_model(data_file)

    # Visualize and evaluate the model
    visualize(results)  
    evaluate_model(model, X_test, y_test)
