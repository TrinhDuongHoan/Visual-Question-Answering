import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_history(history):
    epochs = range(1, len(history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_loss.png")
    plt.show()

    

