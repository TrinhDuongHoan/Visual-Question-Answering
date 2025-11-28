import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import textwrap

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

    
def unnormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def visualize_vqa_result(image_tensor, question, hyp, ref=None):

    img = unnormalize_image(image_tensor)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    
    wrapper = textwrap.TextWrapper(width=50)
    q_text = "\n".join(wrapper.wrap(f"Q: {question}"))
    p_text = "\n".join(wrapper.wrap(f"Pred: {hyp}"))
    
    text_info = f"{q_text}\n{p_text}"
    title_color = 'blue' 
    
    if ref:
        r_text = "\n".join(wrapper.wrap(f"True: {ref}"))
        text_info += f"\n{r_text}"
        
        if hyp.strip().lower() == ref.strip().lower():
            title_color = 'green'
        else:
            title_color = 'red'
            
    plt.title(text_info, fontsize=11, color=title_color, loc='left', pad=10)
    plt.tight_layout()
    plt.show()

