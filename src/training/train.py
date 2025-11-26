import torch
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import pandas as pd

def get_optimizer_scheduler(model, dataloader, epochs, lr=1e-4):

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(num_training_steps * 0.1) 

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, scheduler, device):
   
    model.train() 
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        q_ids = batch['question_input_ids'].to(device)
        q_mask = batch['question_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
    
        outputs = model(pixel_values, q_ids, q_mask, labels=labels)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device):
    model.eval() 
    total_loss = 0
    
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            q_ids = batch['question_input_ids'].to(device)
            q_mask = batch['question_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values, q_ids, q_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, checkpoint_path):

    history = {'train_loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Training Loss : {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print("New best model saved.")

    return pd.DataFrame(history)


def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {filename}")
    return model

