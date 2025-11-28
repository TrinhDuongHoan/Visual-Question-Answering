import os
import torch
from tqdm.auto import tqdm
from src.utils.metrics import compute_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.best_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train")
        
        for batch in pbar:
            images = batch["images"].to(self.cfg.DEVICE)
            q_ids  = batch["q_input_ids"].to(self.cfg.DEVICE)
            q_mask = batch["q_attention_mask"].to(self.cfg.DEVICE)
            ans_ids= batch["answer_ids"].to(self.cfg.DEVICE)
            
            self.optimizer.zero_grad()
            logits, targets = self.model(images, q_ids, q_mask, ans_ids)
            
            loss = compute_loss(logits, targets, self.model.vocab.PAD_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_loader)

    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val"):
                images = batch["images"].to(self.cfg.DEVICE)
                q_ids  = batch["q_input_ids"].to(self.cfg.DEVICE)
                q_mask = batch["q_attention_mask"].to(self.cfg.DEVICE)
                ans_ids= batch["answer_ids"].to(self.cfg.DEVICE)
                
                logits, targets = self.model(images, q_ids, q_mask, ans_ids)
                loss = compute_loss(logits, targets, self.model.vocab.PAD_ID)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        os.makedirs(self.cfg.CHECKPOINT_DIR, exist_ok=True)
        no_imp = 0
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.eval_epoch(epoch)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.cfg.EARLY_STOP_PATIENCE:
                    print("Early stopping triggered.")
                    break