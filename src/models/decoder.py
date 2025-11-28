import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(LSTMDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, init_states):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds, init_states)
        logits = self.linear(lstm_out)
        return logits

    def generate(self, init_states, start_token_id, max_length=20):
        h, c = init_states
        batch_size = h.size(1)
        device = h.device
        
        curr_token = torch.tensor([start_token_id] * batch_size, device=device).unsqueeze(1)
        
        outputs = []
        
        for _ in range(max_length):
            embeds = self.embedding(curr_token) 
            lstm_out, (h, c) = self.lstm(embeds, (h, c))
            
            logits = self.linear(lstm_out.squeeze(1))
            pred_id = logits.argmax(1)
            
            outputs.append(pred_id)
            curr_token = pred_id.unsqueeze(1)
            
        return torch.stack(outputs, dim=1)