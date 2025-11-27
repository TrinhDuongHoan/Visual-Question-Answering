import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(LSTMDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Embedding Layer: Chuyển ID từ thành vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM Core
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 3. Output Layer: Dự đoán từ tiếp theo trong từ điển
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, init_states):
        """
        Training mode.
        Args:
            input_ids: IDs của câu trả lời đúng (Teacher Forcing).
            init_states: Tuple (h0, c0) được tạo từ Encoder.
        """
        # Embed câu trả lời
        # Input shape: (Batch, Seq_Len) -> (Batch, Seq_Len, Embed_Dim)
        embeds = self.embedding(input_ids)
        
        # Chạy LSTM
        # output shape: (Batch, Seq_Len, Hidden_Dim)
        lstm_out, _ = self.lstm(embeds, init_states)
        
        # Dự đoán từ
        logits = self.linear(lstm_out)
        return logits

    def generate(self, init_states, start_token_id, max_length=20):
        """
        Inference mode (Sinh từ).
        """
        h, c = init_states
        batch_size = h.size(1)
        device = h.device
        
        # Bắt đầu với token <BOS>
        curr_token = torch.tensor([start_token_id] * batch_size, device=device).unsqueeze(1)
        
        outputs = []
        
        for _ in range(max_length):
            embeds = self.embedding(curr_token) # (Batch, 1, Embed)
            
            # Bước LSTM đơn lẻ
            lstm_out, (h, c) = self.lstm(embeds, (h, c))
            
            # Dự đoán
            logits = self.linear(lstm_out.squeeze(1))
            pred_id = logits.argmax(1)
            
            outputs.append(pred_id)
            curr_token = pred_id.unsqueeze(1)
            
        return torch.stack(outputs, dim=1)