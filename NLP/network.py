import torch.nn as nn
import torch
import torch.nn.functional as F

# Define the model
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
        self.W_q = nn.Parameter(torch.Tensor(
            hidden_dim, hidden_dim))
        self.W_k = nn.Parameter(torch.Tensor(hidden_dim, 1))

        self.W_v = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        
        nn.init.uniform_(self.W_q, -0.1, 0.1)
        nn.init.uniform_(self.W_k, -0.1, 0.1)
        nn.init.uniform_(self.W_v, -0.1, 0.1)

    def forward(self, text):
        # text: (seq_len, batch_size)
        embedded = self.embedding(text)
        # embedded: (seq_len, batch_size, embedding_dim)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output: (seq_len, batch_size, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell: (num_layers, batch_size, hidden_dim)
        x = lstm_output.permute(1, 0, 2)
        # x: (batch_size, seq_len, hidden_dim)
        # Attention过程
        V = x @ self.W_v
        # V 形状是(batch_size, seq_len, hidden_dim)
        # 获得V 向量
        K = self.W_k
        # K 形状是(batch_size, seq_len, 1)
        Q = x @ self.W_q
        # Q 形状是(batch_size, seq_len, hidden_dim)
        att = Q @ K
        # att形状是(batch_size, seq_len, 1)
        # 获取注意力权重
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        # 用注意力权重对V进行线性组合
        scored_x = V.permute(0, 2, 1) @ att_score
        # scored_x形状是(batch_size, hidden_dim)
        # Attention过程结束
        # last_hidden = hidden[-1, :, :]
        # 对特征进行线性组合
        # feat = torch.sum(scored_x, dim=1)
        # feat 
        # attn_out = self.attn(lstm_output, last_hidden)
        # last_hidden: (batch_size, hidden_dim)
        # out = self.fc(self.dropout(last_hidden))
        out = self.fc(scored_x.squeeze(-1))
        # out = self.fc(last_hidden)
        # out: (batch_size, output_dim)
        return out #self.softmax(out)