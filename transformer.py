import torch
import torch.nn as nn
import math

# --- Scaled Dot-Product Attention ---
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask=None):
        bs = Q.size(0)
        Q = self.linear_Q(Q).view(bs, -1, self.heads, self.d_k).transpose(1,2)
        K = self.linear_K(K).view(bs, -1, self.heads, self.d_k).transpose(1,2)
        V = self.linear_V(V).view(bs, -1, self.heads, self.d_k).transpose(1,2)

        scores, attn = self.attn(Q, K, V, mask)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.heads*self.d_k)
        output = self.fc(concat)
        return self.norm(Q.transpose(1,2).contiguous().view(bs, -1, self.heads*self.d_k) + self.dropout(output)), attn

# --- Position-wise FeedForward ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.dropout(self.fc2(torch.relu(self.fc1(x)))))

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # batch次元
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# --- Encoder Layer ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x, _ = self.attn(x, x, x, mask)
        x = self.ff(x)
        return x

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])

    def forward(self, src, mask=None):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
