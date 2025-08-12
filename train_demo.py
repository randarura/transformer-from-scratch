import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

from transformer import Encoder  # あなたのEncoder（修正版MHA推奨）

# --- ハイパラ ---
vocab_size = 100
d_model = 64        # 可視化しやすく少し上げる
N = 2
heads = 4
d_ff = 128
lr = 1e-3
epochs = 20
seq_len = 20
batch_size = 32
pad_id = 0

# --- モデル ---
enc = Encoder(vocab_size, d_model, N, heads, d_ff)
lm_head = nn.Linear(d_model, vocab_size)           # ★ d_model → vocab
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optim = torch.optim.Adam(list(enc.parameters()) + list(lm_head.parameters()), lr=lr)

# --- ダミーデータ作成（次トークン予測） ---
def make_batch(B, L, V):
    # 先頭に非padトークンを入れて、最後だけpadが混ざる感じの簡易データ
    x = torch.randint(1, V, (B, L))  # 1..V-1
    x[:, 0] = torch.randint(1, V, (B,))  # 先頭は必ず非pad
    y = x.clone()
    # next-token: 入力は 0..L-2、ラベルは 1..L-1（末尾をpad）
    x_in = x[:, :-1]
    y_next = y[:, 1:]
    return x_in, y_next

def accuracy(logits, target):
    # logits: (B, L-1, V), target: (B, L-1)
    with torch.no_grad():
        pred = logits.argmax(-1)
        mask = (target != pad_id)
        if mask.sum() == 0:
            return 0.0
        correct = (pred[mask] == target[mask]).float().mean().item()
        return correct

loss_hist, acc_hist = [], []

for epoch in range(1, epochs+1):
    enc.train(); lm_head.train()
    x, y = make_batch(batch_size, seq_len, vocab_size)  # x:(B,L-1), y:(B,L-1)

    optim.zero_grad()
    h = enc(x)                        # (B, L-1, d_model)
    logits = lm_head(h)               # (B, L-1, V)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    optim.step()

    acc = accuracy(logits, y)
    loss_hist.append(loss.item())
    acc_hist.append(acc)
    print(f"Epoch {epoch:02d} | loss {loss.item():.4f} | acc {acc*100:.2f}%")

# --- 可視化 ---
plt.figure()
plt.plot(loss_hist)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy")
plt.tight_layout()
outdir = Path("runs")
outdir.mkdir(exist_ok=True, parents=True)
plt.savefig(outdir/"loss.png")

plt.figure()
plt.plot(acc_hist)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.tight_layout()
plt.savefig(outdir/"acc.png")
print("Saved plots to runs/loss.png and runs/acc.png")
