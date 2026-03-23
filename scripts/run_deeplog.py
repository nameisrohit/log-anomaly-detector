import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load data ──
print("Loading data...")
df = pd.read_parquet("data/processed/hdfs_sequences.parquet")
with open("data/processed/template_map.json", 'r') as f:
    template_map = json.load(f)

num_events = max(e for seq in df['sequence'] for e in seq) + 1
normal_df = df[df['label'] == 'Normal']
anomaly_df = df[df['label'] == 'Anomaly']
print(f"Normal: {len(normal_df):,} | Anomaly: {len(anomaly_df):,}")

# ── Sliding windows ──
WINDOW_SIZE = 10

def create_windows(sequences, window_size):
    X, y = [], []
    for seq in sequences:
        if len(seq) <= window_size:
            continue
        for i in range(len(seq) - window_size):
            X.append(seq[i:i + window_size])
            y.append(seq[i + window_size])
    return np.array(X), np.array(y)

train_seqs, test_normal_seqs = train_test_split(
    normal_df['sequence'].tolist(), test_size=0.2, random_state=42
)

X_train, y_train = create_windows(train_seqs, WINDOW_SIZE)
print(f"Training windows: {len(X_train):,}")

# ── Dataset ──
class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(LogDataset(X_train, y_train), batch_size=1024, shuffle=True)

# ── Model ──
class DeepLog(nn.Module):
    def __init__(self, num_events, hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_events, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_events)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

model = DeepLog(num_events=num_events).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Train ──
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
for epoch in range(10):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        loss = criterion(out, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == by).sum().item()
        total += by.size(0)
    print(f"  Epoch {epoch+1}/10 | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

torch.save(model.state_dict(), 'models/checkpoints/deeplog_v1.pt')
print("\nModel saved!")

# ── Detect anomalies ──
def detect(model, sequences, window_size, top_k, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in sequences:
            if len(seq) <= window_size:
                preds.append(1)
                continue
            windows = [seq[i:i+window_size] for i in range(len(seq)-window_size)]
            targets = [seq[i+window_size] for i in range(len(seq)-window_size)]
            out = model(torch.LongTensor(windows).to(device))
            _, topk = torch.topk(out, top_k, dim=1)
            topk = topk.cpu().numpy()
            is_anom = any(targets[j] not in topk[j] for j in range(len(targets)))
            preds.append(1 if is_anom else 0)
    return np.array(preds)

TOP_K = 9
sample_normal = test_normal_seqs[:10_000]
sample_anomaly = anomaly_df['sequence'].tolist()[:5_000]

print(f"\nDetecting anomalies (K={TOP_K})...")
print(f"  Testing: {len(sample_normal):,} normal + {len(sample_anomaly):,} anomaly")

pred_n = detect(model, sample_normal, WINDOW_SIZE, TOP_K, device)
print(f"  Normal flagged: {pred_n.sum():,}/{len(pred_n):,}")

pred_a = detect(model, sample_anomaly, WINDOW_SIZE, TOP_K, device)
print(f"  Anomaly flagged: {pred_a.sum():,}/{len(pred_a):,}")

y_true = np.array([0]*len(pred_n) + [1]*len(pred_a))
y_pred = np.concatenate([pred_n, pred_a])

print(f"\n{'='*50}")
print(f"DeepLog Results (Top-K={TOP_K})")
print(f"{'='*50}")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))