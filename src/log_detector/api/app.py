import sys
import os
import re
import json
import time
import asyncio
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from collections import deque
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
TEMPLATE_MAP = BASE_DIR / "data" / "processed" / "template_map.json"
MODEL_PATH = BASE_DIR / "models" / "checkpoints" / "deeplog_v1.pt"
LOG_FILE = BASE_DIR / "data" / "raw" / "HDFS.log"
DASHBOARD_HTML = BASE_DIR / "templates" / "dashboard.html"

# ── DeepLog Model ──
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

# ── Preprocessing ──
def preprocess_message(message):
    msg = re.sub(r'blk_-?\d+', '<BLK>', message)
    msg = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', msg)
    msg = re.sub(r':\d{4,5}', ':<PORT>', msg)
    msg = re.sub(r'/[\w/\.\-_]+', '<PATH>', msg)
    msg = re.sub(r'\b\d+\b', '<NUM>', msg)
    return msg

def parse_log_line(line):
    match = re.match(r'(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w\.\$]+):\s*(.*)', line.strip())
    if not match:
        return None
    date, time_str, pid, level, component, message = match.groups()
    block_match = re.search(r'(blk_-?\d+)', message)
    return {
        'date': date,
        'time': time_str,
        'level': level,
        'component': component,
        'message': message,
        'block_id': block_match.group(1) if block_match else None,
    }

# ── App Setup ──
app = FastAPI(title="SENTINEL.log — Anomaly Detector")

# State
logs = deque(maxlen=200)
anomalies = deque(maxlen=50)
block_sequences = {}
counters = {"total": 0, "anomalies": 0, "errors": 0}
error_history = deque(maxlen=30)
throughput_history = deque(maxlen=30)
connected_clients = []

WINDOW_SIZE = 10
TOP_K = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print(f"Loading model from {MODEL_PATH}...")
with open(TEMPLATE_MAP, 'r') as f:
    template_map = json.load(f)
num_events = max(int(k) for k in template_map.keys()) + 1

model = DeepLog(num_events=num_events).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded! ({sum(p.numel() for p in model.parameters()):,} params)")

# Drain3
drain_config = TemplateMinerConfig()
drain_config.profiling_enabled = False
template_miner = TemplateMiner(config=drain_config)

# ── Anomaly Detection ──
def check_anomaly(sequence):
    if len(sequence) <= WINDOW_SIZE:
        return False
    window = sequence[-WINDOW_SIZE - 1:-1]
    target = sequence[-1]
    with torch.no_grad():
        x = torch.LongTensor([window]).to(device)
        output = model(x)
        _, topk = torch.topk(output, TOP_K, dim=1)
        topk = topk.cpu().numpy()[0]
    return int(target) not in topk

# ── Routes ──
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML.read_text()

@app.get("/api/state")
async def get_state():
    return {
        "logs": list(logs),
        "anomalies": list(anomalies),
        "counters": dict(counters),
        "error_history": list(error_history),
        "throughput_history": list(throughput_history),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def broadcast(data):
    for client in connected_clients[:]:
        try:
            await client.send_json(data)
        except:
            connected_clients.remove(client)

# ── Log Streaming ──
async def stream_logs():
    await asyncio.sleep(2)
    print(f"Starting log stream from {LOG_FILE}...")
    
    with open(LOG_FILE, 'r') as f:
        for i, line in enumerate(f):
            parsed = parse_log_line(line)
            if not parsed:
                continue
            
            clean_msg = preprocess_message(parsed['message'])
            result = template_miner.add_log_message(clean_msg)
            event_id = result['cluster_id']
            
            log_entry = {
                'id': i,
                'time': f"{parsed['time'][:2]}:{parsed['time'][2:4]}:{parsed['time'][4:]}",
                'level': parsed['level'],
                'component': parsed['component'][:20],
                'message': parsed['message'][:100],
                'block_id': parsed['block_id'],
                'event_id': event_id,
                'is_anomaly': False,
            }
            
            if parsed['block_id']:
                bid = parsed['block_id']
                if bid not in block_sequences:
                    block_sequences[bid] = []
                block_sequences[bid].append(event_id)
                
                if len(block_sequences[bid]) > WINDOW_SIZE:
                    is_anom = check_anomaly(block_sequences[bid])
                    if is_anom:
                        log_entry['is_anomaly'] = True
                        counters['anomalies'] += 1
                        anomaly_entry = {
                            'id': i,
                            'time': log_entry['time'],
                            'block_id': bid,
                            'message': parsed['message'][:80],
                            'event_id': event_id,
                            'seq_length': len(block_sequences[bid]),
                        }
                        anomalies.appendleft(anomaly_entry)
                
                if len(block_sequences[bid]) > 50:
                    block_sequences[bid] = block_sequences[bid][-30:]
            
            logs.appendleft(log_entry)
            counters['total'] += 1
            
            if counters['total'] % 10 == 0:
                await broadcast({
                    'type': 'update',
                    'log': log_entry,
                    'counters': dict(counters),
                })
            
            await asyncio.sleep(0.05)
            
            if i > 0 and i % 1000 == 0:
                print(f"  Processed {i:,} logs | Anomalies: {counters['anomalies']}")

@app.on_event("startup")
async def startup():
    asyncio.create_task(stream_logs())