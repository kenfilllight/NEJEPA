"""
模型評估腳本
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, '.')

from train_full import VJEPA2Backbone, ActionConditionedPredictor, DummyDataset

print("=" * 80)
print("Intent-VOE-Collab 評估")
print("=" * 80)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ 設備: {device}")
print()

# 載入模型
print("載入模型...")
encoder = VJEPA2Backbone(768).to(device)
predictor = ActionConditionedPredictor(768, 7, 6).to(device)

encoder.load_state_dict(torch.load('checkpoints/encoder_final.pt', map_location=device))
predictor.load_state_dict(torch.load('checkpoints/predictor_final.pt', map_location=device))

encoder.eval()
predictor.eval()

print("✓ 模型已載入")
print()

# 評估數據
print("建立評估數據...")
test_dataset = DummyDataset(size=50)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0)
print(f"✓ 評估數據已建立 (樣本數: {len(test_dataset)})")
print()

# 評估
print("執行評估...")
criterion = nn.MSELoss()
total_loss = 0
num_batches = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        frames = batch['frames'].to(device)
        actions = batch['actions'].to(device)
        
        embeddings = encoder(frames)
        pred_actions = predictor(embeddings)
        
        loss = criterion(pred_actions, actions)
        total_loss += loss.item()
        num_batches += 1
        
        print(f"Batch {batch_idx+1}/{len(test_loader)} | Loss: {loss.item():.6f}")

avg_loss = total_loss / num_batches
print()
print(f"✓ 平均損失: {avg_loss:.6f}")
print()

print("=" * 80)
print("✓ 評估完成！")
print("=" * 80)
