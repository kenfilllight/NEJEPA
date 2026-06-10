"""
完整的 VOE 訓練腳本 - 修復版本 v2
"""
import os
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TORCH_DISABLE_DYNAMO'] = '1'

import sys
sys.path.insert(0, '.')

import torch
torch._dynamo.config.suppress_errors = True

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.encoders import VJEPA2Backbone
from src.predictors import ActionConditionedPredictor
from src.voe import DeviationScore, DynamicThreshold, CEMIntentSearch

print("=" * 80)
print("Intent-VOE-Collab 完整訓練 v2")
print("=" * 80)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ 設備: {device}")
if torch.cuda.is_available():
	print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
	print(f"✓ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# ============================================================================
# 配置
# ============================================================================
config = {
	'hidden_dim': 768,
	'num_layers': 6,
	'input_dim': 768,
	'action_dim': 7,
	'batch_size': 16,
	'epochs': 5,
	'lr': 0.0001,
	'device': device,
}

print("訓練配置:")
for k, v in config.items():
	if k != 'device':
		print(f"  {k}: {v}")
print()

# ============================================================================
# 數據集
# ============================================================================
class DummyDataset(Dataset):
	def __init__(self, size=100):
		self.size = size
	
	def __len__(self):
		return self.size
	
	def __getitem__(self, idx):
		return {
			'frames': torch.randn(10, 3, 224, 224),
			'actions': torch.randn(10, 7),
		}

# ============================================================================
# 訓練函數
# ============================================================================
def train():
	print("建立模型...")
	
	encoder = VJEPA2Backbone(config).to(device)
	predictor = ActionConditionedPredictor(config).to(device)
	
	# VOE 模組
	deviation_score = DeviationScore().to(device)
	dynamic_threshold = DynamicThreshold(config)
	cem_search = CEMIntentSearch(config)
	
	print("✓ 模型已建立")
	print()
	
	# 計算參數數量
	total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
	total_params += sum(p.numel() for p in predictor.parameters() if p.requires_grad)
	print(f"✓ 總參數數量: {total_params:,}")
	print()
	
	# 優化器
	print("建立優化器...")
	optimizer = torch.optim.SGD(
		list(encoder.parameters()) + list(predictor.parameters()),
		lr=config['lr'],
		momentum=0.9,
		weight_decay=1e-4
	)
	print("✓ 優化器已建立")
	print()
	
	criterion = nn.MSELoss()
	
	# 數據加載器
	print("建立數據加載器...")
	dataset = DummyDataset(size=100)
	train_loader = DataLoader(
		dataset,
		batch_size=config['batch_size'],
		shuffle=True,
		num_workers=0,  # 改為 0，避免多進程問題
		pin_memory=True
	)
	print(f"✓ 數據加載器已建立")
	print(f"  dataset 大小: {len(dataset)}")
	print(f"  train_loader 大小: {len(train_loader)}")
	print()
	
	os.makedirs('checkpoints', exist_ok=True)
	
	print("開始訓練...")
	print("-" * 80)
	
	best_loss = float('inf')
	voe_violations = []
	
	for epoch in range(config['epochs']):
		total_loss = 0
		total_voe_loss = 0
		num_batches = 0
		epoch_violations = 0
		
		for batch_idx, batch in enumerate(train_loader):
			frames = batch['frames'].to(device)
			actions = batch['actions'].to(device)
			
			# 前向傳播
			embeddings = encoder(frames)
			
			# DEBUG: 第一個 batch 打印形狀
			if batch_idx == 0 and epoch == 0:
				print(f"DEBUG: frames shape = {frames.shape}")
				print(f"DEBUG: embeddings shape = {embeddings.shape}")
				print(f"DEBUG: actions shape = {actions.shape}")
			
			pred_actions = predictor(embeddings)
			
			# DEBUG: 第一個 batch 打印形狀
			if batch_idx == 0 and epoch == 0:
				print(f"DEBUG: pred_actions shape = {pred_actions.shape}")
				print()
			
			# VOE 計算
			observed_embeddings = embeddings + torch.randn_like(embeddings) * 0.1
			deviation = deviation_score(embeddings, observed_embeddings)
			
			# 動態閾值
			current_threshold = dynamic_threshold.update(deviation)
			violations = dynamic_threshold.is_violation(deviation)
			epoch_violations += violations.sum().item()
			
			# 損失
			action_loss = criterion(pred_actions, actions)
			
			# VOE 損失
			voe_loss = (violations.float() * deviation).mean()
			
			total_loss_val = action_loss + 0.1 * voe_loss
			
			# 反向傳播
			optimizer.zero_grad()
			total_loss_val.backward()
			torch.nn.utils.clip_grad_norm_(
				list(encoder.parameters()) + list(predictor.parameters()),
				max_norm=1.0
			)
			optimizer.step()
			
			total_loss += total_loss_val.item()
			total_voe_loss += voe_loss.item()
			num_batches += 1
			
			if batch_idx % 2 == 0:
				print(f"Epoch {epoch+1}/{config['epochs']} | "
					  f"Batch {batch_idx+1:3d}/{len(train_loader)} | "
					  f"Loss: {total_loss_val.item():.6f} | "
					  f"VOE Loss: {voe_loss.item():.6f} | "
					  f"Threshold: {current_threshold:.4f}")
		
		# 修復：檢查 num_batches
		if num_batches == 0:
			print(f"ERROR: num_batches = 0")
			print(f"  dataset 大小: {len(dataset)}")
			print(f"  train_loader 大小: {len(train_loader)}")
			break
		
		avg_loss = total_loss / num_batches
		avg_voe_loss = total_voe_loss / num_batches
		
		print(f"✓ Epoch {epoch+1} 完成 | "
			  f"平均損失: {avg_loss:.6f} | "
			  f"平均 VOE 損失: {avg_voe_loss:.6f} | "
			  f"VOE 違反數: {epoch_violations}")
		
		voe_violations.append(epoch_violations)
		
		# 保存最佳模型
		if avg_loss < best_loss:
			best_loss = avg_loss
			torch.save(encoder.state_dict(), 'checkpoints/encoder_best.pt')
			torch.save(predictor.state_dict(), 'checkpoints/predictor_best.pt')
			print(f"  ✓ 最佳模型已保存 (損失: {best_loss:.6f})")
		
		print()
	
	print("-" * 80)
	print("✓ 訓練完成！")
	print()
	
	# 保存最終模型
	torch.save(encoder.state_dict(), 'checkpoints/encoder_final.pt')
	torch.save(predictor.state_dict(), 'checkpoints/predictor_final.pt')
	
	print("✓ 模型已保存:")
	print("  - checkpoints/encoder_final.pt")
	print("  - checkpoints/predictor_final.pt")
	print("  - checkpoints/encoder_best.pt")
	print("  - checkpoints/predictor_best.pt")
	print()
	
	# 統計
	print("訓練統計:")
	print(f"  最佳損失: {best_loss:.6f}")
	print(f"  總 VOE 違反: {sum(voe_violations)}")
	if config['epochs'] > 0:
		print(f"  平均每個 epoch 的違反: {sum(voe_violations) / config['epochs']:.2f}")
	print()

# ============================================================================
# 執行訓練
# ============================================================================
if __name__ == "__main__":
	try:
		train()
		print("=" * 80)
		print("✓ 訓練成功完成！")
		print("=" * 80)
	except Exception as e:
		print(f"✗ 訓練失敗: {e}")
		import traceback
		traceback.print_exc()
