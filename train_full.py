"""
完整的 GPU 訓練腳本
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
from pathlib import Path

sys.path.insert(0, '.')

print("=" * 80)
print("Intent-VOE-Collab GPU 訓練")
print("=" * 80)
print()

# ============================================================================
# 1. 設備設置
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ 設備: {device}")
print(f"✓ PyTorch 版本: {torch.__version__}")

if torch.cuda.is_available():
	print(f"✓ GPU 數量: {torch.cuda.device_count()}")
	print(f"✓ GPU 名稱: {torch.cuda.get_device_name(0)}")
	print(f"✓ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
	print(f"✓ CUDA 版本: {torch.version.cuda}")
print()

# ============================================================================
# 2. 配置
# ============================================================================
config = {
	'hidden_dim': 768,
	'num_layers': 6,
	'input_dim': 768,
	'action_dim': 7,
	'batch_size': 16,
	'epochs': 5,
	'lr': 1e-4,
	'device': device,
	'num_workers': 4,  # GPU 可以用多個 workers
}

print("訓練配置:")
for key, value in config.items():
	if key != 'device':
		print(f"  {key}: {value}")
print()

# ============================================================================
# 3. 簡單的數據集
# ============================================================================
class DummyDataset(Dataset):
	"""虛擬數據集用於測試"""
	def __init__(self, size=100):
		self.size = size
	
	def __len__(self):
		return self.size
	
	def __getitem__(self, idx):
		frames = torch.randn(10, 3, 224, 224)  # (T, C, H, W)
		actions = torch.randn(10, 7)            # (T, A)
		language = torch.randn(10, 768)         # (T, D)
		return {
			'frames': frames,
			'actions': actions,
			'language': language
		}

# ============================================================================
# 4. 模型定義
# ============================================================================
class VJEPA2Backbone(nn.Module):
	"""視覺骨幹"""
	def __init__(self, hidden_dim=768):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(3*224*224, 2048),
			nn.ReLU(),
			nn.Linear(2048, hidden_dim),
		)
	
	def forward(self, frames):
		# frames: (B, T, C, H, W)
		B, T = frames.shape[:2]
		frames = frames.view(B, T, -1)  # (B, T, C*H*W)
		embeddings = self.encoder(frames)  # (B, T, D)
		return embeddings

class ActionConditionedPredictor(nn.Module):
	"""動作預測器"""
	def __init__(self, input_dim=768, action_dim=7, num_layers=6):
		super().__init__()
		self.transformer = nn.TransformerEncoder(
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=input_dim,
				nhead=8,
				dim_feedforward=2048,
				batch_first=True,
				dropout=0.1
			),
			num_layers=num_layers
		)
		self.action_head = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim)
		)
	
	def forward(self, embeddings):
		# embeddings: (B, T, D)
		encoded = self.transformer(embeddings)
		actions = self.action_head(encoded)
		return actions

class ThreeStreamFusion(nn.Module):
	"""三流融合"""
	def __init__(self, dim=768):
		super().__init__()
		self.vision_proj = nn.Linear(768, 256)
		self.action_proj = nn.Linear(7, 256)
		self.language_proj = nn.Linear(768, 256)
		
		self.fusion = nn.MultiheadAttention(
			embed_dim=256,
			num_heads=8,
			batch_first=True
		)
	
	def forward(self, vision, actions, language):
		v_proj = self.vision_proj(vision)
		a_proj = self.action_proj(actions)
		l_proj = self.language_proj(language)
		
		fused, _ = self.fusion(v_proj, a_proj, l_proj)
		return fused

# ============================================================================
# 5. 訓練函數
# ============================================================================
def train():
	print("建立模型...")
	encoder = VJEPA2Backbone(config['hidden_dim']).to(device)
	predictor = ActionConditionedPredictor(
		input_dim=config['input_dim'],
		action_dim=config['action_dim'],
		num_layers=config['num_layers']
	).to(device)
	fusion = ThreeStreamFusion().to(device)
	
	print("✓ 模型已建立")
	print()
	
	# 計算參數數量
	total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
	total_params += sum(p.numel() for p in predictor.parameters() if p.requires_grad)
	total_params += sum(p.numel() for p in fusion.parameters() if p.requires_grad)
	print(f"✓ 總參數數量: {total_params:,}")
	print()
	
	# 優化器
	optimizer = torch.optim.Adam(
		list(encoder.parameters()) + 
		list(predictor.parameters()) + 
		list(fusion.parameters()),
		lr=config['lr']
	)
	
	# 損失函數
	criterion = nn.MSELoss()
	
	# 數據加載器
	print("建立數據加載器...")
	dataset = DummyDataset(size=100)
	train_loader = DataLoader(
		dataset,
		batch_size=config['batch_size'],
		shuffle=True,
		num_workers=config['num_workers'],
		pin_memory=True  # GPU 優化
	)
	print(f"✓ 數據加載器已建立 (batch_size={config['batch_size']})")
	print()
	
	# 建立 checkpoints 目錄
	os.makedirs('checkpoints', exist_ok=True)
	
	# 訓練循環
	print("開始訓練...")
	print("-" * 80)
	
	best_loss = float('inf')
	
	for epoch in range(config['epochs']):
		total_loss = 0
		num_batches = 0
		
		for batch_idx, batch in enumerate(train_loader):
			# 移動數據到 GPU
			frames = batch['frames'].to(device)
			actions = batch['actions'].to(device)
			language = batch['language'].to(device)
			
			# 前向傳播
			embeddings = encoder(frames)
			pred_actions = predictor(embeddings)
			fused = fusion(embeddings, actions, language)
			
			# 計算損失
			loss = criterion(pred_actions, actions)
			
			# 反向傳播
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(
				list(encoder.parameters()) + 
				list(predictor.parameters()) + 
				list(fusion.parameters()),
				max_norm=1.0
			)
			optimizer.step()
			
			total_loss += loss.item()
			num_batches += 1
			
			if batch_idx % 2 == 0:
				print(f"Epoch {epoch+1}/{config['epochs']} | "
					  f"Batch {batch_idx+1:3d}/{len(train_loader)} | "
					  f"Loss: {loss.item():.6f}")
		
		avg_loss = total_loss / num_batches
		print(f"Epoch {epoch+1} 完成 | 平均損失: {avg_loss:.6f}")
		
		# 保存最佳模型
		if avg_loss < best_loss:
			best_loss = avg_loss
			torch.save(encoder.state_dict(), 'checkpoints/encoder_best.pt')
			torch.save(predictor.state_dict(), 'checkpoints/predictor_best.pt')
			torch.save(fusion.state_dict(), 'checkpoints/fusion_best.pt')
			print(f"✓ 最佳模型已保存 (損失: {best_loss:.6f})")
		
		print()
	
	print("-" * 80)
	print("✓ 訓練完成！")
	print()
	
	# 保存最終模型
	torch.save(encoder.state_dict(), 'checkpoints/encoder_final.pt')
	torch.save(predictor.state_dict(), 'checkpoints/predictor_final.pt')
	torch.save(fusion.state_dict(), 'checkpoints/fusion_final.pt')
	
	print("✓ 模型已保存:")
	print("  - checkpoints/encoder_final.pt")
	print("  - checkpoints/predictor_final.pt")
	print("  - checkpoints/fusion_final.pt")
	print()

# ============================================================================
# 6. 執行訓練
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
