import torch
import torch.nn as nn

class ActionConditionedPredictor(nn.Module):
	def __init__(self, config):
		super().__init__()
		
		# 確保是整數
		if isinstance(config, dict):
			hidden_dim = config.get('hidden_dim', 768)
			num_layers = config.get('num_layers', 6)
			action_dim = config.get('action_dim', 7)
		else:
			hidden_dim = 768
			num_layers = 6
			action_dim = 7
		
		self.hidden_dim = int(hidden_dim)
		self.num_layers = int(num_layers)
		self.action_dim = int(action_dim)
		
		print(f"DEBUG ActionConditionedPredictor: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, action_dim={self.action_dim}")
		
		# Vision projection
		vision_flat_dim = 150528  # 3 * 224 * 224
		self.vision_proj = nn.Linear(vision_flat_dim, self.hidden_dim)
		
		# Action embedding
		self.action_embed = nn.Linear(self.action_dim, self.hidden_dim)
		
		# Transformer
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=self.hidden_dim,
			nhead=8,
			dim_feedforward=2048,
			batch_first=True,
			dropout=0.1
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
		
		# Prediction head
		self.action_head = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.action_dim)
		)
	
	def forward(self, vision_embed, action_embed=None):
		# Handle 5D input (B, T, C, H, W)
		if vision_embed.dim() == 5:
			B, T, C, H, W = vision_embed.shape
			vision_embed = vision_embed.view(B, T, -1)
			vision_embed = self.vision_proj(vision_embed)
		
		# Handle 3D input (B, T, D)
		elif vision_embed.dim() == 3:
			B, T, D = vision_embed.shape
			if D != self.hidden_dim:
				vision_embed = self.vision_proj(vision_embed)
		
		# Fuse with action if provided
		if action_embed is not None:
			action_feat = self.action_embed(action_embed)
			combined = vision_embed + action_feat
		else:
			combined = vision_embed
		
		# Transformer encoding
		encoded = self.transformer(combined)
		
		# Predict actions
		pred_actions = self.action_head(encoded)
		
		return pred_actions
