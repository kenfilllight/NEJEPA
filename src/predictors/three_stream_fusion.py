from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeStreamFusion(nn.Module):
	"""
	三流融合主模組
	融合視覺、動作、語言三個流
	"""
	def __init__(self, config: Optional[Dict] = None):
		super().__init__()
		self.config = config or {}
		
		# 配置參數
		self.vision_dim = self.config.get('vision_dim', 768)
		self.action_dim = self.config.get('action_dim', 7)
		self.language_dim = self.config.get('language_dim', 768)
		self.fusion_dim = self.config.get('fusion_dim', 256)
		self.num_heads = self.config.get('num_heads', 8)
		self.dropout = self.config.get('dropout', 0.1)
		
		self._validate_config()
		
		# 三個流的投影層
		self.vision_proj = nn.Linear(self.vision_dim, self.fusion_dim)
		self.action_proj = nn.Linear(self.action_dim, self.fusion_dim)
		self.language_proj = nn.Linear(self.language_dim, self.fusion_dim)
		
		# 融合層 - 多頭注意力
		self.fusion = nn.MultiheadAttention(
			embed_dim=self.fusion_dim,
			num_heads=self.num_heads,
			dropout=self.dropout,
			batch_first=True
		)
		
		# 後融合層
		self.post_fusion = nn.Sequential(
			nn.Linear(self.fusion_dim, self.fusion_dim * 2),
			nn.ReLU(),
			nn.Dropout(self.dropout),
			nn.Linear(self.fusion_dim * 2, self.fusion_dim)
		)
		
		# 層正規化
		self.norm1 = nn.LayerNorm(self.fusion_dim)
		self.norm2 = nn.LayerNorm(self.fusion_dim)
	
	def _validate_config(self) -> None:
		"""驗證配置參數"""
		assert self.vision_dim > 0, "vision_dim 必須 > 0"
		assert self.action_dim > 0, "action_dim 必須 > 0"
		assert self.language_dim > 0, "language_dim 必須 > 0"
		assert self.fusion_dim > 0, "fusion_dim 必須 > 0"
		assert self.fusion_dim % self.num_heads == 0, \
			f"fusion_dim ({self.fusion_dim}) 必須能被 num_heads ({self.num_heads}) 整除"
	
	def forward(self,
				vision: torch.Tensor,
				actions: torch.Tensor,
				language: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		融合視覺、動作、語言三個流
		
		Args:
			vision: (B, T, 768) - 視覺嵌入
			actions: (B, T, 7) - 動作序列
			language: (B, L, 768) - 語言嵌入（可選）
		
		Returns:
			fused: (B, T, 256) - 融合結果
		"""
		# 投影到共同空間
		v_proj = self.vision_proj(vision)      # (B, T, 256)
		a_proj = self.action_proj(actions)     # (B, T, 256)
		
		# 處理語言流
		if language is not None:
			l_proj = self.language_proj(language)  # (B, L, 256)
		else:
			l_proj = torch.zeros_like(v_proj)      # (B, T, 256)
		
		# 多頭注意力融合
		# Query: 視覺, Key: 動作, Value: 語言
		attn_output, _ = self.fusion(v_proj, a_proj, l_proj)
		
		# 殘差連接 + 層正規化
		fused = self.norm1(v_proj + attn_output)
		
		# 後融合層
		post_output = self.post_fusion(fused)
		
		# 殘差連接 + 層正規化
		output = self.norm2(fused + post_output)
		
		return output
	
	def forward_with_attention(self,
							  vision: torch.Tensor,
							  actions: torch.Tensor,
							  language: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		融合並返回注意力權重
		
		Args:
			vision: (B, T, 768)
			actions: (B, T, 7)
			language: (B, L, 768)
		
		Returns:
			fused: (B, T, 256)
			attention_weights: (B, num_heads, T, T)
		"""
		v_proj = self.vision_proj(vision)
		a_proj = self.action_proj(actions)
		
		if language is not None:
			l_proj = self.language_proj(language)
		else:
			l_proj = torch.zeros_like(v_proj)
		
		# 獲取注意力權重
		attn_output, attn_weights = self.fusion(v_proj, a_proj, l_proj)
		
		fused = self.norm1(v_proj + attn_output)
		post_output = self.post_fusion(fused)
		output = self.norm2(fused + post_output)
		
		return output, attn_weights
