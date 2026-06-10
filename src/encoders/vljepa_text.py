from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class VLJEPAAlign(nn.Module):
	"""
	視覺-語言對齊模組
	使用 CLIP/T5 + 交叉注意力進行對齊
	"""
	
	def __init__(self, config: Optional[Dict] = None):
		super().__init__()
		self.config = config or {}
		
		# 配置參數
		self.vision_dim = self.config.get('vision_dim', 768)
		self.text_dim = self.config.get('text_dim', 768)
		self.num_heads = self.config.get('num_heads', 8)
		self.dropout = self.config.get('dropout', 0.1)
		self.text_encoder_type = self.config.get('text_encoder_type', 'identity')
		self.vocab_size = self.config.get('vocab_size', 30522)
		
		self._validate_config()
		
		# 文本編碼器
		self.text_encoder = self._init_text_encoder()
		
		# 文本投影層（將文本嵌入投影到視覺維度）
		self.text_proj = nn.Linear(self.text_dim, self.vision_dim)
		
		# 交叉注意力層
		self.cross_attention = nn.MultiheadAttention(
			embed_dim=self.vision_dim,
			num_heads=self.num_heads,
			dropout=self.dropout,
			batch_first=True
		)
		
		# 自注意力層（可選）
		self.self_attention = nn.MultiheadAttention(
			embed_dim=self.vision_dim,
			num_heads=self.num_heads,
			dropout=self.dropout,
			batch_first=True
		)
		
		# 層正規化
		self.norm1 = nn.LayerNorm(self.vision_dim)
		self.norm2 = nn.LayerNorm(self.vision_dim)
		self.norm3 = nn.LayerNorm(self.vision_dim)
		
		# 前饋網路
		self.ffn = nn.Sequential(
			nn.Linear(self.vision_dim, self.vision_dim * 4),
			nn.ReLU(),
			nn.Dropout(self.dropout),
			nn.Linear(self.vision_dim * 4, self.vision_dim)
		)
		
		# 對齊損失層
		self.alignment_head = nn.Sequential(
			nn.Linear(self.vision_dim, 512),
			nn.ReLU(),
			nn.Dropout(self.dropout),
			nn.Linear(512, 1)
		)
	
	def _validate_config(self) -> None:
		"""驗證配置參數"""
		assert self.vision_dim > 0, "vision_dim 必須 > 0"
		assert self.text_dim > 0, "text_dim 必須 > 0"
		assert self.vision_dim % self.num_heads == 0, \
			f"vision_dim ({self.vision_dim}) 必須能被 num_heads ({self.num_heads}) 整除"
	
	def _init_text_encoder(self) -> nn.Module:
		"""初始化文本編碼器"""
		if self.text_encoder_type == 'identity':
			return nn.Identity()
		elif self.text_encoder_type == 'embedding':
			return nn.Embedding(self.vocab_size, self.text_dim)
		elif self.text_encoder_type == 'lstm':
			return nn.LSTM(
				input_size=self.text_dim,
				hidden_size=self.text_dim,
				num_layers=2,
				batch_first=True,
				bidirectional=True
			)
		else:
			raise ValueError(f"未知的文本編碼器類型: {self.text_encoder_type}")
	
	def forward(self, 
				vision_embed: torch.Tensor,
				text_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		視覺-語言對齊
		
		Args:
			vision_embed: (B, T, D_v) - 視覺嵌入
			text_tokens: (B, L) - 文本 token（可選）
		
		Returns:
			aligned_vision: (B, T, D) - 對齊後的視覺
			aligned_text: (B, L, D) - 對齊後的文本
		"""
		batch_size, seq_len, _ = vision_embed.shape
		
		# 編碼文本
		if text_tokens is not None:
			text_embed = self._encode_text(text_tokens)
		else:
			text_embed = torch.zeros(
				batch_size, seq_len, self.text_dim,
				device=vision_embed.device,
				dtype=vision_embed.dtype
			)
		
		# 投影文本到視覺維度
		text_proj = self.text_proj(text_embed)  # (B, L, D_v)
		
		# 交叉注意力對齊
		cross_attn_out, _ = self.cross_attention(
			vision_embed, text_proj, text_proj
		)
		
		# 殘差連接 + 層正規化
		vision_aligned = self.norm1(vision_embed + cross_attn_out)
		
		# 自注意力
		self_attn_out, _ = self.self_attention(
			vision_aligned, vision_aligned, vision_aligned
		)
		vision_aligned = self.norm2(vision_aligned + self_attn_out)
		
		# 前饋網路
		ffn_out = self.ffn(vision_aligned)
		vision_aligned = self.norm3(vision_aligned + ffn_out)
		
		return vision_aligned, text_proj
	
	def _encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
		"""編碼文本 token"""
		if self.text_encoder_type == 'identity':
			return text_tokens.float()
		elif self.text_encoder_type == 'embedding':
			return self.text_encoder(text_tokens)
		elif self.text_encoder_type == 'lstm':
			output, _ = self.text_encoder(text_tokens)
			return output
		else:
			return text_tokens.float()
	
	def forward_with_alignment_score(self,
									vision_embed: torch.Tensor,
									text_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		計算對齊分數
		
		Args:
			vision_embed: (B, T, D_v)
			text_tokens: (B, L)
		
		Returns:
			aligned_vision: (B, T, D)
			aligned_text: (B, L, D)
			alignment_score: (B, T)
		"""
		aligned_vision, aligned_text = self.forward(vision_embed, text_tokens)
		
		# 計算對齊分數
		alignment_score = self.alignment_head(aligned_vision).squeeze(-1)  # (B, T)
		
		return aligned_vision, aligned_text, alignment_score
	
	def compute_contrastive_loss(self,
								vision_embed: torch.Tensor,
								text_embed: torch.Tensor,
								temperature: float = 0.07) -> torch.Tensor:
		"""
		計算對比損失
		
		Args:
			vision_embed: (B, T, D)
			text_embed: (B, L, D)
			temperature: 溫度參數
		
		Returns:
			loss: 標量張量
		"""
		# 正規化
		vision_norm = F.normalize(vision_embed, p=2, dim=-1)  # (B, T, D)
		text_norm = F.normalize(text_embed, p=2, dim=-1)      # (B, L, D)
		
		# 計算相似度
		batch_size = vision_norm.shape[0]
		
		# 平均池化
		vision_avg = vision_norm.mean(dim=1)  # (B, D)
		text_avg = text_norm.mean(dim=1)      # (B, D)
		
		# 計算相似度矩陣
		logits = torch.matmul(vision_avg, text_avg.t()) / temperature  # (B, B)
		
		# 標籤：對角線為正樣本
		labels = torch.arange(batch_size, device=vision_embed.device)
		
		# 對比損失
		loss_v2t = F.cross_entropy(logits, labels)
		loss_t2v = F.cross_entropy(logits.t(), labels)
		
		return (loss_v2t + loss_t2v) / 2
