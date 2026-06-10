from typing import Dict, Optional
import torch
import torch.nn as nn

class DeviationScore(nn.Module):
	def __init__(self, config: Optional[Dict] = None):
		super().__init__()
		self.config = config or {}
		self.metric = 'l2'
	
	def forward(self,
				predicted_embed: torch.Tensor,
				observed_embed: torch.Tensor) -> torch.Tensor:
		"""
		計算預測與觀測之間的偏差
		
		Args:
			predicted_embed: (B, T, D) - 批次大小, 時間步, 嵌入維度
			observed_embed: (B, T, D)
		
		Returns:
			deviation: (B, T) - 每個時間步的偏差分數
		"""
		# L2 距離：沿著嵌入維度計算
		deviation = torch.norm(
			predicted_embed - observed_embed,
			p=2,
			dim=-1
		)
		return deviation
