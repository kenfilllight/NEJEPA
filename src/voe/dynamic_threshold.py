# src/voe/dynamic_threshold.py

"""
M5: 動態閾值計算
滑動窗口 + 分位數方法
"""
import torch
import numpy as np
from collections import deque
from typing import Dict

class DynamicThreshold:
	def __init__(self, config: Dict):
		self.window_size = config.get('window_size', 10)
		self.quantile = config.get('quantile', 0.95)
		self.deviation_history = deque(maxlen=self.window_size)
		self.current_threshold = 0.5
	
	def update(self, deviation: torch.Tensor) -> float:
		"""
		更新動態閾值
		
		Args:
			deviation: (B, T) 偏差分數
		Returns:
			threshold: 動態閾值
		"""
		# 添加到歷史
		self.deviation_history.append(deviation.detach().cpu().numpy())
		
		# 計算分位數
		if len(self.deviation_history) > 0:
			all_deviations = np.concatenate(self.deviation_history)
			self.current_threshold = float(np.quantile(all_deviations, self.quantile))
		else:
			self.current_threshold = 0.5
		
		return self.current_threshold
	
	def is_violation(self, deviation: torch.Tensor) -> torch.Tensor:
		"""
		判斷是否違反 VOE
		
		Args:
			deviation: (B, T) 偏差分數
		Returns:
			violations: (B, T) 布爾張量，True 表示違反
		"""
		violations = deviation > self.current_threshold
		return violations
