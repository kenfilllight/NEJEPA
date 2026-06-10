"""早期意圖準確率評估 (30/50/70% 觀測)"""
import torch
from typing import Dict

class EarlyIntentAccuracy:
	def __init__(self):
		self.thresholds = [0.3, 0.5, 0.7]
	
	def compute(self, 
				predictions: torch.Tensor,
				ground_truth: torch.Tensor) -> Dict[float, float]:
		"""
		計算不同觀測比例下的準確率
		
		Args:
			predictions: (B, T, num_intents)
			ground_truth: (B, T)
		Returns:
			accuracies: {0.3: acc, 0.5: acc, 0.7: acc}
		"""
		results = {}
		for threshold in self.thresholds:
			obs_steps = int(predictions.shape[1] * threshold)
			pred_intent = predictions[:, :obs_steps].argmax(dim=-1)
			acc = (pred_intent == ground_truth[:, :obs_steps]).float().mean()
			results[threshold] = acc.item()
		
		return results
