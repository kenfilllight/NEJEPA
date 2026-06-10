"""規劃延遲評估"""
import torch

class PlanningLatency:
	def compute(self,
				detection_time: torch.Tensor,
				true_switch_time: torch.Tensor) -> float:
		"""
		計算平均規劃延遲
		
		Args:
			detection_time: 檢測到意圖切換的時間
			true_switch_time: 真實意圖切換時間
		Returns:
			latency: 平均延遲
		"""
		latency = torch.abs(detection_time - true_switch_time).mean()
		return latency.item()
