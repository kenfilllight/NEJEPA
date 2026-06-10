from typing import Union, Optional, Dict
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class SwitchF1(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.average = self.config.get('average', 'weighted')
        self.zero_division = self.config.get('zero_division', 0)
    
    def forward(self,
                predictions: torch.Tensor,
                ground_truth: torch.Tensor) -> float:
        """forward 方法包裝 compute"""
        return self.compute(predictions, ground_truth)
    
    def compute(self,
                predictions: torch.Tensor,
                ground_truth: torch.Tensor) -> float:
        """
        計算意圖切換的 F1 分數
        
        Args:
            predictions: (B, T) - 預測結果
            ground_truth: (B, T) - 真實標籤
        
        Returns:
            f1: F1 分數（Python float）
        """
        # 轉換為 NumPy 並展平
        pred_flat = self._to_numpy(predictions)
        gt_flat = self._to_numpy(ground_truth)
        
        # 驗證輸入
        self._validate_inputs(pred_flat, gt_flat)
        
        # 計算 F1 分數
        f1 = f1_score(
            gt_flat, 
            pred_flat, 
            average=self.average,
            zero_division=self.zero_division
        )
        
        # 明確轉換為 Python float
        return float(f1)
    
    def compute_detailed(self,
                        predictions: torch.Tensor,
                        ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        計算詳細的評估指標
        
        Args:
            predictions: (B, T)
            ground_truth: (B, T)
        
        Returns:
            metrics: 包含 F1, Precision, Recall 的字典
        """
        pred_flat = self._to_numpy(predictions)
        gt_flat = self._to_numpy(ground_truth)
        
        self._validate_inputs(pred_flat, gt_flat)
        
        metrics = {
            'f1': float(f1_score(
                gt_flat, pred_flat,
                average=self.average,
                zero_division=self.zero_division
            )),
            'precision': float(precision_score(
                gt_flat, pred_flat,
                average=self.average,
                zero_division=self.zero_division
            )),
            'recall': float(recall_score(
                gt_flat, pred_flat,
                average=self.average,
                zero_division=self.zero_division
            ))
        }
        
        return metrics
    
    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """將張量轉換為 NumPy 陣列"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy().flatten()
        return np.asarray(tensor).flatten()
    
    @staticmethod
    def _validate_inputs(pred_flat: np.ndarray, 
                        gt_flat: np.ndarray) -> None:
        """驗證輸入"""
        if pred_flat.shape != gt_flat.shape:
            raise ValueError(
                f"預測和真實標籤形狀不匹配: "
                f"{pred_flat.shape} vs {gt_flat.shape}"
            )
        if len(pred_flat) == 0:
            raise ValueError("輸入陣列不能為空")
