from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

class CEMIntentSearch(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.population_size = self.config.get('population_size', 100)
        self.elite_ratio = self.config.get('elite_ratio', 0.1)
        self.num_iterations = self.config.get('num_iterations', 5)
        self.latent_dim = self.config.get('latent_dim', 256)
        self.temperature = self.config.get('temperature', 1.0)
        
        # 驗證配置
        self._validate_config()
    
    def _validate_config(self) -> None:
        """驗證配置參數"""
        assert 0 < self.elite_ratio < 1, "elite_ratio 必須在 (0, 1) 之間"
        assert self.population_size > 0, "population_size 必須 > 0"
        assert self.num_iterations > 0, "num_iterations 必須 > 0"
        assert self.latent_dim > 0, "latent_dim 必須 > 0"
    
    def forward(self, 
                current_state: torch.Tensor,
                predictor: Optional[nn.Module] = None) -> torch.Tensor:
        """forward 方法包裝 search"""
        return self.search(current_state, predictor)
    
    def search(self, 
               current_state: torch.Tensor,
               predictor: Optional[nn.Module] = None) -> torch.Tensor:
        """
        在潛在空間中搜索最佳意圖
        
        Args:
            current_state: (B, D)
            predictor: 預測模型
        
        Returns:
            best_intent: (B, D)
        """
        batch_size = current_state.shape[0]
        device = current_state.device
        
        # 初始化高斯分佈
        mean = torch.zeros(batch_size, self.latent_dim, device=device)
        std = torch.ones(batch_size, self.latent_dim, device=device)
        
        elite_count = max(1, int(self.population_size * self.elite_ratio))
        
        for iteration in range(self.num_iterations):
            # 採樣
            samples = self._sample(mean, std)  # (pop, batch, latent_dim)
            
            # 評估
            scores = self._evaluate(samples, current_state, predictor)
            
            # 選擇精英
            mean, std = self._update_distribution(
                samples, scores, elite_count, device
            )
        
        return mean
    
    def _sample(self, 
                mean: torch.Tensor, 
                std: torch.Tensor) -> torch.Tensor:
        """採樣"""
        noise = torch.randn(
            self.population_size, *mean.shape,
            device=mean.device
        )
        return noise * std.unsqueeze(0) + mean.unsqueeze(0)
    
    def _evaluate(self,
                  samples: torch.Tensor,
                  current_state: torch.Tensor,
                  predictor: Optional[nn.Module]) -> torch.Tensor:
        """評估樣本"""
        if predictor is None:
            # 預設：最小化距離
            scores = -torch.norm(
                samples - current_state.unsqueeze(0),
                p=2,
                dim=-1
            )
        else:
            pop_size, batch_size = samples.shape[0], samples.shape[1]
            samples_flat = samples.reshape(-1, samples.shape[-1])
            
            with torch.no_grad():
                predictions = predictor(samples_flat)
            
            scores = torch.norm(predictions, p=2, dim=-1)
            scores = scores.reshape(pop_size, batch_size)
        
        return scores
    
    def _update_distribution(self,
                            samples: torch.Tensor,
                            scores: torch.Tensor,
                            elite_count: int,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新分佈參數"""
        _, elite_indices = torch.topk(scores, elite_count, dim=0)
        
        batch_size = samples.shape[1]
        elite_samples = samples[
            elite_indices.squeeze(),
            torch.arange(batch_size, device=device)
        ]
        
        mean = elite_samples.mean(dim=0)
        std = elite_samples.std(dim=0) + 1e-8
        
        return mean, std
