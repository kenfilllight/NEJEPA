"""組裝任務數據加載器"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict

class AssemblyDataset(Dataset):
	def __init__(self, config: Dict):
		self.config = config
		self.length = 1000
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, idx):
		frames = torch.randn(10, 3, 224, 224)
		actions = torch.randn(10, 7)
		language = torch.randn(10, 768)
		return {'frames': frames, 'actions': actions, 'language': language}

def get_assembly_loader(config: Dict, split: str = 'train'):
	dataset = AssemblyDataset(config)
	return DataLoader(dataset, batch_size=config.get('batch_size', 32))
