"""
模型推理腳本
"""
import torch
import sys

sys.path.insert(0, '.')

print("=" * 80)
print("Intent-VOE-Collab 推理")
print("=" * 80)
print()

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ 設備: {device}")
print()

# 載入模型
print("載入模型...")

from train_full import VJEPA2Backbone, ActionConditionedPredictor, ThreeStreamFusion

encoder = VJEPA2Backbone(768).to(device)
predictor = ActionConditionedPredictor(768, 7, 6).to(device)
fusion = ThreeStreamFusion().to(device)

# 載入權重
encoder.load_state_dict(torch.load('checkpoints/encoder_final.pt', map_location=device))
predictor.load_state_dict(torch.load('checkpoints/predictor_final.pt', map_location=device))
fusion.load_state_dict(torch.load('checkpoints/fusion_final.pt', map_location=device))

encoder.eval()
predictor.eval()
fusion.eval()

print("✓ 模型已載入")
print()

# 推理
print("執行推理...")
with torch.no_grad():
    # 建立虛擬輸入
    frames = torch.randn(2, 10, 3, 224, 224).to(device)
    actions = torch.randn(2, 10, 7).to(device)
    language = torch.randn(2, 10, 768).to(device)
    
    # 前向傳播
    embeddings = encoder(frames)
    pred_actions = predictor(embeddings)
    fused = fusion(embeddings, actions, language)
    
    print(f"✓ 輸入形狀: {frames.shape}")
    print(f"✓ 嵌入形狀: {embeddings.shape}")
    print(f"✓ 預測動作形狀: {pred_actions.shape}")
    print(f"✓ 融合輸出形狀: {fused.shape}")
    print()

print("=" * 80)
print("✓ 推理完成！")
print("=" * 80)
