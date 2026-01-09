"""
Generate Confusion Matrix Figure for FG 2026 Paper
Run after experiment_runner.py completes.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from model import OpenFaceAdaptiveNet
from data_loader import MOSEIDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
BATCH_SIZE = 32

EMOTION_LABELS = [
    'Strong Neg', 'Neg', 'Weak Neg', 'Neutral', 
    'Weak Pos', 'Pos', 'Strong Pos'
]

def main():
    # Load Model with auto-detection
    print("Loading trained model...")
    checkpoint_path = "results/model_baseline.pth"
    
    model = None
    for hdim in [256, 192, 128]:
        try:
            model = OpenFaceAdaptiveNet(use_gate=True, use_transformer=True, hidden_dim=hdim).to(DEVICE)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
            print(f"Loaded weights from {checkpoint_path} (hidden_dim={hdim})")
            break
        except (FileNotFoundError, RuntimeError) as e:
            if "size mismatch" in str(e):
                continue
            elif isinstance(e, FileNotFoundError):
                print(f"ERROR: {checkpoint_path} not found. Run experiment_runner.py first!")
                return
    
    if model is None:
        print("ERROR: Could not load model with any hidden_dim")
        return
    
    model.eval()
    
    # Load Data
    print("Loading test data...")
    test_dataset = MOSEIDataset("mosei_data.pkl", split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Predict
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for v, a, t, labels in test_loader:
            v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
            outputs, _, _ = model(v, a, t)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix: OpenFace-Adaptive (7-Class)', fontsize=14)
    plt.tight_layout()
    
    # Save
    output_path = "paper/latexsource/confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved confusion matrix to {output_path}")
    
    # Also print classification report
    print("\n--- Per-Class Performance ---")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, digits=3))

if __name__ == "__main__":
    main()
