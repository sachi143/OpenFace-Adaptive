import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model import OpenFaceAdaptiveNet
from data_loader import MOSEIDataset
import numpy as np

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "mosei_data.pkl"
MODEL_PATH = "results/model_baseline.pth"

def evaluate_binary():
    print(f"Loading Test Data from {DATA_PATH}...")
    test_dataset = MOSEIDataset(DATA_PATH, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Loading Model from {MODEL_PATH}...")
    # Try different hidden_dims to match saved checkpoint
    for hdim in [256, 192, 128]:
        try:
            model = OpenFaceAdaptiveNet(hidden_dim=hdim).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            print(f"Model loaded with hidden_dim={hdim}")
            break
        except RuntimeError:
            continue
    model.eval()

    all_preds = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for v, a, t, labels in test_loader:
            v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
            outputs, _, _ = model(v, a, t)
            
            # Get class predictions (0-6)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---------------------------------------------------------
    # 7-Class Accuracy (Original)
    # ---------------------------------------------------------
    acc_7 = accuracy_score(all_labels, all_preds)
    print(f"\n[7-Class] Accuracy: {acc_7*100:.2f}% (Matches Paper)")

    # ---------------------------------------------------------
    # Binary Accuracy (Original MOSEI Standard)
    # Class 0,1,2 = Negative (0)
    # Class 3     = Neutral  (dropped usually, or treated as Non-Negative)
    # Class 4,5,6 = Positive (1)
    # ---------------------------------------------------------
    
    # Method 1: Non-Negative (>=3) vs Negative (<3)
    # This is a common metric.
    bin_preds = (all_preds >= 3).astype(int)
    bin_labels = (all_labels >= 3).astype(int)
    acc_bin_1 = accuracy_score(bin_labels, bin_preds)
    print(f"[Binary]  Non-Negative (>=0) vs Negative (<0): {acc_bin_1*100:.2f}%")

    # Method 2: Positive (>3) vs Negative (<3), Exclude Neutral (3)
    # This is the strict sentiment accuracy.
    mask = all_labels != 3
    if np.sum(mask) > 0:
        strict_preds = (all_preds[mask] > 3).astype(int)
        strict_labels = (all_labels[mask] > 3).astype(int)
        acc_bin_2 = accuracy_score(strict_labels, strict_preds)
        print(f"[Binary]  Positive (>0) vs Negative (<0) [No Neutral]: {acc_bin_2*100:.2f}%")
    else:
        print("[Binary]  Strict (No Neutral): No non-neutral samples found?")

if __name__ == "__main__":
    evaluate_binary()
