import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from model import OpenFaceAdaptiveNet
from data_loader import MOSEIDataset

# ==========================================
# FOCAL LOSS (Better for class imbalance)
# ==========================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - focuses on hard examples."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# MIXUP AUGMENTATION
# ==========================================
def mixup_data(v, a, t, labels, alpha=0.2):
    """Apply Mixup augmentation to multimodal data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = v.size(0)
    index = torch.randperm(batch_size).to(v.device)
    
    mixed_v = lam * v + (1 - lam) * v[index]
    mixed_a = lam * a + (1 - lam) * a[index]
    mixed_t = lam * t + (1 - lam) * t[index]
    
    return mixed_v, mixed_a, mixed_t, labels, labels[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  # Lower LR for stability
EPOCHS = 200  # Extended training
HIDDEN_DIM = 192  # Balanced capacity
GRADIENT_CLIP = 1.0  # Gradient clipping
MIXUP_ALPHA = 0.2  # Mixup strength
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "experiment_results.json"

def train_model(model, train_loader, valid_loader, modality_mask=None, epochs=EPOCHS):
    """
    modality_mask: dict {'v': bool, 'a': bool, 't': bool}
    If False, input is zeroed out.
    """
    # Calculate Class Weights to handle imbalance
    print("   Computing Class Weights...")
    all_labels = []
    for _, _, _, lbls in train_loader:
        all_labels.extend(lbls.cpu().numpy())
    
    classes = np.unique(all_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"   Class Weights: {class_weights}")
    
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"   Training for {epochs} epochs on {DEVICE}...")
    
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for v, a, t, labels in train_loader:
            v, a, t, labels = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE), labels.to(DEVICE)
            
            # Apply Mixup Augmentation
            v, a, t, labels_a, labels_b, lam = mixup_data(v, a, t, labels, alpha=MIXUP_ALPHA)
            
            # Apply Modality Masking (for Single Modality Ablations)
            if modality_mask:
                if not modality_mask.get('v', True): v = torch.zeros_like(v)
                if not modality_mask.get('a', True): a = torch.zeros_like(a)
                if not modality_mask.get('t', True): t = torch.zeros_like(t)
            
            optimizer.zero_grad()
            outputs, _, _ = model(v, a, t)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        val_acc, val_f1, _, _ = evaluate_model(model, valid_loader, modality_mask)
        
        # Step Scheduler (cosine annealing - step every epoch)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            
        print(f"   Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, loader, modality_mask=None, noise_injection=None):
    """
    noise_injection: dict {'modality': 'v', 'type': 'zero'|'gaussian'}
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for v, a, t, labels in loader:
            v, a, t, labels = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE), labels.to(DEVICE)
            
            # 1. Training-time Masking (Ablation)
            if modality_mask:
                if not modality_mask.get('v', True): v = torch.zeros_like(v)
                if not modality_mask.get('a', True): a = torch.zeros_like(a)
                if not modality_mask.get('t', True): t = torch.zeros_like(t)
            
            # 2. Test-time Noise Injection (Robustness)
            if noise_injection:
                mod = noise_injection['modality']
                ntype = noise_injection['type']
                
                if mod == 'v': target = v
                elif mod == 'a': target = a
                elif mod == 't': target = t
                
                if ntype == 'zero':
                    if mod == 'v': v = torch.zeros_like(v)
                    elif mod == 'a': a = torch.zeros_like(a)
                    elif mod == 't': t = torch.zeros_like(t)
                elif ntype == 'gaussian':
                    noise = torch.randn_like(target) * 2.0 # High noise
                    if mod == 'v': v = v + noise
                    elif mod == 'a': a = a + noise
                    elif mod == 't': t = t + noise

            outputs, _, _ = model(v, a, t)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1, all_labels, all_preds

def save_confusion_matrix(labels, preds, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {filename}')
    plt.savefig(f"results/cm_{filename}.png")
    plt.close()

def main():
    if not os.path.exists("mosei_data.pkl"):
        print("ERROR: mosei_data.pkl not found. Run preprocess_mosei.py first.")
        return
        
    os.makedirs("results", exist_ok=True)
    
    print("Loading Data...")
    train_dataset = MOSEIDataset("mosei_data.pkl", split='train')
    valid_dataset = MOSEIDataset("mosei_data.pkl", split='valid')
    test_dataset = MOSEIDataset("mosei_data.pkl", split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    experiments = [
        # Name, UseGate, UseTrans, ModalityMask
        ("Baseline_Full", True, True, {'v': True, 'a': True, 't': True}),
        ("No_Gate", False, True, {'v': True, 'a': True, 't': True}),
        ("No_Transformer", True, False, {'v': True, 'a': True, 't': True}),
        ("Visual_Only", True, True, {'v': True, 'a': False, 't': False}),
        ("Audio_Only", True, True, {'v': False, 'a': True, 't': False}),
        ("Text_Only", True, True, {'v': False, 'a': False, 't': True}),
    ]
    
    results = {}
    
    # 1. RUN ABLATIONS
    for name, use_gate, use_trans, mask in experiments:
        print(f"\n[EXPERIMENT] Running: {name}")
        print(f"   Config: Gate={use_gate}, Trans={use_trans}, Mask={mask}")
        
        model = OpenFaceAdaptiveNet(use_gate=use_gate, use_transformer=use_trans, hidden_dim=HIDDEN_DIM).to(DEVICE)
        
        # Train
        model = train_model(model, train_loader, valid_loader, modality_mask=mask)
        
        # Test
        test_acc, test_f1, lbls, preds = evaluate_model(model, test_loader, modality_mask=mask)
        print(f"   RESULT: Acc={test_acc:.2f}%, F1={test_f1:.4f}")
        print("\n   [Fairness Check] Per-Class Performance:")
        print(classification_report(lbls, preds, digits=3))
        
        results[name] = {"Accuracy": test_acc, "F1": test_f1}
        
        # Save Iteratively
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save Confusion Matrix for Baseline
        if name == "Baseline_Full":
            save_confusion_matrix(lbls, preds, "baseline")
            torch.save(model.state_dict(), "results/model_baseline.pth")
            
            # 2. ROBUSTNESS TEST (Only on Baseline)
            print("\n[ROBUSTNESS] Testing Baseline Sensitivity...")
            
            # Test V-Noise
            acc_v_noise, _, _, _ = evaluate_model(model, test_loader, noise_injection={'modality': 'v', 'type': 'gaussian'})
            results["Robustness_Visual_Noise"] = acc_v_noise
            print(f"   Baseline with Visual Noise: {acc_v_noise:.2f}%")
            
             # Test A-Noise
            acc_a_noise, _, _, _ = evaluate_model(model, test_loader, noise_injection={'modality': 'a', 'type': 'gaussian'})
            results["Robustness_Audio_Noise"] = acc_a_noise
            print(f"   Baseline with Audio Noise: {acc_a_noise:.2f}%")

            # Compare with No_Gate Model (Need to train it? No, we will train it in loop)
            # Wait, better to store robustness here.
            
    # Re-run Robustness on "No_Gate" to prove benefit
    # We need to find the No_Gate results or re-train. 
    # Since we are in loop, we can just check if name == "No_Gate" and run robustness on it too.
    
        if name == "No_Gate":
            print("\n[ROBUSTNESS] Testing No_Gate Sensitivity...")
            acc_v_noise_ng, _, _, _ = evaluate_model(model, test_loader, noise_injection={'modality': 'v', 'type': 'gaussian'})
            results["NoGate_Visual_Noise"] = acc_v_noise_ng
            print(f"   No_Gate with Visual Noise: {acc_v_noise_ng:.2f}%")

    # Save Results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAll experiments complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
