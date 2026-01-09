import torch
import torch.nn as nn
import torch.nn.functional as F

class ReliabilityGate(nn.Module):
    """
    NOVELTY: Checks if the input is 'noisy' and scales it down.
    Learns a scalar trust score alpha in [0, 1].
    """
    def __init__(self, input_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Outputs a score 0.0 to 1.0
        )

    def forward(self, x):
        reliability = self.gate_net(x)
        return x * reliability, reliability # Scale feature by its own quality

class OpenFaceAdaptiveNet(nn.Module):
    def __init__(self, use_gate=True, use_transformer=True, hidden_dim=128):
        super().__init__()
        
        self.use_gate = use_gate
        self.use_transformer = use_transformer
        self.d = hidden_dim
        
        # 1. Unimodal Encoders (Project to same dim)
        # Dimensions based on CMU-MOSEI standard features
        self.v_proj = nn.Linear(713, self.d)  # Visual (OpenFace 2.0 full features)
        self.a_proj = nn.Linear(74, self.d)   # Audio (COVAREP)
        self.t_proj = nn.Linear(300, self.d)  # Text (Glove 300d)
        
        # 2. Novel Reliability Gates
        if self.use_gate:
            self.v_gate = ReliabilityGate(self.d)
            self.a_gate = ReliabilityGate(self.d)
        
        # Norm Layers (Added for Stability with Raw Data)
        self.v_norm = nn.BatchNorm1d(713)
        self.a_norm = nn.BatchNorm1d(74)
        self.t_norm = nn.BatchNorm1d(300)
        
        # 3. Cross-Modal Fusion
        if self.use_transformer:
            # Graph Transformer: We treat modalities as tokens in a sequence
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        else:
            # Fallback: Simple MLP Fusion (Concatenation)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.d * 3, self.d),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
        # 4. Classifier
        # Transformer output (flattened) or MLP output
        input_dim = self.d * 3 if self.use_transformer else self.d
        
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 7) # 7 Emotions / Sentiment Classes
        )

    def forward(self, v, a, t):
        # Normalize Raw Inputs first
        v = self.v_norm(v)
        a = self.a_norm(a)
        t = self.t_norm(t)
        
        # Project
        v_emb = F.relu(self.v_proj(v))
        a_emb = F.relu(self.a_proj(a))
        t_emb = F.relu(self.t_proj(t))
        
        # Apply Gating (The "Research" Part)
        if self.use_gate:
            v_clean, v_score = self.v_gate(v_emb)
            a_clean, a_score = self.a_gate(a_emb)
        else:
            # Bypass: Trust is 100%
            v_clean = v_emb
            a_clean = a_emb
            v_score = torch.ones(v.size(0), 1, device=v.device)
            a_score = torch.ones(a.size(0), 1, device=a.device)
        
        # Stack inputs: [Visual, Audio, Text]
        # Text is typically "Trusted" (no gate) or gate is implicit in attention
        
        if self.use_transformer:
            # Shape: (Batch, 3, 128)
            sequence = torch.stack([v_clean, a_clean, t_emb], dim=1)
            # Fuse via Self-Attention
            fused = self.transformer(sequence)
            # Flatten: (Batch, 384)
            fused_flat = fused.flatten(start_dim=1)
        else:
            # Concatenate: (Batch, 384)
            cat_features = torch.cat([v_clean, a_clean, t_emb], dim=1)
            # MLP Fusion
            fused_flat = self.fusion_mlp(cat_features)
        
        # Classify
        output = self.classifier(fused_flat)
        
        return output, v_score, a_score