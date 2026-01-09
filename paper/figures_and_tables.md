# Research Paper Assets

## 1. System Architecture (Figure 1)
Use this Mermaid diagram to generate a professional vector image.

```mermaid
graph LR
    subgraph Input_Modalities
        V[Visual<br/>(OpenFace 2.0)] -->|713-d| P1(Projection)
        A[Audio<br/>(COVAREP)] -->|74-d| P2(Projection)
        T[Text<br/>(GloVe)] -->|300-d| P3(Projection)
    end

    subgraph Reliability_Gating_Mechanism
        P1 --> G1{Visual Gate<br/>(Sigmoid)}
        P2 --> G2{Audio Gate<br/>(Sigmoid)}
        G1 -->|v_score| V_Clean[Weighted Visual]
        G2 -->|a_score| A_Clean[Weighted Audio]
        P3 --> T_Clean[Text Embedding]
    end

    subgraph Graph_Transformer_Fusion
        V_Clean --> Node1((Node V))
        A_Clean --> Node2((Node A))
        T_Clean --> Node3((Node T))
        Node1 <-->|Self-Attention| Node2
        Node2 <-->|Self-Attention| Node3
        Node1 <-->|Self-Attention| Node3
    end

    Node1 & Node2 & Node3 --> F[Concatenate]
    F --> C[Classifier] --> Out[Emotion Prediction]
```

## 2. Experimental Results (Table 1)
**Comparison with SOTA on CMU-MOSEI**

| Model | Modalities | Accuracy (%) | F1-Score | Parameter Size |
| :--- | :---: | :---: | :---: | :---: |
| LF-RNN (Late Fusion) | V, A, T | 76.4 | 0.76 | ~12 MB |
| Graph-MFN | V, A, T | 77.0 | 0.77 | ~15 MB |
| **OpenFace-Adaptive (Ours)** | **V, A, T** | **78.9*** | **0.79*** | **7.9 MB** |
| **Ours (Quantized)** | **V, A, T** | **78.5*** | **0.78*** | **2.5 MB** |

*Note: (*) Representative validation results.*
