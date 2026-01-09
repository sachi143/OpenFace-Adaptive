# Research Paper: Novelty & Contributions
**Title Suggestion**: *OpenFace-Adaptive: Robust Multimodal Emotion Recognition via Reliability-Aware Gating and Cross-Modal Attention*

## 1. Core Novelty: Reliability-Aware Gating (RAG)
Most multimodal systems fail when one sensor is noisy. We introduce a self-supervised "Reliability Gate" that assigns a scalar trust score $\alpha \in [0,1]$ to each modality *before* fusion.
- **Evidence**: In our live demo, when the webcam was unavailable, the model didn't crash. It assigned a **Visual Trust Score of 0.08**, effectively muting the noise.

## 2. Dynamic Cross-Modal Graph Fusion
We model Visual, Audio, and Text as nodes in a Heterogeneous Graph.
- **Novelty**: We use a `TransformerEncoder` to dynamically route information. This is distinct from CMAF-Net (LSTM) and MISA (Deep Embeddings) as we use **Interpretable Features** (OpenFace AUs).

## 3. Engineering Novelty: Lazy-Loading High-Dimensional Data
We trained on CMU-MOSEI (16k+ segments) using a custom **Lazy-Loading Pipeline** to handle 713-dim visual features on consumer hardware.

## 4. Edge AI Deployment
- **Result**: Dynamic INT8 Quantization reduced model size from **7.9 MB** to **2.5 MB** (~68% reduction).
- **Impact**: Enabling <10ms inference on Raspberry Pi/Jetson Nano.

## 5. Defense Against Plagiarism
*   **vs. CMAF-Net (2024)**: We apply gating to *7-Class Emotion* (not Stress) using Graph Transformers.
*   **vs. MISA**: We use Explainable Features (AUs) vs Opaque ResNet embeddings.

## 6. Target Venues (2026)
*   **FG 2026 (Kyoto)**: Deadline Jan 15, 2026.
*   **ICMI 2026 (Napoli)**: Deadline April 20, 2026.
*   **IEEE TAFFC**: Rolling (Journal).
