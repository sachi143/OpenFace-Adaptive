# OpenFace-Adaptive 🎭

**Robust Multimodal Emotion Recognition via Reliability-Aware Gating and Cross-Modal Attention**

> 📄 *Submitted at FG 2026 (under review)*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

OpenFace-Adaptive is a robust multimodal emotion recognition framework designed for **"in-the-wild" deployments** where sensor failures are common. The system introduces:

- **Reliability-Aware Gating (RAG)**: Dynamically assigns trust scores to each modality, suppressing noisy signals
- **Heterogeneous Graph Transformer**: Models cross-modal dependencies with attention-based fusion
- **Edge-Ready Deployment**: Quantized to 2.5 MB for real-time inference on resource-constrained devices

## 📊 Results

| Metric | Score |
|--------|-------|
| **7-Class Accuracy** | 44.0% |
| **Binary Accuracy** | 72.1% |
| **Robustness Gain** | +3.6% under noise |
| **Model Size** | 7.9 MB (2.5 MB quantized) |

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  OpenFace   │    │   COVAREP   │    │   GloVe     │
│  (713-d)    │    │   (74-d)    │    │   (300-d)   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Reliability  │    │ Reliability  │    │              │
│   Gate (αv)  │    │   Gate (αa)  │    │   (Text)     │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
              ┌────────────────────────┐
              │ Heterogeneous Graph    │
              │     Transformer        │
              │   (3 layers, 4 heads)  │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │   7-Class Classifier   │
              └────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/OpenFace-Adaptive.git
cd OpenFace-Adaptive

# Create conda environment
conda env create -f environment.yaml
conda activate openface_adaptive

# Install dependencies
pip install torch transformers librosa pyaudio speechrecognition
```

### Training

```bash
# Run full experiment suite (Baseline, Ablations, Robustness tests)
python experiment_runner.py
```

### Live Demo

```bash
# Real-time emotion recognition with webcam + microphone
python live_demo.py
```

### Evaluation

```bash
# Generate confusion matrix
python generate_confusion_matrix.py

# Verify binary accuracy
python verify_binary.py

# Generate trust score visualization
python plot_trust.py
```

## 📁 Project Structure

```
OpenFace-Adaptive/
├── model.py                 # Core model architecture
├── data_loader.py           # CMU-MOSEI dataset loader
├── experiment_runner.py     # Training & evaluation pipeline
├── live_demo.py             # Real-time webcam demo
├── generate_confusion_matrix.py  # Evaluation visualization
├── verify_binary.py         # Binary accuracy computation
├── plot_trust.py            # Trust score visualization
├── quantize.py              # INT8 quantization script
├── preprocess_mosei.py      # Data preprocessing
├── environment.yaml         # Conda environment
├── results/                 # Model checkpoints
│   └── model_baseline.pth
├── paper/                   # LaTeX source for FG 2026 paper
│   └── latexsource/
│       └── submission.pdf
└── OpenFace_2.2.0/          # OpenFace toolkit (not included)
```

## 📦 Model Weights

| Model | Size | Download |
|-------|------|----------|
| Full Model | 7.9 MB | `results/model_baseline.pth` |
| Quantized (INT8) | 2.5 MB | `openface_adaptive_quantized.pth` |

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{openface_adaptive2026,
  title={OpenFace-Adaptive: Robust Multimodal Emotion Recognition via Reliability-Aware Gating and Cross-Modal Attention},
  author={[Authors]},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
  year={2026}
}
```

## 🔐 Access & Collaboration

This is a **private repository** for research purposes. If you are interested in:
- Accessing the full codebase
- Obtaining pre-trained model weights
- Collaborating on this research

Please contact us via:
- 📧 Email: sairam.chennaka@gmail.com
- 🔗 GitHub: Open an issue on this repository

**For reviewers**: Full code, model weights, and dataset access will be provided upon request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CMU-MOSEI](https://www.amir-zadeh.com/datasets) dataset
- [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) toolkit
- [COVAREP](https://github.com/covarep/covarep) audio features
