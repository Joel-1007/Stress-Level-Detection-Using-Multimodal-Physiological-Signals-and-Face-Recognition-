# Stress-Level-Detection-Using-Multimodal-Physiological-Signals-and-Face-Recognition-
Studies on automated emotion recognition from physiological signals are important for healthcare and human-computer interaction. A durable multimodal emotion categorization system using the DEAP dataset is presented in this study.

# 🏗️ Architecture Overview
Model Pipeline
Raw Physiological Signals → Preprocessing → Feature Extraction → 
Multimodal Fusion → CNN-LSTM-Transformer → 4-Quadrant Classification
Emotion Quadrants

HVHA (High Valence, High Arousal): Excitement
HVLA (High Valence, Low Arousal): Contentment
LVHA (Low Valence, High Arousal): Stress/Anger
LVLA (Low Valence, Low Arousal): Sadness

# 📊 Dataset
DEAP Dataset (Database for Emotion Analysis using Physiological signals)

32 participants
40 one-minute music video trials per participant
Multi-channel physiological recordings:

32-channel EEG
Galvanic Skin Response (GSR)
Blood Volume Pulse (BVP)
Respiration
Skin Temperature
Electromyography (EMG)

Download: DEAP Dataset Official Website

# 🚀 Installation
Prerequisites

Python 3.8 or higher
CUDA-compatible GPU 

### Required Packages
```txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
mne>=1.5.0
cvxopt>=1.3.0
pyedflib>=0.1.30
```

# Model Architecture
Hybrid CNN-LSTM-Transformer:

CNN Block: Extracts local spatial patterns (1D convolutions)
Bidirectional LSTM: Captures temporal dependencies
Transformer Encoder: Models global long-range relationships with multi-head attention
Dense Layers: Final classification with dropout and L2 regularization

## 📈 Results

### Performance Metrics

| Model | Test Accuracy | Parameters | Inference Speed |
|-------|--------------|------------|-----------------|
| **Hybrid (Ours)** | **91.8%** | 544K | **898 samples/s** |
| LSTM Baseline | 90.5% | 36K | 726 samples/s |
| GRU Baseline | 89.5% | 29K | 792 samples/s |

### Per-Class Performance (Hybrid Model)

| Emotion Quadrant | Recall |
|------------------|--------|
| HVHA (Excitement) | 96% |
| HVLA (Contentment) | 91% |
| LVHA (Stress) | 88% |
| LVLA (Sadness) | **96%** |

**Key Achievement:** Our model significantly outperforms baselines on the challenging LVLA class (96% vs 83% recall).


