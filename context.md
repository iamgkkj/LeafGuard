# LeafGuard AI - Project Context

## Project Overview
LeafGuard AI is a research project focused on multi-class crop leaf disease identification using deep learning. The project implements and compares a standard MobileNetV3-Small baseline against a CBAM (Convolutional Block Attention Module) enhanced version.

## Project Goals
- [x] Implement MobileNetV3-Small baseline model for plant disease classification
- [x] Implement CBAM attention module enhancement
- [x] Train both models on 38-class plant disease dataset
- [x] Evaluate performance metrics (Accuracy, Macro-F1, Precision, Recall)
- [x] Generate comprehensive visualizations (confusion matrices, ROC curves, PR curves)
- [x] Implement Grad-CAM for model explainability
- [x] Benchmark efficiency (FLOPs, parameters, inference speed)
- [x] Create comprehensive documentation (README.md)
- [x] Export trained models for deployment

## Key Specifications
- **Dataset:** Kaggle New Plant Diseases Dataset (38 classes, ~70K images)
- **Architecture:** MobileNetV3-Small with optional CBAM attention
- **Input Size:** 224×224 RGB images
- **Training:** 20 epochs with early stopping (patience=5)
- **Optimizer:** AdamW (LR=3e-4, weight decay=1e-4)
- **Batch Size:** 32
- **Split:** 80% train / 10% validation / 10% test
- **Model Size:** < 6.5 MB for edge deployment

## Results Summary
- **Baseline Accuracy:** 99.77%
- **CBAM Accuracy:** 99.72%
- **Baseline Parameters:** 1,556,806
- **CBAM Parameters:** 1,562,524 (+0.37%)
- **Baseline FLOPs:** 61.49M
- **CBAM FLOPs:** 62.22M (+1.2%)

## Project Structure
```
capstone/
├── LeafGuard_AI_Research_Experiment_Colab.ipynb  # Main experiment notebook
├── README.md                                      # Project documentation
├── context.md                                     # This file
├── requirements.txt                               # Python dependencies
├── .venv/                                         # Virtual environment
├── data/                                          # Dataset (downloaded automatically)
├── runs/                                          # Experiment outputs
│   ├── figures/                                   # Charts and diagrams (13 files)
│   ├── models/                                    # Model checkpoints (2 files)
│   ├── reports/                                   # CSV reports (9 files)
│   └── splits/                                    # Data split CSVs (3 files)
├── docs/                                          # Additional documentation
├── papers/                                        # Research papers
└── cartoon-style-green-leaf/                      # Assets
```

## Important Notes
- **DO NOT run training on Google Colab** - takes several hours, use local GPU
- Virtual environment required for dependency isolation
- Models are under 6.5 MB suitable for edge deployment
- All results are measured values, not assumed improvements
- CBAM provides interpretability via attention mechanisms

## Dependencies
- torch>=2.0.0
- torchvision>=0.15.0
- numpy, pandas, matplotlib, seaborn
- scikit-learn, pillow, tqdm
- kagglehub, grad-cam, torchinfo, thop, fvcore

## Hardware Requirements
- **Recommended:** NVIDIA GPU with CUDA (tested on RTX 3050 Laptop GPU)
- **Minimum:** 4GB VRAM
- **CPU Training:** Possible but extremely slow

## Progress Status
- ✅ Data preprocessing and splitting
- ✅ Model architecture implementation
- ✅ Training both baseline and CBAM models
- ✅ Performance evaluation and metrics
- ✅ Visualization generation
- ✅ Grad-CAM explainability
- ✅ Efficiency benchmarking
- ✅ Documentation completion

## Next Steps (if applicable)
- Deploy models to edge devices
- Test on real-world leaf images
- Extend to additional disease classes
- Optimize for mobile deployment
- Create web interface for inference
