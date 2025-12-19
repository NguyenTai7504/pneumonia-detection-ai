<div align="center">

# 🩻 Pneumonia Detection AI

### Deep Learning-Powered Chest X-Ray Analysis System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Advanced AI system for pneumonia detection from chest X-ray images using ResNet50 architecture with Grad-CAM visualization**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model Architecture](#-model-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

This project implements a state-of-the-art deep learning system for automated pneumonia detection from chest X-ray images. Built with PyTorch and ResNet50 architecture, it features an intuitive Streamlit web interface and advanced Grad-CAM visualization to highlight affected lung regions.

### Key Highlights

- 🎯 **Binary Classification**: NORMAL vs PNEUMONIA detection
- 🔍 **Explainable AI**: Grad-CAM heatmaps for lesion localization
- 🌐 **Web Interface**: User-friendly Streamlit dashboard
- ⚡ **CPU Optimized**: Runs efficiently without GPU requirements
- 🎨 **Advanced Visualization**: Anti-glare processing with Otsu masking

---

## ✨ Features

### Core Functionality

- **Automated Diagnosis**: Real-time pneumonia detection with confidence scores
- **Visual Explanation**: Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Interactive UI**: Drag-and-drop image upload with sample image testing
- **Medical Recommendations**: Context-aware suggestions based on AI predictions
- **Batch Processing**: Console script for bulk image analysis

### Technical Features

- ResNet50 backbone with custom classification head
- Smart preprocessing pipeline with data augmentation
- Otsu thresholding for body mask extraction
- CLAHE enhancement for better visualization
- Anti-glare heatmap rendering

---

## 🎬 Demo

### Web Interface
```bash
streamlit run app.py
```

### Console Script
```bash
python main.py
```

**Sample Output:**
```
------------------------------
IMAGE: data_samples/PNEUMONIA/pneumonia_001.jpg
RESULT: PNEUMONIA
CONFIDENCE: 94.32%
------------------------------
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- Trained model weights (`.pth` file)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/pneumonia-detection-ai.git
cd pneumonia-detection-ai
```

### Step 2: Install Dependencies

**Option A: Using existing environment**
```bash
.\python_env\python.exe -m pip install -r requirements.txt
```

**Option B: Create new virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Download Sample Data

```bash
python download_samples.py
```

This downloads 10 sample chest X-rays (5 normal + 5 pneumonia cases) from the IEEE8023 COVID-19 dataset.

### Step 4: Place Model Weights

Place your trained model file in the following location:
```
models/final_pneumonia_model.pth
```

> ⚠️ **Important**: The model file is required to run the application. Train your own model or obtain pre-trained weights.

---

## 💻 Usage

### Web Application (Recommended)

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open browser at `http://localhost:8501`

3. Upload a chest X-ray image or use sample images

4. Click "ANALYZE" to get predictions

### Command Line Interface

Edit `IMAGE_PATH` in `main.py` then run:
```bash
python main.py
```

---

## 🏗️ Model Architecture

### Network Structure

```
ResNet50 (Pretrained)
    ├── Convolutional Layers (Frozen)
    ├── Layer4 (Feature Extraction)
    └── Custom Classifier
        ├── Linear(2048 → 256)
        ├── ReLU
        ├── Dropout(0.5)
        └── Linear(256 → 2)
```

### Training Configuration

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 1e-4 with scheduler
- **Data Augmentation**: Random rotation, horizontal flip, color jitter
- **Regularization**: Dropout (0.5) + L2 penalty

---

## 📂 Project Structure

```
pneumonia-detection-ai/
│
├── app.py                      # Streamlit web application
├── main.py                     # Console testing script
├── requirements.txt            # Python dependencies
├── download_samples.py         # Sample data downloader
├── README.md                   # This file
│
├── models/
│   └── final_pneumonia_model.pth  # Model weights (required)
│
├── data_samples/
│   ├── NORMAL/                 # Normal X-ray samples
│   └── PNEUMONIA/              # Pneumonia X-ray samples
│
├── utils/
│   ├── __init__.py
│   └── gradcam.py              # Grad-CAM implementation
│
└── python_env/                 # Python virtual environment
```

---

## 🔬 Methodology

### Data Pipeline

1. **Preprocessing**: Resize to 224×224, normalize with ImageNet statistics
2. **Inference**: Forward pass through ResNet50
3. **Grad-CAM**: Backpropagation to layer4 for activation maps
4. **Post-processing**: Otsu masking + CLAHE enhancement
5. **Visualization**: Heatmap overlay with adjustable opacity

### Grad-CAM Algorithm

```python
# Simplified workflow
1. Forward pass → Get predictions
2. Backward pass → Compute gradients on target layer
3. Global average pooling on gradients
4. Weighted combination of feature maps
5. ReLU activation + normalization
6. Overlay on original image
```

---

## 🧪 Performance Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 95.2% | 92.8% |
| Precision | 94.6% | 91.3% |
| Recall | 96.1% | 93.5% |
| F1-Score | 95.3% | 92.4% |

*Note: Replace with your actual model metrics*

---

## 📊 Dataset

### Training Data
- **Source**: Kaggle Chest X-Ray Images (Pneumonia)
- **Total Images**: 5,863 images
- **Classes**: NORMAL (1,583) / PNEUMONIA (4,280)
- **Format**: JPEG, grayscale/RGB
- **Resolution**: Variable (resized to 224×224)

### Sample Data
- **Source**: [IEEE8023 COVID-19 Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- **License**: Public Domain
- **Usage**: Testing and demonstration purposes

---

## 🛠️ Troubleshooting

### Common Issues

**1. Model file not found**
```bash
# Ensure model is in correct location
ls models/final_pneumonia_model.pth
```

**2. Module import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. CUDA/GPU errors**
```python
# Model automatically uses CPU fallback
# No action needed - GPU not required
```

**4. Streamlit port already in use**
```bash
# Use custom port
streamlit run app.py --server.port 8502
```

---

## ⚠️ Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This application is designed for academic research and educational demonstrations. It should **NOT** be used for:

- Clinical diagnosis or treatment decisions
- Replacing professional medical advice
- Patient care without physician oversight

Always consult qualified healthcare professionals for medical diagnoses.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation accordingly
- Ensure backward compatibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Streamlit** for the web application framework
- **IEEE8023** for the COVID-19 chest X-ray dataset
- **Kaggle Community** for pneumonia detection datasets
- Research papers on medical image analysis and Grad-CAM

---

## 📧 Contact

**Project Maintainer**: Your Name  
**Email**: your.email@example.com  
**GitHub**: [@yourusername](https://github.com/yourusername)

---

## 🔗 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Selvaraju, R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks"
3. Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection"

---

<div align="center">

**Made with ❤️ for Medical AI Research**

⭐ Star this repo if you find it helpful!

</div>
