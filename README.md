# FootSegONN: An Ensemble of Self-ONN-based Models for Diabetic Foot Ulcer Segmentation

## 📖 Overview
FootSegONN is an ensemble deep learning model utilizing Self-ONN architectures for **diabetic foot ulcer segmentation**. It leverages **segmentation_models.pytorch** for state-of-the-art performance in medical image segmentation.

---


# Examples
<div align=center>

![](images/example_img2.png)

</div>


## 📂 Project Structure  

```plaintext
FootsegONN/
│
├── input/
│   └── ...                  # Contains input data or related files
│
├── models/
│   └── ...                  # Contains model architectures and related scripts
│
├── save_masks/
│   └── ...                  # Directory for saving generated masks
│
├── README.md                # Project documentation and overview
├── SelfONN.py               # Implementation of Self-ONN architectures
├── SelfONN_decoders.py      # Decoders for Self-ONN models
├── ensemble_masks.py        # Script for ensemble mask generation
├── inference.ipynb          # Jupyter notebook for inference
├── models.py                # Script defining model architectures
├── prediction_masks.py      # Script for generating prediction masks
├── requirements.txt         # List of required Python packages
└── selfonnlayer.py          # Implementation of Self-ONN layers


```

## 🚀 Installation Guide

### **1️⃣ Recommended Environment**
- **Python 3.10** (Recommended)
- **pip (Latest Version)**
- **Git (Required for dependencies)**

### **2️⃣ Setup Instructions**
Follow these steps to install and set up the environment:



#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/sumon3255/FootsegONN.git
cd FootSegONN


```
