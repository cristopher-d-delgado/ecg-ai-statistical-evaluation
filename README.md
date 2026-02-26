# ECG AI Statistical Validation

*A Reproducible Deep Learning and Biostatistical Evaluation Framework for Clinical ECG Modeling*

---

## 📌 Project Overview

This project develops and statistically validates a deep learning model for multi-label ECG classification using the **PTB-XL dataset**.

It is designed to reflect:

* 🏥 Medical device–grade biostatistical rigor
* 🤖 Clinical AI model development best practices
* 📊 Regulatory-style validation methodology
* 🔬 Reproducible scientific research standards

---

## 🎯 Project Objectives

1. Develop a deep learning model for ECG diagnostic classification.
2. Perform rigorous statistical validation of model performance.
3. Evaluate calibration, robustness, and subgroup behavior.
4. Compare deep learning performance against classical statistical models.
5. Structure the project as if preparing for regulatory or clinical deployment.

---

## 🏥 Clinical Framing

### Intended Use

Automated multi-label classification of major ECG diagnostic superclasses.

### Target Population

Adult patients represented in PTB-XL ECG recordings.

### Primary Endpoint

* **Macro-AUC** across diagnostic superclasses.

### Secondary Endpoints

* Per-class AUC
* Sensitivity & Specificity
* F1-score
* Precision-Recall AUC
* Calibration metrics (ECE, Brier score)

---

## 📊 Statistical Analysis Plan (SAP)

### Primary Analysis

* Compute Macro-AUC on held-out test set.
* Estimate 95% confidence intervals via bootstrap (≥1000 resamples).

### Secondary Analyses

* Per-class ROC curves
* Micro vs Macro AUC
* Sensitivity & Specificity at fixed thresholds
* F1-score evaluation

### Calibration Analysis

* Reliability diagrams
* Expected Calibration Error (ECE)
* Brier Score
* Temperature scaling (post-hoc calibration)

### Model Comparison

Compare:

* Logistic Regression baseline
* Random Forest baseline
* Deep Learning model (ResNet1D)

Statistical comparison via:

* DeLong test for AUC difference

### Subgroup Analyses

Evaluate performance stratified by:

* Sex
* Age group
* Diagnostic category prevalence

---

## 🤖 Deep Learning Framework

### Architecture

* 1D Convolutional Neural Network (ResNet-style)
* Multi-label sigmoid output layer
* Weighted loss for class imbalance

### Training Strategy

* Patient-level train/validation/test split
* Stratified sampling
* Mixed precision training
* Deterministic seeding for reproducibility

---

## 🧪 Validation Philosophy

This project emphasizes:

* No patient-level data leakage
* Reproducible experiment configuration
* Explicit reporting of uncertainty
* Statistical comparison rather than anecdotal improvement
* Calibration assessment (not just discrimination)

---

## 📂 Repository Structure

```
ecg-ai-statistical-validation/
│
├── data/                  # PTB-XL dataset
├── notebooks/             # EDA and exploratory analyses
├── src/
│   ├── data/              # Dataset loading & preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training loops
│   ├── evaluation/        # Metrics & statistical tests
│   └── utils/
│
├── results/               # Model outputs & figures
├── models/                # Saved trained models
├── ptbxl_env.yml          # Conda environment
└── README.md
```

---

## 📈 Evaluation Metrics

| Category            | Metrics                   |
| ------------------- | ------------------------- |
| Discrimination      | Macro-AUC, Micro-AUC      |
| Threshold-based     | Sensitivity, Specificity  |
| Multi-label         | F1-score                  |
| Calibration         | ECE, Brier Score          |
| Statistical Testing | Bootstrap CI, DeLong Test |

---

## 🔬 Reproducibility

* Fixed random seeds
* Environment specification (`ptbxl_env.yml`)
* Deterministic PyTorch settings
* Explicit dataset split logging

---

## 🚀 Installation

### 1️⃣ Create Environment

```bash
conda create -n ptbxl_ai python=3.10
conda activate ptbxl_ai
```

### 2️⃣ Install Dependencies

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pandas scipy scikit-learn matplotlib seaborn wfdb
```

---

## 📥 Dataset Download

Example download via AWS CLI:

```bash
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptbxl
```

---

## 🧠 Positioning

This repository demonstrates:

* Clinical statistical rigor (biostatistician perspective)
* Deep learning implementation expertise (clinical AI scientist perspective)
* Regulatory-style validation discipline
* Real-world medical AI evaluation methodology