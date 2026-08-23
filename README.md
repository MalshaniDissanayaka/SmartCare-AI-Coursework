# 🏥 SmartCare Hospital — AI-Powered 30-Day Patient Readmission Prediction
**CCS3440 — Artificial Intelligence Coursework** | Faculty of Computing & IT at SLTC

---

## 🌟 Overview
Welcome to the official repository for **SmartCare AI**, an intelligent clinical and operational decision-support prototype. Developed as part of the CCS3440 Artificial Intelligence coursework, this system predicts whether a hospital patient is likely to be readmitted within 30 days of discharge. By combining robust data pipelines, machine learning models, and transparent Explainable AI (XAI), SmartCare bridges the gap between raw healthcare data and actionable clinical insights.

🚀 **Explore the Live App:** [SmartCare Streamlit Prototype](https://i7j6zly2spyf5kxtor9znp.streamlit.app/)

---

## 📂 Repository Architecture
```text
📦 SmartCare-Readmission-Prediction
│
├── 📂 data/
│   ├── smartcare_ai_dataset_1000.csv          # Source dataset (1,000 records)
│   └── smartcare_ai_dataset_data_dictionary.csv # Attribute definitions and metadata
│
├── 📂 models/
│   ├── readmission_model.pkl                  # Trained scikit-learn pipeline (Logistic Regression)
│   ├── model_metadata.json                    # Feature lists, category options, and numerical bounds
│   └── model_comparison_results.csv           # Comprehensive evaluation metrics across models
│
├── 📄 app.py                                  # Interactive Streamlit front-end application
├── 📄 SmartCare_Readmission_Prediction.ipynb  # End-to-end Jupyter notebook (EDA, modeling, SHAP)
├── 📄 SmartCare_Readmission_Prediction.py     # Clean Python script version of the core pipeline
├── 📄 SmartCare_Technical_Report.pdf          # Full formal technical report (all 14 required sections)
├── 📄 SmartCare_Presentation.pptx             # Project presentation slides
└── 📄 requirements.txt                        # Project dependencies

The app loads `models/readmission_model.pkl` and `models/model_metadata.json`, so run it
from the repository root (or adjust the paths in `app.py` if you restructure folders).

## Model Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.636 | 0.829 | 0.667 | 0.739 | **0.637** |
| XGBoost | 0.621 | 0.810 | 0.667 | 0.731 | 0.616 |
| SVM (RBF) | 0.682 | 0.826 | 0.745 | 0.784 | 0.593 |
| Random Forest | 0.788 | 0.785 | 1.000 | 0.879 | 0.536 |

Logistic Regression was selected on ROC-AUC (threshold-independent ranking ability),
not raw accuracy — see Section 8.4 of the technical report for the full justification.

## Disclaimer

This is a coursework proof-of-concept trained on a 330-record synthetic dataset. It is
**not clinically validated** and must not be used to inform real patient care decisions.

## Team

| Name | Student ID | Contribution |
|---|---|---|
| Achira Sadharanga | CIT-23-02-0170 | Literature review & technical report |
| Hithesh Maheepala | CIT-23-02-0140 | Data preprocessing & feature engineering |
| Seminda Fernando | CIT-23-02-0122 | Model development & evaluation |
| Navodya Sankalpani | CIT-23-02-0046 | Model development & evaluation |
| Malshani Dissanayaka | CIT-23-02-0026 | Explainable AI & prototype |
