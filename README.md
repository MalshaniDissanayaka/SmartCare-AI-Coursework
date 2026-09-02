# SmartCare Hospital — AI-Powered 30-Day Patient Readmission Prediction

**CCS3440 — Artificial Intelligence Coursework**

---

## 🌟 Overview

Welcome to **SmartCare AI**, an intelligent clinical and operational decision-support prototype developed as part of the **CCS3440 Artificial Intelligence coursework**.

The system predicts whether a hospital patient is likely to be **readmitted within 30 days of discharge**. SmartCare combines data preprocessing, machine learning, model evaluation, and **Explainable AI (XAI)** to transform patient information into understandable and actionable insights.

The project also includes an interactive **Streamlit web application** that allows users to enter patient information and obtain a predicted readmission risk together with an explanation of the prediction.

🚀 **Explore the Live Application:**
[SmartCare Streamlit Prototype](https://i7j6zly2spyf5kxtor9znp.streamlit.app/)

---

## 🎯 Project Objectives

The main objectives of SmartCare AI are to:

* Develop a machine learning model for predicting 30-day hospital readmission.
* Perform data preprocessing and feature engineering on patient-related data.
* Compare multiple machine learning algorithms using appropriate evaluation metrics.
* Select the most suitable model based on predictive performance.
* Apply **Explainable AI (XAI)** techniques to make model predictions easier to understand.
* Develop an interactive Streamlit prototype for demonstrating the prediction system.

---

## 🧠 Machine Learning Approach

Several machine learning models were developed and evaluated:

* Logistic Regression
* XGBoost
* Support Vector Machine (RBF)
* Random Forest

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

**Logistic Regression was selected as the final model based primarily on ROC-AUC**, as ROC-AUC provides a threshold-independent measure of the model's ability to distinguish between patients who are and are not readmitted.

For the complete methodology, evaluation, and model-selection justification, refer to the **SmartCare Technical Report**.

---

## 📊 Model Performance

| Model                              |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ---------------------------------- | --------: | --------: | --------: | --------: | --------: |
| **Logistic Regression (Selected)** |     0.636 |     0.829 |     0.667 |     0.739 | **0.637** |
| XGBoost                            |     0.621 |     0.810 |     0.667 |     0.731 |     0.616 |
| SVM (RBF)                          |     0.682 |     0.826 |     0.745 |     0.784 |     0.593 |
| Random Forest                      | **0.788** |     0.785 | **1.000** | **0.879** |     0.536 |

Although Random Forest achieved higher accuracy, recall, and F1-score on the evaluated test set, **Logistic Regression was selected based on its superior ROC-AUC**, which was the primary model-selection criterion for this project.

---

## 🔍 Explainable AI

SmartCare incorporates **Explainable AI (XAI)** to provide insight into the factors influencing the model's predictions.

The project uses **SHAP (SHapley Additive exPlanations)** to help explain model behaviour and identify the contribution of individual features to predictions.

The explanations are intended to improve the transparency and interpretability of the prototype rather than replace professional clinical judgement.

---

## 📂 Repository Structure

```text
📦 SmartCare-AI-Coursework
│
├── 📂 data/
│   ├── smartcare_ai_dataset_1000.csv
│   └── smartcare_ai_dataset_data_dictionary.csv
│
├── 📂 models/
│   ├── readmission_model.pkl
│   ├── model_metadata.json
│   └── model_comparison_results.csv
│
├── 📄 app.py
├── 📄 SmartCare_Readmission_Prediction.ipynb
├── 📄 SmartCare_Readmission_Prediction.py
├── 📄 SmartCare_Technical_Report.pdf
├── 📄 SmartCare_Presentation.pptx
├── 📄 requirements.txt
└── 📄 README.md
```

### Important

The Streamlit application loads:

```text
models/readmission_model.pkl
models/model_metadata.json
```

Therefore, the application should be run from the **repository root**, or the paths in `app.py` should be adjusted if the folder structure is changed.

---

## 🚀 Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/MalshaniDissanayaka/SmartCare-AI-Coursework.git
```

### 2. Navigate to the project directory

```bash
cd SmartCare-AI-Coursework
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will then open in your browser.

---

## 📓 Project Notebook

The Jupyter Notebook contains the main development workflow, including:

1. Data loading
2. Data exploration
3. Data preprocessing
4. Feature engineering
5. Model training
6. Model comparison
7. Model evaluation
8. Explainable AI / SHAP analysis

The notebook provides the detailed experimental workflow behind the final prototype.

---

## 📁 Dataset

The project uses a **synthetic healthcare dataset** created for educational and coursework purposes.

The dataset contains patient-related attributes used to demonstrate the development of a 30-day readmission prediction system.

The accompanying:

```text
smartcare_ai_dataset_data_dictionary.csv
```

provides descriptions and metadata for the dataset attributes.

---

## 👥 Team Contributions

| Team Member              | Student ID     | Contribution                               |
| ------------------------ | -------------- | ------------------------------------------ |
| **Achira Sadharanga**    | CIT-23-02-0170 | Literature review and technical report     |
| **Hithesh Maheepala**    | CIT-23-02-0140 | Data Preprocessing & Feature Engineering |
| **Seminda Fernando**     | CIT-23-02-0122 | Exploratory Data Analysis and ML Model Development           |
| **Navodya Sankalpani**   | CIT-23-02-0046 | Model Evaluation and Explainable AI Analysis (SHAP)           |
| **Malshani Dissanayaka** | CIT-23-02-0026 | Model Export for Prototype     |

All team members contributed to the overall development, testing, documentation, and completion of the SmartCare AI coursework project.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **SHAP**
* **Streamlit**
* **Matplotlib**
* **Plotly**
* **Joblib**
* **Jupyter Notebook**

---

## 🌐 Live Prototype

The deployed Streamlit prototype is available here:

**[SmartCare AI — Live Streamlit Application](https://i7j6zly2spyf5kxtor9znp.streamlit.app/)**

---

## 📄 Project Documentation

The repository includes:

* **Technical Report** — Detailed explanation of the project methodology, experiments, results, and conclusions.
* **Presentation** — Summary of the project, methodology, results, and prototype.
* **Jupyter Notebook** — Complete experimental and modelling workflow.
* **Python Script** — Clean Python implementation of the core prediction pipeline.

---

## ⚠️ Disclaimer

> **This project is an academic proof-of-concept developed for the CCS3440 Artificial Intelligence coursework. The dataset is synthetic and the system has not been clinically validated. The predictions and explanations generated by this prototype must not be used to make real-world patient care, diagnosis, treatment, or hospital management decisions.**

---

## 👩‍💻 Academic Context

**Course:** CCS3440 — Artificial Intelligence
**Institution:** SLTC University
**Faculty:** Faculty of Computing & IT
**Project:** SmartCare Hospital — AI-Powered 30-Day Patient Readmission Prediction

---

⭐ **SmartCare AI — Turning healthcare data into interpretable predictive insights.**
