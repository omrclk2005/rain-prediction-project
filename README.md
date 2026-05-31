# 🌦️ Australia Rain Prediction AI Project

This repository contains an end-to-end, production-ready Machine Learning pipeline developed to predict whether it will rain tomorrow in Australia. The project strictly follows rigorous data preprocessing, advanced feature engineering, and state-of-the-art model evaluation practices to handle heavily imbalanced weather data.

---

## 📂 Project Structure

The workflow is broken down into structured Jupyter notebooks inside the `notebooks/` directory to ensure complete modularity:

* **`01_exploratory_data_analysis.ipynb`**: Initial data profiling, feature distribution visualization, and missing value detection.
* **`02_data_cleaning_and_processing.ipynb`**: Location-based median imputation for structural missing data and IQR capping to mitigate extreme outlier anomalies.
* **`03_feature_engineering.ipynb`**: Implementation of cyclic temporal encoding (Sine/Cosine for months), data scaling (`StandardScaler`), and custom meteorological interaction features (`Temp_Range`, `Humidity_Temp_Interaction`).
* **`04_model_training_and_evaluation.ipynb`**: Training of 4 distinct models, hyperparameter tuning via `GridSearchCV`, threshold-agnostic evaluation, and SHAP-based model interpretation.

---

## 🏆 Key Methodology & Architectural Decisions

1. **Handling Class Imbalance**: With ~78% "No Rain" vs ~22% "Rain" days, naive accuracy is a deceptive metric. The entire pipeline optimizes for the minority class using `stratify=y` splits and targeting the **F1-Score** during cross-validation.
2. **Feature Interaction**: Engineered custom weather indicators like `Temp_Range` (diurnal variation) and `Humidity_Temp_Interaction` to capture thermodynamic atmospheric instability before rain events.
3. **Hyperparameter Tuning**: Utilized 3-Fold Cross-Validated GridSearch on Random Forest to mathematically extract optimal structural boundaries instead of relying on default architectural baselines.

---

## 📊 Performance Matrix & Evaluation

Four algorithms were benchmarked head-to-head on the unseen 20% test partition:

| Model | Overall Accuracy | Minority Recall (Rain) | Minority F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 84.1% | 0.44 | 0.54 | 0.8642 |
| **Decision Tree** | 83.2% | 0.49 | 0.53 | 0.8120 |
| **Tuned Random Forest** | 86.0% | 0.48 | 0.59 | 0.8785 |
| 👑 **XGBoost (Champion)** | **86.0%** | **0.54** | **0.63** | **0.8878** |

### 📌 Evaluation Takeaways
* **XGBoost Classifier** is crowned as the champion model. It yields the highest **Minority Recall (0.54)** and **F1-Score (0.63)**, significantly reducing False Negatives (missing a real rainy day).
* **Precision-Recall Curves** and **Confusion Matrices** were utilized to robustly validate model reliability beyond standard ROC-AUC constraints.

---

## 🔬 Model Interpretation (SHAP Analysis)

To break the "black-box" barrier of ensemble networks, global model transparency was established using **SHAP (SHapley Additive exPlanations)** on the champion XGBoost model:
* **`Humidity3pm`** emerged as the single most critical indicator; high afternoon humidity strongly accelerates positive rain predictions.
* Atmospheric pressure shifts (**`Pressure3pm`**) and lower sunshine intervals (**`Sunshine`**) follow as primary thermodynamic drivers for precipitation.

---

## ⚙️ Quick Start & Production Inference

To replicate the environment and run this project locally:

1. Clone the repository and navigate to the root directory.
2. Install the production dependencies:
   ```bash
   pip install -r requirements.txt