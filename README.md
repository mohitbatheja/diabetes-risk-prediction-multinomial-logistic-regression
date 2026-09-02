# Diabetes Risk Prediction using Logistic Regression

A machine learning project designed to predict a patient's diabetes risk level (**Low**, **Moderate**, or **High**) using health metrics, demographic indicators, and lifestyle factors.

---

## 📌 Project Overview
This project processes medical and lifestyle data to classify diabetes risk levels using a **One-vs-Rest (OvR) Logistic Regression** classification approach. It handles missing categorical values via mode imputation, performs one-hot encoding on categorical features, scales numerical inputs, and balances class distribution to handle class imbalance effectively.

---

## 📊 Dataset Overview
The model uses `diabetes_risk.csv` containing **15,000 patient records** across 19 features:

- **Patient Demographics:** `age`, `gender`, `city`, `income_bracket`
- **Biometric Factors:** `bmi`, `fasting_blood_sugar`, `hba1c_level`, `blood_pressure_systolic`, `blood_pressure_diastolic`, `waist_circumference_cm`
- **Lifestyle & Medical History:** `family_history_diabetes`, `physical_activity_level`, `diet_type`, `smoking_status`, `alcohol_consumption`, `hours_sleep_per_night`, `stress_level`
- **Target Variable:** `diabetes_risk` (`Low`, `Moderate`, `High`)

---

## 🛠️ Pipeline & Workflow

1. **Data Preprocessing & Cleaning:**
   - Handled missing values in `smoking_status`, `alcohol_consumption`, and `income_bracket` using **Mode Imputation**.
   - Removed irrelevant identifier columns (`patient_id`).
   - Categorical feature encoding using `pd.get_dummies(drop_first=True)`.

2. **Train-Test Split & Scaling:**
   - **80/20 Stratified Split** (12,000 train samples / 3,000 test samples) to preserve class proportions.
   - **StandardScaler** applied to feature columns.

3. **Model Training:**
   - Implemented `OneVsRestClassifier` paired with `LogisticRegression`.
   - Set `class_weight='balanced'` and `max_iter=1000` for optimal convergence on imbalanced data.

---

## 📈 Model Performance

- **Overall Accuracy:** `77.67%`

### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **High Risk** | 0.68 | 0.84 | 0.75 | 450 |
| **Low Risk** | 0.89 | 0.86 | 0.88 | 1800 |
| **Moderate Risk** | 0.57 | 0.54 | 0.55 | 750 |
| **Accuracy** | | | **0.78** | **3000** |
| **Macro Avg** | 0.71 | 0.75 | 0.73 | 3000 |
| **Weighted Avg** | 0.78 | 0.78 | 0.78 | 3000 |

---

```

2. **Install dependencies:**
```bash
pip install pandas scikit-learn

```


3. **Run the Jupyter Notebook:**
```bash
jupyter notebook diabetes_risk.ipynb

```



---

## 🛠️ Tech Stack

* **Python 3.x**
* **Pandas** (Data manipulation)
* **Scikit-Learn** (Model training, preprocessing, evaluation)

```

```
