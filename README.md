# 🚀 **Hazardous Asteroid Prediction**
A machine learning pipeline to predict potentially hazardous asteroids using NASA's **Near Earth Object Web Service (NeoWs)** dataset.  
This project applies **data preprocessing, feature engineering, synthetic data balancing (SMOTE), and machine learning model training** using PyCaret.

---

## **📌 Project Overview**
- **Preprocessing:** Cleans the dataset, removes unnecessary columns, and encodes categorical variables.
- **EDA:** Generates visual insights into asteroid characteristics.
- **Feature Engineering:** Applies **SMOTE** to balance class distribution and scales features.
- **Model Training:** Compares multiple models using **PyCaret** and selects the best one.
- **Evaluation:** Stores model performance metrics for analysis.

---

## **📌 Data Pipeline**
### **1️⃣ Preprocessing**
- Removes redundant features (e.g., duplicate distance measurements).
- Converts categorical values (e.g., `"Hazardous" → 1/0`).
- Saves cleaned dataset.

### **2️⃣ Exploratory Data Analysis (EDA)**
EDA visualizations help us understand the dataset distribution and feature relationships.

#### **Class Distribution of Hazardous vs. Non-Hazardous Asteroids**
![Class Distribution](reports/eda/class_distribution.png)

🔍 **Observations**:
- The dataset is **highly imbalanced**, with far more **non-hazardous asteroids (`0`) than hazardous ones (`1`)**.
- This imbalance explains why the model initially **overfitted**, favoring the majority class.
- **SMOTE was applied** to balance the dataset and ensure the model learns to detect hazardous asteroids correctly.

📌 **Why Does This Matter?**
- Without balancing, the model would classify most asteroids as **"not hazardous"**, leading to **poor recall**.
- **SMOTE improves recall**, ensuring that actual threats are detected.

#### **Feature Correlation Heatmap**
![Feature Correlation Heatmap](reports/eda/correlation_heatmap.png)

🔍 **Observations**:
- **High correlation values (>0.8) suggest feature redundancy**:
  - `Est Dia in KM(min)` and `Est Dia in KM(max)` are **strongly correlated (1.00)** → Keeping both may be unnecessary.
  - `Perihelion Time` and `Epoch Osculation` are **strongly correlated (0.98)**.
  - `Mean Motion` and `Orbital Period` are **negatively correlated (-0.99)** → Likely convey the same information.
  
- **Low correlation with `Hazardous` (~0.2 - 0.3)**:
  - No single feature **directly determines** whether an asteroid is hazardous.
  - The model must learn **complex, non-linear relationships** instead.

📌 **Why Does This Matter?**
- **Removing redundant features improves model efficiency**.
- **Low correlation with `Hazardous`** means the model must combine **multiple weak signals** to make predictions.
- **Feature selection and regularization** help avoid overfitting.

---

## **📌 Model Training & Evaluation**
### **4️⃣ Model Training**
- Trains **Decision Tree, Random Forest, AdaBoost, Gradient Boosting**, and more.
- Uses **PyCaret** to compare models automatically.
- Saves model performance metrics.

| Model                        | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| **Gradient Boosting Classifier** | 🚀 **1.00** | **1.00** | **1.00** | **1.00** |

📌 **Full performance table saved in**: [`reports/model_performance.csv`](reports/model_performance.csv)

### **🛠 Why is Accuracy 100%?**
- **Asteroid impacts are rare** → The dataset might have strong feature correlations.
- **SMOTE added synthetic minority samples** → Might have led to overfitting.
- **Tree-based models can memorize training data** → Need stronger regularization.
- **Next steps**:
  - Apply **cross-validation** to validate performance.
  - Limit **tree depth** to prevent memorization.
  - Experiment with **alternative resampling methods**.

---

## **📌 How to Run the Project**
### **1️⃣ Install Dependencies**
Ensure you have all required libraries installed:
```bash
pip install -r requirements.txt
