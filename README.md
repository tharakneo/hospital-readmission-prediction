# Hospital Readmission Prediction: Classification Analysis

## Overview
Built and compared multiple classification models to predict hospital readmission for diabetic patients using 100K+ medical records. Applied feature selection, class imbalance handling, and hyperparameter tuning to identify the best-performing model.

## Key Findings
- Random Forest with GridSearchCV achieved the highest overall performance
- Chi-Square feature selection narrowed 50+ features down to the top 10 most predictive clinical variables
- SMOTE oversampling improved recall for the minority (readmitted) class
- Logistic Regression provided the most interpretable baseline model

## Methodology
1. **Data Preprocessing:** Cleaned 100K+ records, removed duplicates, encoded categorical variables, scaled features with MinMaxScaler
2. **EDA:** Visualized distributions of age, gender, time in hospital, and correlations with readmission
3. **Feature Selection:** Chi-Square test to select top 10 most relevant features
4. **Class Imbalance:** Applied SMOTE to balance readmitted vs non-readmitted classes
5. **Models Built:** Logistic Regression, Decision Tree, Random Forest
6. **Hyperparameter Tuning:** GridSearchCV for optimal model parameters
7. **Evaluation:** Accuracy, F1-score, Precision, Recall, ROC-AUC curves

## Tools
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, imbalanced-learn (SMOTE)

## Dataset
Diabetic patient readmission dataset (100K+ records, 50+ features)

## Project Structure
```
├── hospital_readmission_prediction.ipynb   # Full analysis notebook
├── data/
│   └── diabetic_data.csv                   # Dataset (100K+ records)
└── README.md
```

## How to Run
1. Open the notebook in Google Colab or Jupyter
2. Dataset is included in the `data/` folder
3. Run all cells sequentially
