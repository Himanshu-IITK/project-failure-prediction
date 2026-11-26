# Project Failure Prediction Using Machine Learning

An end-to-end Machine Learning decision-support system that predicts project failure early in the execution lifecycle by combining **Project Management Earned Value & Risk metrics** with ensemble classification models and an interactive **Streamlit** dashboard.

Developed as a production-grade implementation of the **MBA666 (Project Management)** framework at **IIT Kanpur**.

---

## Key Features & Engineering Highlights

- **Multi-Domain Data Synthesis:** Generates realistic project records across Software Engineering, Agile IT, Enterprise Infrastructure, and Startup domains with correlated budgets, schedules, missing values, outliers, and class imbalance.
- **Domain-Specific PM Feature Engineering:**
  - **Cost Performance Index (CPI):** $\text{Budget} / \text{Actual Cost}$
  - **Schedule Performance Index (SPI):** $\text{Estimated Timeline} / \text{Actual Duration}$
  - **Communication Complexity:** PMBOK channels formula $\frac{n(n-1)}{2}$
  - **Methodology Mismatch Score:** Quantifies friction between rigid methodologies (e.g., Waterfall) and high scope-change volatility.
- **Leak-Free Preprocessing Pipeline:** Stratified 80/20 split performed _before_ median imputation, IQR Winsorization, one-hot encoding, standard scaling, and SMOTE oversampling (applied strictly to training data).
- **Multi-Model Benchmarking:** 5-Fold Stratified Cross-Validation & `GridSearchCV` across Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost.
- **Interactive Decision-Support App:** Streamlit web interface allowing project managers to simulate project parameters, view live CPI/SPI health, and receive real-time failure probability scores.

---

## Project Structure

```text
project-failure-prediction/
├── app.py                   # Interactive Streamlit web application
├── data/
│   ├── raw/                 # Synthesized raw dataset
│   └── processed/           # Leak-free train (SMOTE) & untouched test splits
├── models/                  # Serialized preprocessor & tuned ML models (.joblib)
├── reports/
│   ├── figures/             # Automated EDA, ROC curves, confusion matrix & feature importance
│   ├── cv_summary.csv       # 5-fold CV benchmark results
│   └── test_evaluation.csv  # Unbiased test set metrics
├── src/
│   ├── data_generator.py    # Multi-domain dataset synthesis
│   ├── features.py          # Row-wise PM feature engineering & custom transformers
│   ├── eda.py               # Automated EDA plots & correlation analysis
│   ├── preprocess.py        # Sklearn ColumnTransformer + IQR Capper + SMOTE
│   ├── train.py             # 5-fold CV & hyperparameter tuning
│   └── evaluate.py          # Unbiased test evaluation & model selection
└── requirements.txt
```
