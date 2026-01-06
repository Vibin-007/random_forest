# 🏦 Loan Approval Predictor (Random Forest)

A comprehensive Streamlit application that performs both Classification (Loan Approval Status) and Regression (Loan Amount Prediction) using Random Forest algorithms.

## 📊 Features

- **Approval Prediction (Classification)**: Predicts if a loan will be Approved or Rejected.
- **Amount Prediction (Regression)**: Estimates the maximum loan amount an applicant is eligible for.
- **Hybrid Dashboard**: Separate tabs for Classification and Regression analysis.
- **Interactive Tuning**: Adjust Hyperparameters like `n_estimators` (Trees) and `max_depth` on the fly.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vibin-007/random_forest.git
   cd random_forest
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Main application combining both Classification and Regression.
- `classification.py`: Standalone script for approval prediction.
- `regression.py`: Standalone script for amount prediction.
- `loan_approval_dataset.csv`: Dataset containing financial and personal applicants' data.
- `random_forest_analysis.ipynb`: Jupyter notebook for in-depth analysis.
- `requirements.txt`: List of Python dependencies.

## 📈 Model Information

The application uses **Random Forest** for two tasks:
1.  **Classification** (Status): Based on CIBIL score, Income, Assets.
2.  **Regression** (Amount): Based on Income, Dependents, Loan Term.