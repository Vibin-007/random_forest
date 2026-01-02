# 🏦 Loan Approval Analysis: Random Forest Classification & Regression

This project implements a machine learning solution to analyze and predict loan approvals using the **Random Forest** algorithm. It features an interactive **Streamlit UI** to explore the dataset and evaluate both classification and regression models.

## 🚀 Features

- **Classification**: Predicts whether a loan will be **Approved** or **Rejected** based on features like income, assets, and CIBIL score.
- **Regression**: Predicts the **Loan Amount** an applicant might be eligible for.
- **Interactive Dashboard**:
    - **Data Overview**: View raw data and statistical summaries.
    - **Hyperparameter Tuning**: Adjust the number of trees and max depth in real-time.
    - **Visualizations**: Feature importance bars and actual vs. predicted scatter plots.
- **Preprocessing**: Handles data cleaning, column stripping, and label encoding automatically.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vibin-007/random_forest.git
   cd random_forest
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

Run the Streamlit application:
```bash
python -m streamlit run app.py
```

## 📊 Dataset

The project uses a `loan_approval_dataset.csv` containing information such as:
- Number of dependents
- Education level
- Self-employment status
- Annual income
- Asset values (Luxury, Bank, etc.)
- CIBIL score
- Loan status (Target for Classification)
- Loan amount (Target for Regression)

## 📦 Requirements

- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

---
Developed by [Vibin-007](https://github.com/Vibin-007)