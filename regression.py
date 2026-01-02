import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Loan Amount Regression (Random Forest)", layout="wide")

st.title("📈 Loan Amount Regression (Random Forest)")
st.markdown("""
This app predicts the **Loan Amount** using a Random Forest Regressor.
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Preprocessing
def preprocess_data(df):
    df_p = df.copy()
    le = LabelEncoder()
    categorical_cols = ['education', 'self_employed', 'loan_status']
    for col in categorical_cols:
        df_p[col] = le.fit_transform(df_p[col].str.strip())
    
    X = df_p.drop(['loan_id', 'loan_amount'], axis=1)
    y = df_p['loan_amount']
    return X, y

st.header("📊 Dataset Overview")
st.dataframe(df.head(10))

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

if st.button("Train Regressor"):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.success("Model trained successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R2 Score", f"{r2_score(y_test, y_pred):.4f}")
        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):,.0f}")
        
        st.subheader("Feature Importance")
        feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax1, palette="magma")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Actual vs Predicted Scatter Plot")
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred, alpha=0.5)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        st.pyplot(fig2)
        
    st.subheader("Actual vs Predicted (Test Sample)")
    results = pd.DataFrame({
        "Actual Loan Amount": y_test[:10].values,
        "Predicted Loan Amount": y_pred[:10].round(0)
    })
    st.table(results)
