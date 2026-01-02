import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Loan Approval Analysis", layout="wide")

st.title("🏦 Loan Approval Analysis using Random Forest")
st.markdown("""
This application implements **Random Forest** for both **Classification** (predicting loan status) 
and **Regression** (predicting loan amount) using the Loan Approval Dataset.
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the analysis mode", ["Data Overview", "RF Classification", "RF Regression"])

# Preprocessing function
def preprocess_data(df, target_col, mode='classification'):
    df_processed = df.copy()
    le = LabelEncoder()
    
    # Encode categorical columns
    categorical_cols = ['education', 'self_employed', 'loan_status']
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col].str.strip())
    
    if mode == 'classification':
        X = df_processed.drop(['loan_id', 'loan_status'], axis=1)
        y = df_processed['loan_status']
    else:
        # For regression, predict loan_amount
        X = df_processed.drop(['loan_id', 'loan_amount'], axis=1)
        y = df_processed['loan_amount']
        
    return X, y, df_processed

if app_mode == "Data Overview":
    st.header("📊 Dataset Overview")
    st.write("First 10 rows of the dataset:")
    st.dataframe(df.head(10))
    
    st.write("Statistical Summary:")
    st.write(df.describe())
    
    st.write("Column Information:")
    st.write(df.dtypes)

elif app_mode == "RF Classification":
    st.header("🎯 Random Forest Classification")
    st.subheader("Predicting Loan Status (Approved vs Rejected)")
    
    X, y, df_p = preprocess_data(df, 'loan_status', mode='classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 20, 10)
        
        if st.button("Train Classifier"):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.success("Model trained successfully!")
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            # Feature Importance
            st.subheader("Feature Importance")
            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette="viridis")
            ax.set_title("Relative importance of features")
            st.pyplot(fig)

    with col2:
        st.write("Actual vs Predicted (Test Sample):")
        # Map back to labels for readability
        label_map = {1: "Rejected", 0: "Approved"} # Typical encoding for loan_status string strip
        # Need to be careful with LabelEncoder mapping
        le = LabelEncoder()
        le.fit(df['loan_status'].str.strip())
        
        if 'model' in locals():
            results = pd.DataFrame({
                "Actual": le.inverse_transform(y_test[:10]),
                "Predicted": le.inverse_transform(y_pred[:10])
            })
            st.table(results)

elif app_mode == "RF Regression":
    st.header("📈 Random Forest Regression")
    st.subheader("Predicting Loan Amount")
    
    X, y, df_p = preprocess_data(df, 'loan_amount', mode='regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 20, 10)
        
        if st.button("Train Regressor"):
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.success("Model trained successfully!")
            st.metric("R2 Score", f"{r2_score(y_test, y_pred):.4f}")
            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):,.0f}")
            
            # Feature Importance
            st.subheader("Feature Importance")
            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette="magma")
            ax.set_title("Relative importance of features")
            st.pyplot(fig)

    with col2:
        if 'y_pred' in locals():
            st.write("Actual vs Predicted (Test Sample):")
            results = pd.DataFrame({
                "Actual Loan Amount": y_test[:10].values,
                "Predicted Loan Amount": y_pred[:10].round(0)
            })
            st.table(results)
            
            # Prediction Plot
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
