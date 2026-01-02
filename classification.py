import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Loan Classification (Random Forest)", layout="wide")

st.title("🎯 Loan Approval Classification (Random Forest)")
st.markdown("""
This app predicts the **Loan Status** (Approved vs Rejected) using a Random Forest Classifier.
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
    
    X = df_p.drop(['loan_id', 'loan_status'], axis=1)
    y = df_p['loan_status']
    return X, y, le

st.header("📊 Dataset Overview")
st.dataframe(df.head(10))

X, y, le_status = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

if st.button("Train Classifier"):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.success("Model trained successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
    
    with col2:
        st.subheader("Feature Importance")
        feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette="viridis")
        st.pyplot(fig)
        
    st.subheader("Actual vs Predicted (Test Sample)")
    results = pd.DataFrame({
        "Actual": le_status.inverse_transform(y_test[:10]),
        "Predicted": le_status.inverse_transform(y_pred[:10])
    })
    st.table(results)
