import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Approval AI", layout="wide")

st.title("💳 Loan Approval Prediction System")

# ------------------ LOAD DATA SAFELY ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("loan_approval_data.csv")
        return df
    except:
        st.error("❌ CSV file not found. Please upload it.")
        return None

df = load_data()

if df is None:
    st.stop()

# ------------------ PREPROCESS ------------------
def preprocess(df):
    df = df.copy()

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    if "Applicant_ID" in df.columns:
        df.drop("Applicant_ID", axis=1, inplace=True)

    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    return X, y

X, y = preprocess(df)

# ------------------ TRAIN ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ------------------ UI ------------------
st.sidebar.header("Input Details")

income = st.sidebar.number_input("Applicant Income", 0, 100000, 15000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 100000, 20000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 60, 30)

# Dummy input (simple version)
input_data = pd.DataFrame({
    "Applicant_Income": [income],
    "Loan_Amount": [loan_amount],
    "Credit_Score": [credit_score],
    "Age": [age]
})

# Align columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.sidebar.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    if pred == 1:
        st.success(f"✅ Loan Approved ({prob:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected ({prob:.2f}%)")

# ------------------ DATA VIS ------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

fig = px.histogram(df, x="Applicant_Income", color="Loan_Approved")
st.plotly_chart(fig)
