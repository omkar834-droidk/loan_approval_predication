import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Loan AI Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_data.csv")
    return df

df = load_data()

# ---------------- PREPROCESS ----------------
def preprocess(df):
    df = df.copy()

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    le = LabelEncoder()
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    return X, y

X, y = preprocess(df)

# ---------------- TRAIN ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ---------------- SIDEBAR ----------------
st.sidebar.title("💳 Loan AI")
page = st.sidebar.radio("Navigation",
                        ["Dashboard", "Dataset", "Model", "Predictor", "Insights"])

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 Loan Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Applications", len(df))
    col2.metric("Avg Income", int(df["Applicant_Income"].mean()))
    col3.metric("Approval Rate", f"{(df['Loan_Approved'].mean()*100):.1f}%")

    fig = px.histogram(df, x="Applicant_Income", color="Loan_Approved")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- DATASET ----------------
elif page == "Dataset":
    st.title("📂 Dataset Explorer")
    st.dataframe(df)

# ---------------- MODEL ----------------
elif page == "Model":
    st.title("🤖 Model Performance")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    st.metric("Accuracy", f"{acc*100:.2f}%")

# ---------------- PREDICTOR ----------------
elif page == "Predictor":
    st.title("🔮 Loan Predictor")

    income = st.slider("Income", 0, 100000, 20000)
    loan = st.slider("Loan Amount", 1000, 100000, 30000)
    credit = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 60, 30)

    input_data = pd.DataFrame({
        "Applicant_Income": [income],
        "Loan_Amount": [loan],
        "Credit_Score": [credit],
        "Age": [age]
    })

    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]*100

        if pred == 1:
            st.success(f"Approved ({prob:.2f}%)")
        else:
            st.error(f"Rejected ({prob:.2f}%)")

# ---------------- INSIGHTS ----------------
elif page == "Insights":
    st.title("📈 Insights")

    fig1 = px.box(df, x="Loan_Approved", y="Applicant_Income")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="Applicant_Income", y="Loan_Amount",
                      color="Loan_Approved")
    st.plotly_chart(fig2, use_container_width=True)
