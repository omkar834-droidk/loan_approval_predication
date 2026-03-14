# 💳 CreditWise — ML Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=flat-square\&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-MachineLearning-orange?style=flat-square\&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An **end-to-end Machine Learning web application** that predicts **loan approval eligibility** using financial and demographic features.
The application provides **real-time predictions and probability scoring through an interactive Streamlit dashboard.**

🔗 **Live App:** [credit-risk-app-app-qnfzw2ntuscz6f2nvssjmv.streamlit.app](https://loanapprovalpredication-5lt2rnbpjjhe4edyt7jwwu.streamlit.app/)

---

# 🚀 Project Overview

CreditWise demonstrates a **complete Data Science workflow**:

* Data preprocessing and feature engineering
* Training multiple machine learning models
* Evaluating model performance
* Deploying predictions through an interactive **Streamlit web application**

The model evaluates **loan applications** and predicts whether the loan should be **approved or rejected**, along with a probability score.

---

# 📊 Application Features

### 🔮 Loan Predictor

* Interactive applicant form
* Real-time approval prediction
* Approval probability gauge meter

### 📈 Dashboard

* Total applications
* Loan approval distribution
* Income vs loan relationship
* Financial indicators visualization

### 🤖 Model Performance

* Comparison of top ML models
* Visualization of evaluation metrics

---

# 🧠 Machine Learning Pipeline

### Dataset

* **1000 loan applications**
* **19 raw features**
* Financial, employment, and demographic attributes

Example features:

* Applicant Income
* Co-Applicant Income
* Credit Score
* DTI Ratio
* Savings
* Collateral Value
* Loan Amount
* Loan Term
* Employment Status

---

# ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

* Missing value imputation
* Label encoding for categorical features
* One-hot encoding for nominal features
* Feature scaling using **StandardScaler**

---

# 🧪 Feature Engineering

Additional engineered features:

* `DTI_Ratio_sq` — captures nonlinear debt impact
* `Credit_Score_sq` — captures nonlinear credit effect
* `Applicant_Income_log` — reduces income skew

Final dataset contains **28 engineered features**.

---

# 🤖 Machine Learning Models

The following classification models were trained and evaluated:

| Model            | Precision | Recall | F1 Score  | Accuracy  |
| ---------------- | --------- | ------ | --------- | --------- |
| **XGBoost 🚀**   | 83.8%     | 93.4%  | **88.3%** | **92.5%** |
| Decision Tree 🌳 | 82.3%     | 91.8%  | 86.8%     | 91.5%     |
| Random Forest 🌲 | 82.2%     | 83.6%  | 82.9%     | 89.5%     |

**Best Model:** XGBoost — selected based on highest **F1 Score**, balancing precision and recall.

---

# 🗂️ Project Structure

```
loan_approval_predication
│
├── app.py
├── requirements.txt
├── README.md
│
├── data
│   └── loan_approval_data.csv
│
├── model
│   └── loan_approval_model.pkl
│
├── notebook
│   └── loan_approval.ipynb
```

---

# ⚙️ Run the Project Locally

### 1️⃣ Clone the repository

```
git clone https://github.com/omkar834-droidk/loan_approval_predication.git
cd loan_approval_predication
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```
streamlit run app.py
```

---

# 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
joblib
```

---

# 🛠️ Tech Stack

| Technology     | Purpose                 |
| -------------- | ----------------------- |
| Python         | Core programming        |
| Pandas / NumPy | Data processing         |
| Scikit-learn   | Machine learning models |
| XGBoost        | Gradient boosting model |
| Plotly         | Interactive charts      |
| Streamlit      | Web application         |
| GitHub         | Version control         |

---

# 👤 Author

**Omkar Salunke**

GitHub
https://github.com/omkar834-droidk

---

⭐ If you like this project, consider giving it a **star on GitHub**.

