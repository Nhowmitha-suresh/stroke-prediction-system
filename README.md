Stroke Prediction System 🧠
Overview

The Stroke Prediction System is a simple web application built using Streamlit. It predicts the risk of stroke in individuals based on their personal, medical, and lifestyle information. The system displays:

Stroke risk probability
Risk status (High or Low)
Comparison charts of BMI, Age, and Glucose levels with dataset averages

Session history of predictions

This project is intended for educational purposes and preliminary risk assessment.

Features

User-friendly input form with patient details

Real-time stroke risk prediction using a trained ML model

Visual comparison with dataset averages

Session history of all predictions

Background image for a professional look

Requirements

Python 3.10+

Libraries: streamlit, pandas, numpy, matplotlib, joblib, scikit-learn

Install dependencies:

pip install streamlit pandas numpy matplotlib scikit-learn joblib

Files

Make sure the following files are in the project folder:

app.py – Streamlit application

model.pkl – Pre-trained stroke prediction model

cleaned_stroke_data.csv – Dataset for reference

How to Run

Open VS Code and navigate to the project folder.

Activate your virtual environment (optional but recommended):

.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux


Run the app:

streamlit run app.py


Open the local URL in your browser (http://localhost:8501).

Enter patient information and click Predict to see results.

Input Features

Gender: Male, Female, Other

Age: 0–120

Hypertension: 0 = No, 1 = Yes

Heart Disease: 0 = No, 1 = Yes

Ever Married: Yes, No

Work Type: Private, Self-employed, Govt_job, Children, Never_worked

Residence Type: Urban, Rural

Average Glucose Level: mg/dL

BMI: Body Mass Index

Smoking Status: formerly smoked, never smoked, smokes, Unknown

Author

Nhowmitha Suresh
3rd Year AI & DS, Roll No: 23AIA63
