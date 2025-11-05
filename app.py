import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Mapping dictionaries
# -----------------------------
gender_map = {"Male": 1, "Female": 0, "Other": 2}  
ever_married_map = {"Yes": 1, "No": 0}
work_type_map = {
    "Private": 2,
    "Self-employed": 3,
    "Govt_job": 1,
    "Children": 0,
    "Never_worked": 4
}
residence_map = {"Urban": 1, "Rural": 0}
smoking_map = {
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3,
    "Unknown": 0
}

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load your trained stroke model
# -----------------------------
try:
    model = joblib.load("model.pkl")  # <- make sure this is your stroke model file
except FileNotFoundError:
    st.error("Stroke model file not found! Make sure 'model.pkl' exists in the folder.")
    st.stop()

# Load dataset for comparison
try:
    df = pd.read_csv("cleaned_stroke_data.csv")
except FileNotFoundError:
    st.error("Dataset file not found! Make sure 'cleaned_stroke_data.csv' exists in the folder.")
    st.stop()

# -----------------------------
# Background image
# -----------------------------
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1588776814546-c28d2d7a7b07?auto=format&fit=crop&w=1470&q=80");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🧠 Stroke Prediction Info")
    st.markdown("""
    **Stroke Prevention Tips:**  
    - Maintain healthy blood pressure  
    - Exercise regularly  
    - Eat a balanced diet  
    - Avoid smoking and alcohol  
    - Monitor BMI and glucose levels
    """)
    st.markdown("---")
    st.subheader("Dataset Stats")
    st.write("Total Records:", df.shape[0])
    st.write("Stroke Cases:", df['stroke'].sum())

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Stroke Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)

# -----------------------------
# Input fields
# -----------------------------
st.subheader("Patient Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=20.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# -----------------------------
# Encode inputs
# -----------------------------
input_dict = {
    "gender": gender_map[gender],
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married_map[ever_married],
    "work_type": work_type_map[work_type],
    "Residence_type": residence_map[Residence_type],
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_map[smoking_status]
}
input_data = np.array([list(input_dict.values())])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        st.markdown(f"<h3 style='text-align: center; color: white;'>Probability of Stroke: {prediction_proba*100:.2f}%</h3>", unsafe_allow_html=True)
        st.progress(int(prediction_proba*100))
        
        if prediction == 1:
            st.markdown("<h2 style='text-align: center; color: red;'>⚠️ High Risk of Stroke!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>✅ Low Risk of Stroke!</h2>", unsafe_allow_html=True)
        
        # Visualization: Patient vs Dataset
        st.subheader("📊 Your Stats vs Dataset Average")
        stats = pd.DataFrame({
            "Metric": ["BMI", "Average Glucose Level", "Age"],
            "Your Value": [bmi, avg_glucose_level, age],
            "Dataset Average": [df["bmi"].mean(), df["avg_glucose_level"].mean(), df["age"].mean()]
        })
        
        fig, ax = plt.subplots()
        x = np.arange(len(stats["Metric"]))
        width = 0.35
        ax.bar(x - width/2, stats["Your Value"], width, label='Your Value', color='skyblue')
        ax.bar(x + width/2, stats["Dataset Average"], width, label='Dataset Avg', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(stats["Metric"])
        ax.set_ylabel("Value")
        ax.set_title("Patient vs Dataset Comparison")
        ax.legend()
        st.pyplot(fig)
        
        # Session history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'Age': age,
            'Prediction': 'High Risk' if prediction==1 else 'Low Risk',
            'Probability': f"{prediction_proba*100:.2f}%"
        })
        st.subheader("📝 Prediction History")
        st.table(pd.DataFrame(st.session_state['history']))
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
