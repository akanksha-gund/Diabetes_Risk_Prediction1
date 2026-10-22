import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration and light theme settings
st.set_page_config(
    page_title="Health Prediction ML App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Theme & Attractive UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained SVM model safely
@st.cache_resource
def load_model():
    return joblib.load('svm.pkl')

model = load_model()

# Header Section
st.markdown("<h1 style='text-align: center; color: #0d6efd;'>🏥 Health & Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>Please fill in your details below to get predictions from the ML model.</p>", unsafe_allow_html=True)
st.write("---")

# Input Form
with st.form("prediction_form"):
    st.subheader("📋 Patient Details Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.text_input("City", "Mumbai")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
        family_history_diabetes = st.selectbox("Family History Diabetes", ["Yes", "No"])
        physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        
    with col2:
        diet_type = st.selectbox("Diet Type", ["Veg", "Non-Veg", "Vegan"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Low", "High"])
        hours_sleep_per_night = st.number_input("Hours Sleep Per Night", min_value=1.0, max_value=24.0, value=7.0)
        stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50.0, max_value=300.0, value=100.0)
        
    with col3:
        hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
        blood_pressure_systolic = st.number_input("Blood Pressure Systolic", min_value=80, max_value=200, value=120)
        blood_pressure_diastolic = st.number_input("Blood Pressure Diastolic", min_value=50, max_value=130, value=80)
        waist_circumference_cm = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=80.0)
        income_bracket = st.selectbox("Income Bracket", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("🚀 Predict Result")

if submitted:
    # Note: If your model requires LabelEncoder or OneHotEncoder for categorical variables, 
    # the corresponding preprocessing must be added here.
    
    st.success("Data successfully processed! Displaying model output.")
