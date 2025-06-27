import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="wide")

# Load Model and Scaler
try:
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Model and Scaler Loaded Successfully")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# Sidebar Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# Apply dark mode styles
if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #121212; color: white; }
            .stButton>button { background-color: #bb86fc; color: white; }
            .sidebar .sidebar-content { background-color: #333333; color: white; }
            h1, h2, h3, h4, h5, h6 { color: #ff6f61; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Title ---
st.title("💓 AI-Powered Heart Disease Prediction")
st.write("🔬 **A Smart Healthcare Solution to Predict Heart Disease Risks.**")

# --- Sidebar for Patient Input ---
st.sidebar.header("🩺 Enter Patient Details")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.radio("Sex", ["Female", "Male"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol Level (mg/dL)", 100, 500, 200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
ekg_results = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Probable LV Hypertrophy"])
max_hr = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.radio("Exercise-Induced Angina", ["No", "Yes"])
st_depression = st.sidebar.slider("ST Depression", 0.0, 10.0, 1.0)
slope_st = st.sidebar.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
vessels = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thallium = st.sidebar.selectbox("Thallium Stress Test Result", [3, 6, 7])

# BMI Calculator
st.sidebar.subheader("⚖️ BMI Calculator")
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
bmi = weight / ((height / 100) ** 2)
st.sidebar.write(f"**Your BMI:** {bmi:.2f} ({'Underweight' if bmi < 18.5 else 'Normal' if bmi < 24.9 else 'Overweight' if bmi < 29.9 else 'Obese'})")

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
chest_pain = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}[chest_pain]
fbs = 1 if fbs == "Yes" else 0
ekg_results = {"Normal": 0, "ST-T Wave Abnormality": 1, "Probable LV Hypertrophy": 2}[ekg_results]
exercise_angina = 1 if exercise_angina == "Yes" else 0
slope_st = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope_st]

# --- Prediction Section ---
st.markdown("### 🧪 Prediction Results")

if st.sidebar.button("🔍 Predict"):
    try:
        # Prepare input data
        input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg_results, 
                                max_hr, exercise_angina, st_depression, slope_st, vessels, thallium]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # --- Risk Meter Visualization ---
        risk_level = "High Risk 🚨" if prediction == 1 else "Low Risk ✅"
        color = "#E63946" if prediction == 1 else "#28A745"

        st.markdown(f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: white; background-color: {color}; padding: 10px; border-radius: 10px;">
                {risk_level} (Confidence: {probability:.2f}%)
            </div>
        """, unsafe_allow_html=True)

        # --- Pie Chart ---
        fig, ax = plt.subplots(figsize=(4, 4))
        labels = ['No Disease', 'Heart Disease']
        colors = ['#28A745', '#E63946']
        ax.pie([100 - probability, probability], labels=labels, autopct='%1.0f%%', colors=colors, startangle=90)
        st.pyplot(fig)

        # --- Feature Importance Visualization ---
        st.subheader("📊 Feature Importance")
        feature_names = ["Age", "Sex", "CP", "BP", "Cholesterol", "FBS", "ECG", "Max HR", "Angina", "ST Depression", "Slope", "Vessels", "Thal"]
        feature_importance = model.feature_importances_
        fig_importance = px.bar(x=feature_importance, y=feature_names, orientation='h', title="Feature Importance in Prediction")
        st.plotly_chart(fig_importance)

        # --- Download Report ---
        report = f"""Patient Report:\nAge: {age}\nSex: {sex}\nRisk: {'High' if prediction == 1 else 'Low'}\nConfidence: {probability:.2f}%"""
        st.download_button(label="📄 Download Report", data=report, file_name="heart_disease_report.txt")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

# Chatbot
st.sidebar.subheader("💬 Health Assistant Chatbot")
user_question = st.sidebar.text_input("Ask me anything about heart health:")
responses = {
    "hello": "Hi! How can I assist you?",
    "what is heart disease?": "Heart disease refers to various conditions affecting heart function.",
    "symptoms of heart disease": "Common symptoms include chest pain, shortness of breath, and dizziness.",
    "how to prevent heart disease?": "Maintain a healthy diet, exercise regularly, and avoid smoking."
}
st.sidebar.write("🤖 AI Response: ", responses.get(user_question.lower(), "I'm sorry, I don't understand that."))

# Doctor Consultation
st.markdown("### 💬 Need Expert Advice?")
st.button("📞 Connect with a Doctor", on_click=lambda: st.markdown("📌 **Call: +1-800-HEALTHCARE**"))

# Footer
st.markdown("---")
st.markdown("💡 **Note:** This prediction is AI-based and should not replace medical consultation.")


# venv\Scripts\activate
# streamlit run app.py

"""
import streamlit as st
import numpy as np
import joblib

# Set up page title and layout
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load the trained model and scaler
try:
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Model and Scaler Loaded Successfully")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# Title
st.title("💓 Heart Disease Prediction App")

# User Input Form
st.sidebar.header("Enter Patient Details:")
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain = st.sidebar.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4])
bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol Level (mg/dL)", 100, 500, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL (0 = No, 1 = Yes)", [0, 1])
ekg_results = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
max_hr = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", [0, 1])
st_depression = st.sidebar.slider("ST Depression (0.0 - 10.0)", 0.0, 10.0, 1.0)
slope_st = st.sidebar.selectbox("Slope of ST Segment (1-3)", [1, 2, 3])
vessels = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thallium = st.sidebar.selectbox("Thallium Stress Test Result (3, 6, 7)", [3, 6, 7])

# Convert inputs to numpy array
input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg_results, 
                        max_hr, exercise_angina, st_depression, slope_st, vessels, thallium]])

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        if prediction == 1:
            st.error("🚨 **Prediction: High risk of Heart Disease!**")
        else:
            st.success("✅ **Prediction: No Heart Disease detected.**")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Consult a medical professional for accurate diagnosis.")

"""
