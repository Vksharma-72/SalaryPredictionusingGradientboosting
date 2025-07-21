# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model (assumes it's a pipeline with preprocessing)
model = joblib.load("GradientBoosting.pkl")

# Set Streamlit app settings
st.set_page_config(page_title="Employee Salary Prediction ", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Ensure these values match your training data
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
workclass = st.sidebar.selectbox("Work Class", [ 
    'Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc'
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Ensure column names match model training
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],   # ⚠️ MATCH EXACT NAME used during training
    'experience': [experience],
    'workclass': [workclass]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict single input
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Batch Prediction Section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())

        # Ensure all required columns are present
        expected_columns = ['age', 'education', 'occupation', 'hours-per-week', 'experience', 'workclass']
        missing_cols = [col for col in expected_columns if col not in batch_data.columns]
        if missing_cols:
            st.error(f"❌ The uploaded file is missing required columns: {missing_cols}")
        else:
            batch_data = batch_data[expected_columns]

            # Predict
            batch_preds = model.predict(batch_data)
            batch_data['PredictedClass'] = batch_preds

            st.write("✅ Predictions:")
            st.write(batch_data.head())

            # Download CSV
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error(f"❌ Batch prediction failed: {e}")