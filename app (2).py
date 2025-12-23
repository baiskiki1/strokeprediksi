
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Load Model and Scaler (pre-trained) ---
@st.cache_resource
def load_artifacts():
    model = load_model('stroke_prediction_model.h5')
    scaler = joblib.load('scaler.joblib') # Load the fitted scaler
    return model, scaler

model, scaler = load_artifacts()

# --- 2. Streamlit App Interface ---
st.title('Stroke Prediction App')
st.write('Enter patient details to predict the likelihood of stroke.')

# --- User Inputs ---
with st.sidebar:
    st.header('Patient Data Input')
    gender = st.selectbox('Gender', ['Male', 'Female']) # Removed 'Other' for consistency with preprocessing
    age = st.slider('Age', 0, 100, 40)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.slider('Average Glucose Level', 50.0, 300.0, 100.0)
    bmi = st.slider('BMI', 10.0, 60.0, 25.0)
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# --- Preprocessing User Input ---
def preprocess_input(gender, age, hypertension, heart_disease, ever_married,
                       work_type, residence_type, avg_glucose_level, bmi, smoking_status, scaler):

    # Create a dictionary for the input features
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # Apply the same preprocessing steps as training
    # The training code removed 'Other' entries, so ensuring consistency here.
    df_input = df_input[df_input['gender'] != 'Other'] # This line handles potential 'Other' if it somehow slips in, or if the selectbox was different.

    # 3. Encode categorical features using one-hot encoding
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # Create dummy columns for all possible categories to ensure consistent feature order
    gender_options = ['Male', 'Female']
    ever_married_options = ['Yes', 'No']
    work_type_options = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    residence_type_options = ['Urban', 'Rural']
    smoking_status_options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

    # Set up categorical type for consistency
    for col, options in zip(categorical_cols, [gender_options, ever_married_options, work_type_options, residence_type_options, smoking_status_options]):
        df_input[col] = pd.Categorical(df_input[col], categories=options)

    df_processed = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Ensure all expected columns from training are present, fill missing with 0
    # This list should ideally be saved with the model or scaler
    expected_columns = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Male', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
    ]

    # Add missing columns with 0
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Reorder columns to match the training data's feature order
    df_processed = df_processed[expected_columns]

    # 4. Scale numerical features (age, avg_glucose_level, bmi)
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    return df_processed

# --- Prediction Button and Logic ---
if st.button('Predict Stroke Risk'):
    processed_input = preprocess_input(gender, age, hypertension, heart_disease, ever_married,
                                       work_type, residence_type, avg_glucose_level, bmi, smoking_status, scaler)

    prediction_proba = model.predict(processed_input)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f'High Risk of Stroke! (Probability: {prediction_proba:.2f})')
    else:
        st.success(f'Low Risk of Stroke. (Probability: {prediction_proba:.2f})')

    st.write('---')
    st.subheader('Input Features:')
    st.write(processed_input)
