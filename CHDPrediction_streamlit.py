#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries

import numpy as np
import pickle
import streamlit as st


# In[2]:


# Load files

with open('logistic_regression_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


# ```
# ---
# 2 pickle files:
# 
# logistic_regression_model.pkl -> contains the trained model
# scaler.pkl -> contains the fitted scalers (or any other preprocessing transformer)
# ---
# ```

# In[4]:


# Create prediction function

def CHD_prediction(input_data):
    input_data_nparray = np.asarray(input_data) # Change data into NumPy array
    input_data_reshaped = input_data_nparray.reshape(1, -1) # Reshape array with 1 column (1) and as many rows to accomodate data (-1)
    
    new_data_scaled = loaded_scaler.transform(input_data_reshaped) 
    prediction = loaded_model.predict(new_data_scaled)

    if prediction == 0:
        return 'Individual is NOT PREDICTED to develop Coronary Heart Disease (CHD) within the next 10 years'
    else:
        return 'Individual IS PREDICTED to develop Coronary Heart Disease (CHD) within the next 10 years'

def main():
    # Title
    st.title('10-year risk prediction for developing CHD')

    # User input
    male = st.text_input('Gender (1 = male, 2 = female):')
    age = st.text_input('Age:')
    education = st.text_input('Education (1 = some HS, 2 = HS grad, 3 = some college, 4 = college grad):')
    currentSmoker = st.text_input('Curent smoker (1 = yes, 2 = no):')
    cigsPerDay = st.text_input('Number of cigarettes per day:')
    BPMeds = st.text_input('Takes blood pressure medication(s) (1 = yes, 2 = no):')
    prevalentStroke = st.text_input('History of stroke (1 = yes, 2 = no):')
    prevalentHyp = st.text_input('History of high blood pressure (1 = yes, 2 = no):')
    diabetes = st.text_input('History of diabetes (1 = yes, 2 = no):')
    totChol = st.text_input('Total cholesterol:')
    sysBP = st.text_input('Systolic blood pressure:')
    diaBP = st.text_input('Diastolic blood pressure:')
    BMI = st.text_input('BMI:')
    heartRate = st.text_input('Heart rate:')
    glucose = st.text_input('Blood sugar:')

    # CHD prediction
    dx = ''

    # Button for CHD prediction
    if st.button('CHD Prediction'):
        dx = CHD_prediction(
            [male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
            diaBP, BMI, heartRate, glucose])
    st.success(dx)

if __name__ == '__main__':
    main()

