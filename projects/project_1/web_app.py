import streamlit as st
import pickle
import numpy as np

ML_model=pickle.load(open('model.pkl','rb'))

scaler=pickle.load(open('scaler.pkl','rb'))

st.title('Diabates Predition System')

Pregnancies = st.text_input('Number of Pregnancies')
Glucose = st.text_input('Glucose Level')
BloodPressure = st.text_input('Blood Pressure value')
SkinThickness = st.text_input('Skin Thickness value')
Insulin = st.text_input('Insulin Level')
BMI = st.text_input('BMI value')
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
Age = st.text_input('Age of the Person')

if st.button('Diabetes Test Result'):
        query=np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        # reshape the array as we are predicting for one instance
        query_reshaped = query.reshape(1,-1)

        # standardize the input data
        std_data = scaler.transform(query_reshaped)
        prediction = ML_model.predict(std_data)
        if (prediction[0] == 0):
                st.title('The person is not diabetic')
        else:
                st.title('The person is diabetic')
