import streamlit as st
import pickle
import numpy as np

ML_model=pickle.load(open('model.pkl','rb'))

scaler=pickle.load(open('scaler.pkl','rb'))

ohe_encoder = pickle.load(open('ohe_encoder.pkl', 'rb'))

st.title('Placement Prediction System')

age = st.text_input('Age of the student')
cgpa = st.text_input('CGPA')
internships = st.text_input('No. of Internships')
gender = st.text_input('Gender')
stream = st.text_input('Department/Stream')
hostel_status = st.text_input('Does the student stay at college hostel? (y/n)')
backlog_status = st.text_input('Any history of backlogs (y/n)')

if hostel_status == 'y':
        hostel_status = 1
else:
        hostel_status = 0

if backlog_status == 'y':
        backlog_status = 1
else:
        backlog_status = 0

if st.button('Prediction Result'):

        feat_to_be_scaled=np.asarray([age,cgpa,internships]).reshape(1,-1)

        scaled_feat = scaler.transform(feat_to_be_scaled)

        feat_to_be_encoded=np.array([gender,stream]).reshape(1,-1)

        encoded_feat=ohe_encoder.transform(feat_to_be_encoded)

        unaltered_feat=np.asarray([hostel_status,backlog_status]).reshape(1,-1)

        input_features = np.concatenate([scaled_feat ,encoded_feat,unaltered_feat], axis=1)

        prediction = ML_model.predict(input_features)
        
        print(prediction)

        if (prediction[0] == 0):
                st.title('The chances of the student being placed are lower')
        else:
                st.title('It is highly probable that the student will be placed')
        