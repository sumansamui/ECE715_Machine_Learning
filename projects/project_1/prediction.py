import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler

ML_model = pickle.load(open('model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))

print("Welcome to Diabetes prediction system!")

print('please enter your features/attributes, so that model can predict')

a=input("Number of times pregnant:")

b=input("Plasma glucose concentration:")

c=input("Diastolic blood pressure:")

d=input("Triceps skin fold thickness:")

e=input("2-Hour serum insulin:")

f=input("Body mass index:")

g=input("Diabetes pedigree function:")

h=input("Age (years):")


input_data = (a,b,c,d,e,f,g,h)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = ML_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')