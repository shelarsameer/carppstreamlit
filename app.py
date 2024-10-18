import streamlit as st
import pandas as pd
import joblib

# Load the saved model and label encoders
dt_regressor = joblib.load('dt_regressor_model.joblib')
le_model = joblib.load('le_model.joblib')
le_transmission = joblib.load('le_transmission.joblib')
le_fuelType = joblib.load('le_fuelType.joblib')

st.title('Audi Car Price Predictor')

# Create input fields for user
model = st.selectbox('Model', le_model.classes_)
year = st.number_input('Year', min_value=2000, max_value=2023, value=2020)
transmission = st.selectbox('Transmission', le_transmission.classes_)
mileage = st.number_input('Mileage', min_value=0, value=10000)
fuel_type = st.selectbox('Fuel Type', le_fuelType.classes_)
tax = st.number_input('Tax', min_value=0, value=150)
mpg = st.number_input('MPG', min_value=0.0, value=50.0)
engine_size = st.number_input('Engine Size', min_value=0.0, value=2.0, step=0.1)

# Create a dataframe with user input
input_data = pd.DataFrame({
    'model': [le_model.transform([model])[0]],
    'year': [year],
    'transmission': [le_transmission.transform([transmission])[0]],
    'mileage': [mileage],
    'fuelType': [le_fuelType.transform([fuel_type])[0]],
    'tax': [tax],
    'mpg': [mpg],
    'engineSize': [engine_size]
})

# Make prediction when user clicks the button
if st.button('Predict Price'):
    prediction = dt_regressor.predict(input_data)[0]
    st.success(f'The predicted price for this Audi car is £{prediction:.2f}')

st.write('Note: This is a simple model and predictions may not be entirely accurate.')