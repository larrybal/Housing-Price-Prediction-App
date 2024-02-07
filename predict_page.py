import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_bathrooms = data["le_bathrooms"]
le_bedrooms = data["le_bedrooms"]

def show_prediction_page():
    st.title("Housing Price Prediction")

    st.write("""### We need some information to predict the housing price""")

    bath = st.slider("Bathrooms", 0, 10, 1)
    bed = st.slider("Bedrooms", 0, 10, 1)
    square_feet = st.slider("Square Feet", 0, 5000, 250)
    acres = st.slider("Acres", 0.0, 5.0, 0.2)

    button = st.button("Calculate Housing Price")
    if button:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({'Bathrooms': [bath], 'Bedrooms': [bed], 'Square Feet': [square_feet], 'Acres': [acres]})

        # Apply transformations to the input data
        input_data['Bathrooms'] = le_bathrooms.transform(input_data['Bathrooms'].astype(str))
        input_data['Bedrooms'] = le_bedrooms.transform(input_data['Bedrooms'].astype(str))
        input_data = input_data.astype(float)
        
        # Make the prediction
        housing_price = regressor.predict(input_data)
        
        # Display the result
        st.subheader(f"The estimated housing price is ${housing_price[0]:,.2f}")


