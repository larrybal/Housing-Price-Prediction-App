# explore_page.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Assuming you have the necessary functions and data loaded
from predict_page import load_model

data = load_model()

regressor = data["model"]
le_bathrooms = data["le_bathrooms"]
le_bedrooms = data["le_bedrooms"]

def show_explore_page():
    st.title("Explore Housing Price Predictions")

    st.write("""### Explore housing price prediction trends""")

    # Example: Display a histogram of predicted housing prices
    st.write(
        """
    #### Distribution of Predicted Housing Prices
    """
    )

    # Assuming you have a function to get sample predictions
    # Replace this with your actual data or exploration logic
    sample_data = pd.DataFrame({
        'Bathrooms': np.random.randint(1, 6, 100),
        'Bedrooms': np.random.randint(1, 6, 100),
        'Square Feet': np.random.randint(1000, 4000, 100),
        'Acres': np.random.uniform(0.1, 1.0, 100)
    })

    # Apply transformations to the input data for predictions
    sample_data['Bathrooms'] = le_bathrooms.transform(sample_data['Bathrooms'].astype(str))
    sample_data['Bedrooms'] = le_bedrooms.transform(sample_data['Bedrooms'].astype(str))
    sample_data = sample_data.astype(float)

    # Make predictions
    predicted_prices = regressor.predict(sample_data)

    # Display the predicted prices as a histogram
    st.hist_chart(predicted_prices)

