# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:33:29 2026

@author: lenovo
"""

import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# =====================================
# PAGE TITLE
# =====================================

st.set_page_config(page_title="Car Price Predictor")
st.title("🚗 Car Price Prediction System")

# =====================================
# LOAD DATASET
# =====================================

@st.cache_data
def load_data():

    df = pd.read_csv("quikr_car.csv")

    # Remove invalid prices
    df = df[df['Price'] != 'Ask For Price']

    # Clean Price column
    df['Price'] = df['Price'].astype(str)
    df['Price'] = df['Price'].str.replace(',', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Clean year column
    df = df[df['year'].astype(str).str.isnumeric()]
    df['year'] = df['year'].astype(int)

    # Clean kms_driven column
    df['kms_driven'] = df['kms_driven'].astype(str)
    df = df[df['kms_driven'].str.contains('kms', na=False)]

    df['kms_driven'] = df['kms_driven'].str.replace(' kms', '', regex=False)
    df['kms_driven'] = df['kms_driven'].str.replace(',', '', regex=False)

    df = df[df['kms_driven'].str.isnumeric()]
    df['kms_driven'] = df['kms_driven'].astype(int)

    # Remove null values
    df.dropna(inplace=True)

    return df


df = load_data()

# =====================================
# TRAIN MODEL
# =====================================

X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

ohe = OneHotEncoder(handle_unknown='ignore')

column_transformer = ColumnTransformer(
    [('encoder', ohe, ['name', 'company', 'fuel_type'])],
    remainder='passthrough'
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipe = Pipeline([
    ('preprocessor', column_transformer),
    ('model', model)
])

pipe.fit(X, y)

# =====================================
# INPUT FORM
# =====================================

with st.form("prediction_form"):

    st.subheader("Enter Car Details")

    name = st.selectbox(
        "Car Name",
        sorted(df['name'].unique())
    )

    company = st.selectbox(
        "Company",
        sorted(df['company'].unique())
    )

    year = st.selectbox(
        "Year",
        sorted(df['year'].unique(), reverse=True)
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        sorted(df['fuel_type'].unique())
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000,
        step=1000
    )

    submit = st.form_submit_button("Predict Price")

# =====================================
# PREDICTION
# =====================================

if submit:

    input_df = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })

    prediction = pipe.predict(input_df)

    st.success(
        f"Estimated Car Price: ₹ {int(prediction[0]):,}"
    )