# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:07:46 2026

@author: lenovo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Sales Prediction Using Advertising Data")

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Remove serial number column
df = df.drop("Unnamed: 0", axis=1)

# Preview
st.subheader("Dataset Preview")
st.write(df.head())

# Dataset Info
st.subheader("Dataset Shape")
st.write(df.shape)

# Correlation
st.subheader("Correlation Matrix")
st.write(df.corr())

# Features and Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

st.subheader("Model Score (R²)")
st.write(f"{score:.4f}")

# User Inputs
st.subheader("Predict Sales")

tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

if st.button("Predict Sales"):

    prediction = model.predict(
        [[tv, radio, newspaper]]
    )

    st.success(f"Predicted Sales: {prediction[0]:.2f}")