# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:26:04 2026

@author: lenovo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Iris Flower Classification")

# Load Dataset
df = pd.read_csv("Iris.csv")

# Dataset Preview
st.subheader("Dataset Preview")
st.write(df.head())

# Dataset Information
st.subheader("Dataset Shape")
st.write(df.shape)

# Check Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Species Distribution
st.subheader("Species Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Species", ax=ax)
st.pyplot(fig)

# Features and Target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy*100:.2f}%")

# User Input
st.subheader("Predict Iris Species")

sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    prediction = model.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )

    st.success(f"Predicted Species: {prediction[0]}")