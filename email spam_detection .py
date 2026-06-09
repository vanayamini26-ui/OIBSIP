# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:56:27 2026

@author: lenovo
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.title("Email Spam Detection")

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Features and Target
X = df['message']
y = df['label']

# Text Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"{accuracy*100:.2f}%")

# User Input
st.subheader("Check Your Email")

email = st.text_area("Enter Email Text")

if st.button("Predict"):
    data = cv.transform([email])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Spam Email")
    else:
        st.success("Not Spam (Ham)")