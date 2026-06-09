# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:39:58 2026

@author: lenovo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Unemployment Analysis in India")

df = pd.read_csv("Unemployment in India.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# Top Regions
top_regions = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10,5))
top_regions.head(10).plot(kind="bar", ax=ax)
ax.set_title("Top 10 Regions by Unemployment")
st.pyplot(fig)

# Urban vs Rural
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(data=df,
            x="Area",
            y="Estimated Unemployment Rate (%)",
            ax=ax2)
st.pyplot(fig2)