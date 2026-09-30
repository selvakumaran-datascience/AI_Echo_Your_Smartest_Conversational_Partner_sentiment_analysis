import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

    
with open("C:/Users/Selva.M/Downloads/data_science/mini_projects/project_5/best_review_model.pkl", "rb") as f:
    model = pickle.load(f)
    
st.markdown("""<h1 style='text-align: center; color: #4CAF50;'> AI Sentiment Analysis Dashboard </h1>""",unsafe_allow_html=True)
st.divider()

st.subheader("Customer Experience and Business Analytics")


df = pd.read_csv("C:/Users/Selva.M/Downloads/data_science/mini_projects/project_5/cleaned_chatgpt_style_reviews.csv")
st.write(df.head(10))
vectorizer=model.get("vectorizer")
st.write("Vectorizer features:", len(vectorizer.get_feature_names_out()))

st.header("Rating Distribution")
colors = ['red', 'orange', 'yellow', 'orange', 'green']
fig, ax = plt.subplots(figsize=(8, 5))
df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax, color=colors)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

st.header("Sentiment Distribution")
colors = ['red', 'green', 'yellow']
fig, ax = plt.subplots(figsize=(8, 5))
df['sentiment'].value_counts().plot(kind='bar',color=colors, ax=ax)
st.pyplot(fig)

st.header("Predict Sentiment")
user_input = st.text_area("✍️ Enter a review: ")
st.caption("ℹ️ For better accuracy, please enter a complete sentence or review.")

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


if st.button("Predict"):
    vectorizer=model.get("vectorizer")
    best_model=model.get("model")
    label_map=model.get("label")
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    pred_num = best_model.predict(input_vector)[0]
    pred_label = label_map[pred_num]
    
    # EMOJI mapping
    emoji = {
        "Positive:+": "😊",
        "Neutral:=": "😐",
        "Negative:-": "😞"
    }

    st.success(f"Predicted Sentiment: {pred_label} {emoji.get(pred_label,' ')}")
    

st.divider()

st.markdown("""<h4 style='text-align: center; color: red;'>Developed by SELVAKUMARAN M © 2025</h4>""",unsafe_allow_html=True)