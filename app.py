# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier")

st.write("Enter flower measurements below:")

# User inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Display result
st.subheader("🔮 Prediction")
st.write(f"The flower is predicted to be: **{prediction}**")

# Probability bar chart
st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
st.bar_chart(prob_df.T)

# Feature Importance
st.subheader("🌟 Feature Importance")
feat_importance = pd.Series(model.feature_importances_, index=["sepal_length", "sepal_width", "petal_length", "petal_width"])
fig, ax = plt.subplots()
sns.barplot(x=feat_importance.values, y=feat_importance.index, palette="viridis", ax=ax)
st.pyplot(fig)
