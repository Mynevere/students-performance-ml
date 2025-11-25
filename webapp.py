import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Predictor")

st.sidebar.header("📥 Student Information")

gender_map = {"Male": "male", "Female": "female"}
prep_map = {"Yes": "completed", "No": "none"}
lunch_map = {"Standard": "standard", "Free/Reduced": "free/reduced"}
edu_map = {
    "High School": "high school",
    "Some High School": "some high school",
    "Some College": "some college",
    "Associate's Degree": "associate's degree",
    "Bachelor": "bachelor's degree",
    "Master": "master's degree"
}
race_map = {"Group A": "group A", "Group B": "group B", "Group C": "group C", "Group D": "group D", "Group E": "group E"}

gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", list(race_map.keys()))
parental_education = st.sidebar.selectbox("Parental Education", list(edu_map.keys()))
lunch = st.sidebar.selectbox("Lunch Type", list(lunch_map.keys()))
prep_course = st.sidebar.selectbox("Test Preparation Course", list(prep_map.keys()))
model_choice = st.sidebar.selectbox("Select Model", ["LinearRegression", "RandomForest", "SVM"])

st.subheader("🔮 Predict Student Performance")
if st.button("Run Prediction"):
    data = {
        "gender": gender_map[gender],
        "race_ethnicity": race_map[race_ethnicity],
        "parental_education": edu_map[parental_education],
        "lunch": lunch_map[lunch],
        "test_preparation_course": prep_map[prep_course]
    }
    response = requests.post(f"http://localhost:8000/predict/{model_choice}", json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Score: {round(result['prediction'], 2)}")
    else:
        st.error("Prediction failed. Check API connection.")

st.markdown("## 📊 Gender Comparison Analysis")
if st.button("Show Gender Analysis"):
    response = requests.get("http://localhost:8000/gender-comparison")
    if response.status_code == 200:
        result = pd.DataFrame(response.json())
        result = result.rename_axis("Gender").reset_index()
        st.dataframe(result)
        st.markdown("#### Average Scores by Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        result.set_index("Gender").T.plot(kind="bar", ax=ax)
        plt.title("Average Scores by Gender")
        plt.ylabel("Average Score")
        st.pyplot(fig)
    else:
        st.error("Unable to load gender comparison data.")

st.markdown("## 🔍 Feature Importance (Random Forest)")
if st.button("Show Feature Importance"):
    response = requests.get("http://localhost:8000/feature-importance")
    if response.status_code == 200:
        importance = pd.Series(response.json()).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        importance.plot(kind="barh", ax=ax)
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        st.pyplot(fig)
    else:
        st.error("Unable to load feature importance.")
