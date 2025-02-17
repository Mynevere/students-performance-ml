import streamlit as st
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Add a description and instructions
st.title("🎓 Parashikimi i Performancës së Studentëve")
st.markdown("""
Ky aplikacion parashikon performancën e studentëve bazuar në faktorë të ndryshëm. Plotësoni informacionin 
e studentit në anën e majtë dhe klikoni 'Parashiko Performancën' për të parë rezultatet.
""")

# Move the styling configuration here, after the Streamlit context is established
sns.set(style="whitegrid")

# Create two columns for better organization in the sidebar
st.sidebar.header("📝 Informacioni i Studentit")
col1, col2 = st.sidebar.columns(2)

with col1:
    gender = st.selectbox("Gjinia", ["Mashkull", "Femër"])
    race_ethnicity = st.selectbox("Raca/Etnia", ["Grupi A", "Grupi B", "Grupi C", "Grupi D", "Grupi E"])
    education = st.selectbox("Edukimi i Prindërve", ["Shkollë e Mesme", "Bachelor", "Master"])
    prep_course = st.selectbox("Kursi i Përgatitjes për Test", ["Po", "Jo"])

with col2:
    lunch = st.selectbox("Lloji i Drekës", ["Standarde", "Falas/E Reduktuar"])
    reading_score = st.slider("Rezultati në Lexim", 0, 100, 70, help="Rezultati i vlerësimit të leximit të studentit")
    writing_score = st.slider("Rezultati në Shkrim", 0, 100, 70, help="Rezultati i vlerësimit të shkrimit të studentit")
    study_time = st.selectbox("Koha e Studimit", ["E Ulët", "Mesatare", "E Lartë"])

# Add model selection with description
st.sidebar.markdown("---")
st.sidebar.header("🤖 Zgjedhja e Modelit")
model_choice = st.sidebar.selectbox(
    "Zgjidhni Modelin",
    ["RandomForest", "LinearRegression", "SVM"],
    help="Zgjidhni modelin e machine learning për parashikim"
)

# Style the predict button
predict_button = st.sidebar.button(
    "Parashiko Performancën",
    type="primary",
    use_container_width=True
)

# Predict button
if predict_button:
    with st.spinner('Duke llogaritur parashikimin...'):
        student_data = {
            "gender": 1 if gender == "Mashkull" else 0,
            "race/ethnicity": race_ethnicity,
            "parental level of education": education,
            "lunch": 1 if lunch == "Standarde" else 0,
            "test preparation course": 1 if prep_course == "Po" else 0,
            "reading score": reading_score,
            "writing score": writing_score,
            "study time": study_time
        }

        # Send prediction request to API
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"models": model_choice, "inputs": student_data}
        )

        if response.status_code == 200:
            prediction = response.json()["Performance Level"]
            confidence = response.json().get("confidence", 0.85)

            # Create evaluation data with Albanian labels
            evaluation_data = {
                "Niveli i Performancës": ["I Lartë", "I Ulët", "Mesatar", "nan"],
                "Precizioni": [0.84, 0.66, 0.71, 0.00],
                "Recall": [0.83, 0.60, 0.75, 0.00],
                "F1-Score": [0.83, 0.63, 0.73, 0.00],
                "Mbështetja": [75, 35, 89, 1]
            }
            eval_df = pd.DataFrame(evaluation_data)

            # Then display metrics and visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Niveli i Parashikuar i Performancës",
                    value=prediction,
                    delta=f"{confidence * 100:.1f}% besueshmëri"
                )
            
            with col2:
                st.metric(
                    label="Modeli i Përdorur",
                    value=model_choice
                )

            # Now we can use eval_df safely
            with st.expander("Shiko Metrikat e Detajuara të Modelit"):
                st.dataframe(
                    eval_df.style.background_gradient(cmap='Blues'),
                    hide_index=True
                )

            # Display performance bar chart
            performance_data = {"Performance Level": [prediction, "Low", "Medium", "High"], "Confidence": [confidence, 0, 0, 0]}
            performance_df = pd.DataFrame(performance_data)

            # Update the visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Performance Level", y="Confidence", data=performance_df, palette="Blues_d", ax=ax)
            plt.title("Besueshmëria e Parashikimit të Performancës")
            ax.set_ylabel("Rezultati i Besueshmërisë")
            ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.error("❌ Gabim: Nuk mund të merrej parashikimi. Ju lutem provoni përsëri.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Krijuar me ❤️ duke përdorur Streamlit | Projekt Master</small>
</div>
""", unsafe_allow_html=True)
