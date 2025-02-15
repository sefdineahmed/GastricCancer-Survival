# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import shap

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Survie - Cancer Gastrique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#292929,#2F4F4F);
    color: white;
}
h1 {
    color: #2F4F4F;
    border-bottom: 3px solid #2F4F4F;
}
</style>
""", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_excel("data/GastricCancerData.xlsx", skiprows=1)
    # Préprocessing...
    return df

df = load_data()

# Entraînement des modèles
@st.cache_resource
def train_models(_df):
    # Modèle Cox
    cph = CoxPHFitter()
    cph.fit(_df, duration_col='Tempsdesuivi (Mois)', event_col='Deces')
    
    # Modèle Random Survival Forest
    rsf = RandomSurvivalForest()
    rsf.fit(X_train, y_train)
    
    return cph, rsf

cph, rsf = train_models(df)

# Sidebar - Menu créatif
with st.sidebar:
    st.title("🔍 Navigation")
    menu = st.radio("", ["Accueil", "Exploration des Données", "Analyse de Survie", "Modèles Prédictifs", "À Propos"])
    
    st.markdown("---")
    st.info("""
    **Auteur** : Votre Nom  
    **Version** : 1.0  
    """)

# Contenu principal
if menu == "Accueil":
    st.title("🩺 Plateforme d'Analyse de Survie en Oncologie Digestive")
    st.image("https://example.com/medical-banner.jpg", use_column_width=True)
    st.markdown("""
    ## Bienvenue sur l'interface d'analyse prédictive
    **Exploration interactive des données de survie des patients atteints de cancer gastrique**
    """)

elif menu == "Exploration des Données":
    st.title("🔎 Exploration des Données Cliniques")
    
    with st.expander("Aperçu des Données Brutes"):
        st.dataframe(df.style.highlight_max(axis=0), height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des Variables")
        selected_var = st.selectbox("Choisir une variable", df.columns)
        fig, ax = plt.subplots()
        df[selected_var].hist(ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Matrice de Corrélation")
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

elif menu == "Analyse de Survie":
    st.title("📈 Analyse de Survie par Kaplan-Meier")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        variable = st.selectbox("Variable Stratificatrice", df.columns)
        time_col = 'Tempsdesuivi (Mois)'
        event_col = 'Deces'
    
    with col2:
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10,6))
        
        for value in df[variable].unique():
            mask = df[variable] == value
            kmf.fit(df[mask][time_col], df[mask][event_col], label=f"{variable}={value}")
            kmf.plot_survival_function()
            
        st.pyplot(plt.gcf())

elif menu == "Modèles Prédictifs":
    st.title("🤖 Modèles Prédictifs de Survie")
    
    tab1, tab2, tab3 = st.tabs(["Cox PH", "Random Forest", "SHAP Analysis"])
    
    with tab1:
        st.subheader("Modèle à Risques Proportionnels de Cox")
        st.write(cph.print_summary())
        
    with tab2:
        st.subheader("Forêt de Survie Aléatoire")
        st.metric("C-index", f"{rsf.score(X_test, y_test):.2f}")
        
    with tab3:
        st.subheader("Interprétabilité par SHAP")
        explainer = shap.TreeExplainer(rsf)
        shap_values = explainer.shap_values(X_train)
        fig = shap.summary_plot(shap_values, X_train)
        st.pyplot(fig)

elif menu == "À Propos":
    st.title("📚 Documentation Scientifique")
    st.markdown("""
    ## Méthodologie
    - **Sources de données** : Registre hospitalier 2015-2020
    - **Algorithmes implémentés** : 
        - Modèle de Cox
        - Random Survival Forest
    """)
