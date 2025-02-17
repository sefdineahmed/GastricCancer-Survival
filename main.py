import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import LabelEncoder

# Configuration de la page
st.set_page_config(page_title="Survival Analysis", layout="wide")

# Initialisation de l'état de session
if 'df' not in st.session_state:
    st.session_state.update({
        'df': None,
        'encoded': False,
        'patient_data': None,
        'models_loaded': False
    })

# Chemins et constantes
MODEL_PATH = "models"
MODELS = {
    "Regression de Cox": "cox.pkl",
    "Random Survival Forest": "rsf.pkl",
    "Gradient Boosting": "gbst.pkl"
}
VARIABLES = [
    'AGE', 'Cardiopathie', 'Ulceregastrique', 'Douleurepigastrique',
    'Ulcero-bourgeonnant', 'Denitrution', 'Tabac', 'Mucineux',
    'Infiltrant', 'Stenosant', 'Metastases', 'Adenopathie',
    'Tempsdesuivi (Mois)', 'Deces', 'Traitement', 'SEXE'
]

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une section :", ["Menu", "Tableau de bord", "Modèles"])

# Section Menu
if page == "Menu":
    st.header("Chargement et préparation des données")
    
    # Chargement des données
    uploaded_file = st.file_uploader("Importer des données", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success("Données chargées avec succès !")
        except Exception as e:
            st.error(f"Erreur de chargement : {str(e)}")
    
    if st.session_state.df is not None:
        # Vérification des colonnes nécessaires
        required_columns = ['Tempsdesuivi (Mois)', 'Deces']
        missing_cols = [col for col in required_columns if col not in st.session_state.df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes : {', '.join(missing_cols)}")
        else:
            # Aperçu des données
            st.subheader("Aperçu des données")
            st.dataframe(st.session_state.df[VARIABLES].head())
            
            # Métriques clés
            st.subheader("Statistiques descriptives")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Nombre de patients", len(st.session_state.df))
            with cols[1]:
                missing = st.session_state.df[VARIABLES].isna().sum().sum()
                st.metric("Valeurs manquantes", missing)
            with cols[2]:
                st.metric("Variables analysées", len(VARIABLES))
            
            # Distribution des variables
            st.subheader("Distribution des variables")
            selected_var = st.selectbox("Choisir une variable", VARIABLES)
            
            try:
                fig, ax = plt.subplots()
                if selected_var == 'AGE':
                    sns.histplot(st.session_state.df[selected_var], kde=True, ax=ax)
                else:
                    counts = st.session_state.df[selected_var].value_counts()
                    if len(counts) > 10:
                        st.warning("Trop de catégories pour l'affichage")
                    else:
                        counts.plot(kind='bar', ax=ax)
                plt.title(f"Distribution de {selected_var}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur de visualisation : {str(e)}")
            
            # Analyse des valeurs manquantes
            st.subheader("Analyse des valeurs manquantes")
            missing_data = st.session_state.df[VARIABLES].isna().sum()
            
            if missing_data.any():
                fig, ax = plt.subplots()
                missing_data[missing_data > 0].plot(kind='bar', ax=ax)
                plt.title("Valeurs manquantes par variable")
                st.pyplot(fig)
            else:
                st.success("Aucune valeur manquante détectée !")
            
            # Encodage des variables
            st.subheader("Préparation des données")
            if st.button("Encoder les variables catégorielles"):
                try:
                    le = LabelEncoder()
                    cat_vars = [v for v in VARIABLES if v not in ['AGE', 'Tempsdesuivi (Mois)']]
                    
                    for col in cat_vars:
                        if st.session_state.df[col].dtype == 'object':
                            st.session_state.df[col] = le.fit_transform(st.session_state.df[col].astype(str))
                    
                    st.session_state.encoded = True
                    st.success("Encodage terminé !")
                except Exception as e:
                    st.error(f"Erreur d'encodage : {str(e)}")

# Autres sections
else:
    if st.session_state.df is None:
        st.warning("Veuillez charger des données dans la section Menu")
    else:
        # Section Tableau de bord
        if page == "Tableau de bord":
            st.header("Analyse de survie")
            
            # Configuration de l'analyse
            time_var = 'Tempsdesuivi (Mois)'
            event_var = 'Deces'
            
            # Courbe de Kaplan-Meier globale
            try:
                kmf = KaplanMeierFitter()
                kmf.fit(st.session_state.df[time_var], st.session_state.df[event_var])
                
                fig, ax = plt.subplots()
                kmf.plot_survival_function(ax=ax)
                plt.title("Courbe de survie globale")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur d'analyse : {str(e)}")
            
            # Analyse stratifiée
            st.subheader("Analyse comparative")
            strat_var = st.selectbox("Variable de stratification", VARIABLES)
            
            try:
                if st.session_state.df[strat_var].nunique() < 5:
                    fig, ax = plt.subplots()
                    kmf = KaplanMeierFitter()
                    
                    for group in st.session_state.df[strat_var].unique():
                        mask = st.session_state.df[strat_var] == group
                        kmf.fit(st.session_state.df[time_var][mask], 
                               st.session_state.df[event_var][mask], 
                               label=f"{strat_var}={group}")
                        kmf.plot_survival_function(ax=ax)
                    
                    plt.title(f"Courbes de survie par {strat_var}")
                    st.pyplot(fig)
                    
                    # Test du log-rank
                    if st.session_state.df[strat_var].nunique() == 2:
                        group0 = st.session_state.df[strat_var].unique()[0]
                        group1 = st.session_state.df[strat_var].unique()[1]
                        
                        results = logrank_test(
                            st.session_state.df[time_var][st.session_state.df[strat_var] == group0],
                            st.session_state.df[time_var][st.session_state.df[strat_var] == group1],
                            st.session_state.df[event_var][st.session_state.df[strat_var] == group0],
                            st.session_state.df[event_var][st.session_state.df[strat_var] == group1]
                        )
                        st.write(f"**Résultat du test log-rank** : p-value = {results.p_value:.4f}")
            except Exception as e:
                st.error(f"Erreur d'analyse : {str(e)}")

        # Section Modèles
        elif page == "Modèles":
            st.header("Prédiction de survie")
            
            # Vérification des modèles
            try:
                models_available = all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in MODELS.values())
                if not models_available:
                    missing_models = [f for f in MODELS.values() if not os.path.exists(os.path.join(MODEL_PATH, f))]
                    st.error(f"Modèles manquants : {', '.join(missing_models)}")
                    st.stop()
            except Exception as e:
                st.error(f"Erreur de vérification des modèles : {str(e)}")
                st.stop()
            
            # Sélection du modèle
            model_option = st.selectbox("Modèle", list(MODELS.keys()))
            model_path = os.path.join(MODEL_PATH, MODELS[model_option])
            
            # Formulaire de prédiction
            st.subheader("Nouveau patient")
            inputs = {}
            cols = st.columns(2)
            
            with cols[0]:
                for var in VARIABLES[:6]:
                    if var == 'AGE':
                        inputs[var] = st.number_input(var, 0, 100, 50)
                    else:
                        inputs[var] = st.selectbox(var, options=[0, 1])
            
            with cols[1]:
                for var in VARIABLES[6:12]:
                    inputs[var] = st.selectbox(var, options=[0, 1])
            
            # Prédiction
            if st.button("Calculer la survie"):
                try:
                    patient_df = pd.DataFrame([inputs])
                    
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    
                    if "Cox" in model_option:
                        survival_func = model.predict_survival_function(patient_df)
                        fig, ax = plt.subplots()
                        survival_func.plot(ax=ax)
                        plt.title("Fonction de survie prédite")
                        st.pyplot(fig)
                    else:
                        prediction = model.predict(patient_df)
                        st.metric("Temps de survie prédit", f"{prediction[0]:.0f} mois")
                    
                    st.session_state.patient_data = patient_df
                except Exception as e:
                    st.error(f"Erreur de prédiction : {str(e)}")
            
            # Sauvegarde
            if st.session_state.patient_data is not None:
                if st.button("Sauvegarder les résultats"):
                    try:
                        st.session_state.patient_data.to_csv("predictions.csv", mode='a', header=False)
                        st.success("Données sauvegardées !")
                    except Exception as e:
                        st.error(f"Erreur de sauvegarde : {str(e)}")