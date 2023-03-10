import numpy as np
import pandas as pd 
import seaborn as sns 
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

def shap_plots(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig, axes = plt.subplots(ncols=len(X.columns), figsize=(6*len(X.columns),6))
    for i, feature in enumerate(X.columns):
        shap.dependence_plot(feature, shap_values[1], X, show=False, ax=axes[i])
    st.pyplot(fig)



def ML_inter():
     
    df=pd.read_csv("./Data/Data_ML.csv")
    salaires = pd.read_csv("./Data/salaires_dp.csv")
    st.title("Interpretaion sur le département 69")
    model = joblib.load('./Modeles/RandomForestRegressor.joblib')
    local = df[df['DEP'].isin(['69'])]
    
    local.set_index('DEP', inplace = True)
    df.set_index('DEP', inplace = True)
    st.dataframe(local)
    prediction = model.predict(local)
    prediction = float(np.round(prediction, 2))
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(local)

    st.markdown('Expected Value:')
    st.info(explainer.expected_value)
    st.markdown("*C'est la valeur moyenne des valeurs SHAP pour toutes les instances du jeux de test.*")
    st.markdown("*Elle permet de comprendre l'importance relative de chaque fonctionnalité pour le modèle de prédiction*")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, local, plot_type="bar")
    st.pyplot(fig1)
    st.markdown("**Explication**")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, local)
    st.pyplot(fig1)
    st.markdown("**Explication**")
    