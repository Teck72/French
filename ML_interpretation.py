import numpy as np
import pandas as pd 
import seaborn as sns 
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular





def ML_inter():
     
    df=pd.read_csv("./Data/Data_ML.csv")
    salaires = pd.read_csv("./Data/salaires_dp.csv")
    st.title("Evaluation de la prédiction")
   
    df.set_index('DEP', inplace = True)
  
      
    column_list = list(salaires)[1:]
    selected_column = st.selectbox('Sélectionnez La categorie de Salaire :', column_list)
    
    
    
    
    if selected_column == 'SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor.joblib')
    if selected_column == 'cadre_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_cadre.joblib')
    if selected_column == 'cadre_moyen_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_cadre_moyen.joblib')
    if selected_column == 'employé_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_employe.joblib')
    if selected_column == 'travailleur_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_travailleur.joblib')
    if selected_column == '18_25ans_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_18_25ans.joblib')
    if selected_column == '26_50ans_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_26_50ans.joblib')
    if selected_column == '>50ans_SNHM' :
        model = joblib.load('./Modeles/RandomForestRegressor_50ans.joblib')
        
    predictions = model.predict(df)
    df_pred = pd.DataFrame(predictions, columns=['valeur_predite'])
       
    fig, ax = plt.subplots()


    ax.plot(salaires['DEP'].head(20),salaires[selected_column].head(20), label='Valeur réelle')
    ax.plot(df_pred['valeur_predite'].head(20), label='Valeur prédite')


    min_y, max_y = salaires[selected_column].min(), salaires[selected_column].max()
    plt.ylim(min_y, max_y)

    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')
    ax.legend()

    st.pyplot(fig)


    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[20:40],salaires[selected_column].iloc[20:40], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[20:40],df_pred['valeur_predite'].iloc[20:40], label='Valeur prédite')
    plt.ylim(min_y, max_y)
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[40:60],salaires[selected_column].iloc[40:60], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[40:60],df_pred['valeur_predite'].iloc[40:60], label='Valeur prédite')
    plt.ylim(min_y, max_y)
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[60:80],salaires[selected_column].iloc[60:80], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[60:80],df_pred['valeur_predite'].iloc[60:80], label='Valeur prédite')
    plt.ylim(min_y, max_y)
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[80:96],salaires[selected_column].iloc[80:96], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[80:96],df_pred['valeur_predite'].iloc[80:96], label='Valeur prédite')
    plt.ylim(min_y, max_y)
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')
    ax.legend()
    st.pyplot(fig)
  
    