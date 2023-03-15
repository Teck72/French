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
    model = joblib.load('./Modeles/RandomForestRegressor.joblib')
    df.set_index('DEP', inplace = True)
    predictions = model.predict(df)
    df_pred = pd.DataFrame(predictions, columns=['valeur_predite'])
    
    column_list = list(salaires)[1:]
    selected_column = st.selectbox('Sélectionnez une colonne', column_list)
    
       
    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].head(20),salaires[selected_column].head(20), label='Valeur réelle')
    ax.plot(df_pred['valeur_predite'].head(20), label='Valeur prédite')
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')


    ax.legend()


    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[20:40],salaires[selected_column].iloc[20:40], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[20:40],df_pred['valeur_predite'].iloc[20:40], label='Valeur prédite')
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')


    ax.legend()


    st.pyplot(fig)
    
   
    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[40:60],salaires[selected_column].iloc[40:60], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[40:60],df_pred['valeur_predite'].iloc[40:60], label='Valeur prédite')
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')


    ax.legend()


    st.pyplot(fig)
    
   
    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[60:80],salaires[selected_column].iloc[60:80], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[60:80],df_pred['valeur_predite'].iloc[60:80], label='Valeur prédite')
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')


    ax.legend()


    st.pyplot(fig)
    
 
   
    fig, ax = plt.subplots()
    ax.plot(salaires['DEP'].iloc[80:96],salaires[selected_column].iloc[80:96], label='Valeur réelle')
    ax.plot(salaires['DEP'].iloc[80:96],df_pred['valeur_predite'].iloc[80:96], label='Valeur prédite')
    ax.set_xlabel('Département')
    ax.set_ylabel('Salaire Horaire')
    ax.set_title('Résultats prédits')


    ax.legend()


    st.pyplot(fig)
    
  
    