import numpy as np
import pandas as pd 
import seaborn as sns 
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import plotly.graph_objs as go





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
    
    ticktext = list(salaires['DEP'].iloc[:45])
    tickvals = salaires.index[:45] 

   
    
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=salaires.index[:45], y=salaires[selected_column].iloc[:45], mode='lines+markers', name='Valeur réelle', line=dict(width=2,color = 'Blue')))
    fig1.add_trace(go.Scatter(x=salaires.index[:45], y=df_pred['valeur_predite'].iloc[:45], mode='lines+markers', name='Valeur prédite', line=dict(width=2, color = 'Green')))
    fig1.update_layout(title='Première partie des départements',xaxis=dict(
        title='Départements',
        ticktext=ticktext,
        tickvals=tickvals,
       )
 )
    
    
    ticktext = list(salaires['DEP'].iloc[45:])
    tickvals = salaires.index[45:] 

    

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=salaires.index[45:], y=salaires[selected_column].iloc[45:], mode='lines+markers', name='Valeur réelle', line=dict(width=2,color = 'Blue')))
    fig2.add_trace(go.Scatter(x=salaires.index[45:], y=df_pred['valeur_predite'].iloc[45:], mode='lines+markers', name='Valeur prédite', line=dict(width=2, color = 'Green')))
    fig2.update_layout(title='Deuxième partie des départements',xaxis=dict(
        title='Départements',
        ticktext=ticktext,
        tickvals=tickvals,
       )
 )


    max_salary = max(salaires[selected_column])
    y_axis_range = [0, max_salary * 1.1]


    st.plotly_chart(fig1.update_yaxes(range=y_axis_range))
    st.plotly_chart(fig2.update_yaxes(range=y_axis_range))


