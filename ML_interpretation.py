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
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer






def ML_inter():
     
    df=pd.read_csv("./Data/Data_ML.csv")
    salaires = pd.read_csv("./Data/salaires_dp.csv")
    st.title("Evaluation de la prédiction")
   
   
  
      
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
        
        
    df_total = df.merge(salaires, how="left", left_on = "DEP", right_on="DEP")  
    
    
    target = df_total[['DEP','SNHM','cadre_SNHM','cadre_moyen_SNHM','employé_SNHM','travailleur_SNHM','18_25ans_SNHM','26_50ans_SNHM','>50ans_SNHM']]
    
    drop_df = ['SNHM','cadre_SNHM','cadre_moyen_SNHM','employé_SNHM','travailleur_SNHM','18_25ans_SNHM','26_50ans_SNHM','>50ans_SNHM']
    variables_pre = df_total.drop(drop_df, axis = 1)
    
    variables_pre.set_index('DEP', inplace = True)
    

           
    predictions = model.predict(variables_pre)
    
      
    
 
    df_pred = pd.DataFrame(predictions, columns=['valeur_predite'])
    

    
    ticktext = list(target['DEP'].iloc[:45])
    tickvals = target.index[:45] 

    
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=target.index[:45], y=target[selected_column].iloc[:45], mode='lines+markers', name='Valeur réelle', line=dict(width=2,color = 'Blue')))
    fig1.add_trace(go.Scatter(x=target.index[:45], y=df_pred['valeur_predite'].iloc[:45], mode='lines+markers', name='Valeur prédite', line=dict(width=2, color = 'Green')))
    fig1.update_layout(title='Première partie des départements',xaxis=dict(
        title='Départements',
        ticktext=ticktext,
        tickvals=tickvals,
       )
 )
    
    
    ticktext = list(target['DEP'].iloc[45:])
    tickvals = salaires.index[45:] 

    

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=target.index[45:], y=target[selected_column].iloc[45:], mode='lines+markers', name='Valeur réelle', line=dict(width=2,color = 'Blue')))
    fig2.add_trace(go.Scatter(x=target.index[45:], y=df_pred['valeur_predite'].iloc[45:], mode='lines+markers', name='Valeur prédite', line=dict(width=2, color = 'Green')))
    fig2.update_layout(title='Deuxième partie des départements',xaxis=dict(
        title='Départements',
        ticktext=ticktext,
        tickvals=tickvals,
       )
 )


 


    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    
    st.header( 'Etude du fonctionnement de notre modéle sur un département : ')
  
  
    dep = '33'
    local = df[df['DEP'].isin([dep])]
    local.set_index('DEP', inplace = True)
    st.dataframe(local)
    
    prediction = model.predict(local)
    prediction = float(np.round(prediction, 2))
    
    st.success(prediction)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(local)

    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.summary_plot(shap_values, local, plot_type="bar"))
    
      
     
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, local)
    st.pyplot(fig1)
    st.markdown("*Importance de chaque variable explicative par rapport à la variation de notre variable cible (que ce soit en positif ou en négatif)*")


    

    st.markdown('Expected Value:')
    st.markdown(explainer.expected_value)

    valeur = pd.DataFrame(shap_values).head()
    
    st.dataframe(valeur)
    st.dataframe(local)
    
    
    
    
  