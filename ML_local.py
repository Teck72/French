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




def local():
    df=pd.read_csv("./Data/Data_ML.csv")
    salaires = pd.read_csv("./Data/salaires_dp.csv")
    st.title("Etude de la prédiction sur le département 33")
    model = joblib.load('./Modeles/RandomForestRegressor.joblib')
    
  
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


    

    st.markdown('Expected Value :')
    st.success(explainer.expected_value)
    st.markdown("*La valeur Expected Value est la sortie moyenne attendue du modèle lorsque toutes les variables d'entrée ont une valeur moyenne.*")
    
    st.markdown('Matrice SHAP :')
    valeur = pd.DataFrame(shap_values,columns=local.columns).head()
    st.dataframe(valeur)
    
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("**Explication :**  ")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown(" les valeurs indiquent que %InfoComm, %STServAdmi, %Juniors et %Masters sont les variables les plus importantes pour expliquer la prédiction du modèle, tandis que %Moyenne, %const et %AutreServ ont une influence plus faible  ")
    st.markdown("La variable la plus importante est %STServAdmi avec une valeur de SHAP de 0.5315, ce qui suggère que les valeurs élevées de cette variable ont un impact positif important sur la prédiction.")
    st.markdown("  ")
    st.markdown("Pour expliquer les valeurs de SHAP données, prenons l'exemple de la variable %STServAdmi. La contribution moyenne de cette variable à la prédiction est de 0.5315. Cela signifie que, en moyenne, une augmentation de 1% de cette variable entraînera une augmentation de la prédiction de 0.5315, par rapport à la prédiction moyenne de toutes les observations d'entraînement. De même, pour la variable %Moyenne, la contribution moyenne est de -0,0171, ce qui signifie qu'une diminution de 1% de cette variable entraînera une diminution de la prédiction de 0,0171, par rapport à la prédiction moyenne.  ")
    
    st.markdown("**Impact du changement de certaines variables sur note prédiction :**")
    info_comm_values = np.arange(0, 40, 2)


    predictions_df = pd.DataFrame({'%InfoComm': info_comm_values})


    for info_comm in info_comm_values:
        local_copy = local.copy()
        local_copy['%InfoComm'] = info_comm
        prediction = model.predict(local_copy)
        prediction = float(np.round(prediction, 2))
        predictions_df.loc[predictions_df['%InfoComm'] == info_comm, 'Prédiction'] = prediction

    
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df['%InfoComm'], y=predictions_df['Prédiction'], mode='lines', name='Prédictions'))


    fig.update_layout(title="En fonction de pourcentage d'entreprise d'Information et de communication :",
                  xaxis_title='%InfoComm',
                  yaxis_title='Prédiction')


    st.plotly_chart(fig)
    
    info_comm_values = np.arange(0, 40, 2)
    predictions_df2 = pd.DataFrame({'%STServAdmi': info_comm_values})

    for info_comm in info_comm_values:
        local_copy = local.copy()
        local_copy['%STServAdmi'] = info_comm
        prediction = model.predict(local_copy)
        prediction = float(np.round(prediction, 2))
        predictions_df2.loc[predictions_df2['%STServAdmi'] == info_comm, 'Prédiction'] = prediction

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df2['%STServAdmi'], y=predictions_df2['Prédiction'], mode='lines', name='Prédictions'))

    fig.update_layout(title="En fonction de pourcentage d'entreprise d'Activités spécialisées, scient et techn, et activités de service et administratif :",
                  xaxis_title='%STServAdmi',
                  yaxis_title='Prédiction')

    st.plotly_chart(fig)
    
    info_comm_values = np.arange(0, 40, 2)
    predictions_df2 = pd.DataFrame({'%Masters': info_comm_values})

    for info_comm in info_comm_values:
        local_copy = local.copy()
        local_copy['%Masters'] = info_comm
        prediction = model.predict(local_copy)
        prediction = float(np.round(prediction, 2))
        predictions_df2.loc[predictions_df2['%Masters'] == info_comm, 'Prédiction'] = prediction

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df2['%Masters'], y=predictions_df2['Prédiction'], mode='lines', name='Prédictions'))

    fig.update_layout(title="En fonction de pourcentage de personnes de 45 à 64 ans :",
                  xaxis_title='%Masters',
                  yaxis_title='Prédiction')

    st.plotly_chart(fig)