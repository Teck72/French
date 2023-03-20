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
    valeur = pd.DataFrame(shap_values).head()
    st.dataframe(valeur)
    
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("**Explication :**  ")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown(" les valeurs indiquent que %InfoComm, %STServAdmi, %Juniors et %Masters sont les variables les plus importantes pour expliquer la prédiction du modèle, tandis que %Moyenne, %const et %AutreServ ont une influence plus faible  ")
    st.markdown("La variable la plus importante est %STServAdmi avec une valeur de SHAP de 0.5315, ce qui suggère que les valeurs élevées de cette variable ont un impact positif important sur la prédiction.")
    st.markdown("  ")
    st.markdown("  ")
    
    st.markdown("**Impact du changement de certaines variables sur note prédiction :**")
    info_comm_values = np.arange(0, 51, 5)


    predictions_df = pd.DataFrame({'%InfoComm': info_comm_values})


    for info_comm in info_comm_values:
        local_copy = local.copy()
        local_copy['%InfoComm'] = info_comm
        prediction = model.predict(local_copy)
        prediction = float(np.round(prediction, 2))
        predictions_df.loc[predictions_df['%InfoComm'] == info_comm, 'Prédiction'] = prediction

    st.markdown("En fonction de pourcentage d'entreprise d'Information et de communication : ")  
    st.dataframe(predictions_df)
    
    info_comm_values = np.arange(0, 40, 3)
    predictions_df2 = pd.DataFrame({'%STServAdmi': info_comm_values})


    for info_comm in info_comm_values:
        local_copy = local.copy()
        local_copy['%STServAdmi'] = info_comm
        prediction = model.predict(local_copy)
        prediction = float(np.round(prediction, 2))
        predictions_df2.loc[predictions_df2['%STServAdmi'] == info_comm, 'Prédiction'] = prediction

    st.markdown("En fonction de pourcentage d'entreprise d'Activités spécialisées, scient et techn, et activités de service et administratif : ") 
    st.dataframe(predictions_df2)
 
    
    