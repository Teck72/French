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
    st.markdown("** Explication :**  ")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown(" les valeurs indiquent que %InfoComm, %STServAdmi, %Juniors et %Masters sont les variables les plus importantes pour expliquer la prédiction du modèle, tandis que %Moyenne, %const et %AutreServ ont une influence plus faible  ")
    st.markdown("La variable la plus importante est %InfoComm avec une valeur de SHAP de 0.3832, ce qui suggère que les valeurs élevées de cette variable ont un impact positif important sur la prédiction.")
    st.markdown("  ")
    st.markdown("  ")
    expected_info_comm = 27
    info_comm_values = np.arange(0, 51, 5)
    prediction_changes = local['InfoComm'] * (info_comm_values - expected_info_comm)


    new_predictions =  prediction + prediction_changes


    for i in range(len(info_comm_values)):
        print("Lorsque %InfoComm est égal à {:.0f}, la prédiction est {:.2f}".format(info_comm_values[i], new_predictions[i]))
    
 
    
    