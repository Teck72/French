import numpy as np
import pandas as pd 
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objs as go





def local():
    df=pd.read_csv("./Data/Data_ML.csv")
    st.title("Etude de la prédiction sur le département 33")
    model = joblib.load('./Modeles/RandomForestRegressor.joblib')
    
  
    dep = '33'
    local = df[df['DEP'].isin([dep])]
    local.set_index('DEP', inplace = True)
    st.markdown("Données réelles du département :")
    st.dataframe(local)
    
    
    prediction = model.predict(local)
    prediction = float(np.round(prediction, 2))
    
    st.markdown("Prédiction de la moyenne des salaire net moyen par heure : ")
    st.success(prediction)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(local)
    
    trace = go.Bar(y=local.columns, x=shap_values[0], orientation='h')


    layout = go.Layout(title="Importance des fonctionnalités pour la prédiction du salaire net moyen par heure",
                   yaxis=dict(title="Fonctionnalités"),
                   xaxis=dict(title="SHAP value"),
                   height=400,
                   margin=dict(l=100, r=20, t=50, b=50))


    fig = go.Figure(data=[trace], layout=layout)


    st.plotly_chart(fig)
     
    trace = go.Scatter(y=local.columns, x=shap_values[0], mode='markers', 
                   marker=dict(color=shap_values[0], colorscale='RdBu', size=10),
                   text=local.values[0])


    layout = go.Layout(title="Importance des fonctionnalités pour la prédiction du salaire net moyen par heure",
                   yaxis=dict(title="Fonctionnalités"),
                   xaxis=dict(title="SHAP value"),
                   height=400,
                   margin=dict(l=100, r=20, t=50, b=50))


    fig = go.Figure(data=[trace], layout=layout)


    st.plotly_chart(fig)
    
    st.markdown("*Importance de chaque variable explicative par rapport à la variation de notre variable cible (que ce soit en positif ou en négatif)*")


    

    st.markdown('Expected Value :')
    st.success(explainer.expected_value)
    st.markdown("*La valeur Expected Value est la valeur moyenne attendue du modèle lorsque toutes les variables explicatives ont une valeur égale à leur moyenne.*")
    
    st.markdown('Matrice SHAP :')
    valeur = pd.DataFrame(shap_values,columns=local.columns).head()
    st.dataframe(valeur)
    
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("**Interprétation :**  ")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown(" Les graphes du SHAP indiquent que %InfoComm, %STServAdmi, %indus %Juniors et %Masters sont les variables les plus importantes pour expliquer la prédiction du modèle, tandis que %Moyenne, %const et %CTRH exercent une influence bien moindre.  ")
    st.markdown("La variable la plus importante est %InfoComm avec une valeur de SHAP de 0.3965, ce qui suggère que les valeurs élevées de cette variable ont un impact positif important sur la prédiction.")
    st.markdown("Ici, il y a une différence de 1.66€ entre l’expected value et la valeur prédite (14.89 – 13.23).   ")
    st.markdown("Cet écart s’explique par :   ")
    st.markdown("•	0.3965€ venant de la variable % InfoComm  ")
    st.markdown("•	0.2894€ venant de la variable %STServAdmi ")
    st.markdown("•	0.2152€ venant de la varible %Indus  ")
    st.markdown("•	0.1794€ venant de la variable %Master  ")
    st.markdown("•	0.1508€ venant de la variable %Junior  ")
    st.markdown("•	Etc….  ")
    st.markdown("  ")
    
  
    
    st.title("**Courbe d’évolution de la prédiction en fonction de la modification de nos facteurs.**")
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
    
    st.markdown("*Ici l’utilisateur à l’indication que sa prédiction peut varier de minimum 14.34€/h à maximum 15.13€/h en modifiant le % d’entreprise InfoComm entre 0 et 8%. Au-delà le 8%, il n’y aura pas de changement de la prédiction.*")
    
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
    
    st.markdown("*Ici l’utilisateur à l’indication que sa prédiction peut varier de minimum 14.38€/h à maximum 15.83€/h en modifiant le % d’entreprise InfoComm entre 14.38% et 26%. Avant ou au-delà de ces seuils, il n’y aura pas de changement de la prédiction.*")
    
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
    
    st.markdown("*Ici l’utilisateur à l’indication qu’il ne fera pas varier la prédiction de salaire en modulant la variable %Master entre 0 et 26%. *")
    st.markdown("*Au-delà, il créera une légère augmentation de la prédiction jusqu’à 28%, puis au-delà, l’augmentation de la population fera chuter la prédiction de salaire de 15.55€ à 14.44€/h. *") 
    
  
local()    