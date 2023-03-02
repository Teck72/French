import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import joblib
import shap
from sklearn.tree import plot_tree
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


sns.set_theme()


path='/content/drive/MyDrive/shared-with-me'

base_etablissement=pd.read_csv("./Data/base_etablissement_dp.csv")
salary=pd.read_csv("./Data/dp_salaires.csv")
popu=pd.read_csv("./Data/Popu_DEP.csv")
te_100 = pd.read_csv("./Data/te_100.csv")
dep_loyer = pd.read_csv('./Data/dep_loyer_app.csv')
metrics = pd.read_csv('./Data/Metrics.csv')
metrics_total = pd.read_csv('./Data/Metrics_total.csv')

base_etablissement = base_etablissement.drop(['Unnamed: 0'], axis=1)
popu = popu.drop(['Unnamed: 0'], axis=1)
te_100 = te_100.drop(['Unnamed: 0'], axis=1)
dep_loyer  = dep_loyer.drop(['Unnamed: 0'], axis=1)
salary  = salary.drop(['Unnamed: 0'], axis=1)

df = base_etablissement.merge(popu, how="left", left_on = "DEP", right_on="DEP")
df = df.merge(te_100, how="left", left_on = "DEP", right_on="DEP")
df = df.merge(dep_loyer, how="left", left_on = "DEP", right_on="DEP")


df.set_index('DEP', inplace = True)

df.rename(columns = {'indus':'%indus', 'const':'%const',
                              'CTRH':'%CTRH','InfoComm' :'%InfoComm','STServAdmi':'%STServAdmi','AutreServ':'%AutreServ'}, inplace = True)

def ML_etude():
    st.title("Etude MACHINE LEARNING")
    st.markdown(" Nous allons etudié deux modéles de régression pour la prédiction du salaire Moyen d'un départment : ")
    st.markdown("       -DecisionTreeRegressor ")
    st.markdown("       -RandomForestRegressor ")
    st.markdown("   ")
    st.markdown("   ")
    st.markdown(" Etude des vaiables de notre Dataset nettoyé :   ")
    st.dataframe(df)
    st.markdown("   ")
    st.markdown("**Analyse des corrélations :**")
   
    image = Image.open('./Images/Correlation.png')
    st.image(image)
    st.markdown("   ")
    st.markdown("Nous constatons une forte corrélation des variables TOT et %SumMG avec d'autres.    ")
    st.markdown("Nous les supprimons pour notre étude  ")
    
    drop_df = ['%SumMG','TOT']
    df2 =df.drop(drop_df, axis = 1 )
    
    st.markdown("   ")
    st.markdown("Nous regroupons notre base avec celle de la cible ( Salary ) afin d'identifier les valeurs les plus corrélées avec celle_ci.")
    st.markdown("   ")
    
    df_cor = df2.merge(salary, how="left", left_on = "DEP", right_on="DEP")
    df_cor.set_index('DEP', inplace = True)
    
    st.markdown("**Analyse des corrélations avec le variables cibles :**")
    
  
    image = Image.open('./Images/Correlation_cible.png')
    st.image(image)
    st.markdown("   ")
    st.markdown("**COMMENTAIRES**")
    drop_df = ['ApESS','Immo','FinAss']
    df3 = df2.drop(drop_df, axis = 1 )
    
    st.markdown("   ")
    st.markdown("   ")
    st.markdown("**Etudes de la valeur Cible du Salaire Moyen : SNHM**")
    target = salary.SNHM
    salary.SNHM = round(salary.SNHM,2)
    X_train, X_test, y_train, y_test = train_test_split(df3, target, test_size = 0.25, random_state=35)
    
       
    
    st.markdown("   ")
    modele = st.selectbox('Choix du modéle de régression :',('RandomForestRegressor','DecisionTreeRegressor'))
    if modele == 'DecisionTreeRegressor' :
        model = joblib.load('./Modeles/DecisionTreeRegressor.joblib')
            
    else :
        model = joblib.load('./Modeles/RandomForestRegressor.joblib')  
      
    st.markdown("   ")
    st.markdown("Affichage SHAP")
    

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    st.markdown("Expected Value :   ")
    st.text(explainer.expected_value)
    st.dataframe(shap_values)
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    st.pyplot(fig1)
    st.markdown("**Importance de chaque variable explicative par rapport à la variation de notre variable cible (que ce soit en positif ou en négatif)**")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_test)
    st.pyplot(fig1)
    st.markdown("**Il y a deux infos principales :**")
    st.markdown("  **#Le SHAP = plus le chiffre est élevé positivement ou négativement, plus la variable cible à de l’importance dans la valeure de notre variable cible.**")            
    st.markdown("  **#La COULEUR des observations, ici plus elle est rouge plus la valeur dans notre base de donnée est élevé.**")
    if modele == 'DecisionTreeRegressor' :
        image = Image.open('./Images/Tree_DecisionTreeRegressor.png')
        st.image(image,output_format='PNG')
            
    else :
        image = Image.open('./Images/Tree_RandomForestRegressor.png')
        st.image(image,output_format='PNG')
    
    
   



    st.title("Metrics pour le salaire Moyen  :")
    st.dataframe(metrics)
    st.markdown("   ")
    st.markdown("   ")
    st.title("Metrics pour toutes les valeurs cibles  :")
    st.dataframe(metrics_total)
    
    
    
    