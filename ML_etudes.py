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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score





sns.set_theme()


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
    st.markdown("*Nous constatons une forte corrélation des variables TOT et %SumMG avec d'autres.*    ")
    st.markdown("*Nous les supprimons pour notre étude*  ")
    
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
    st.markdown("*Nous constatons 3 Variables qui ont trés peu de corrélation avec nos variables cibles : ApESS,Immo etFinAss* ")
    st.markdown("*Nous allons les supprimer pour l'étude de nos modèles.*    ")
    drop_df = ['ApESS','Immo','FinAss']
    df3 = df2.drop(drop_df, axis = 1 )
    
    
    st.markdown("   ")
    st.markdown("   ")
    st.markdown("**Etudes de la valeur Cible du Salaire Moyen : SNHM**")
    with st.echo():
        target = salary.SNHM
        salary.SNHM = round(salary.SNHM,2)
        X_train, X_test, y_train,y_test = train_test_split(df3, target, test_size = 0.25, random_state=35)
    
       
    
    st.markdown("   ")
    modele = st.selectbox('Choix du modéle de régression :',('DecisionTreeRegressor','RandomForestRegressor'))
    if modele == 'DecisionTreeRegressor' :
        model = joblib.load('./Modeles/DecisionTreeRegressor.joblib')
        st.markdown("**Modéle DecisionTreeRegressor :**")
        st.markdown("*Ce modèle est un modèle de régression basé sur un arbre de décision.*")
        st.markdown("*L'arbre de décision est construit en divisant récursivement l'ensemble de données d'entraînement en sous-ensembles en fonction des valeurs des variables d'entrée.*")
        st.markdown("*Le processus de construction de l'arbre commence par la sélection d'une variable d'entrée pour diviser l'ensemble de données en deux sous-ensembles. L'objectif est de choisir une variable qui minimise la somme des erreurs quadratiques (ou tout autre critère de qualité de séparation) des deux sous-ensembles résultants.*")
        st.markdown("*Le processus de division est répété de manière récursive sur chaque sous-ensemble jusqu'à ce qu'un critère d'arrêt soit atteint, tel que le nombre minimum de données dans un sous-ensemble ou le nombre maximum de niveaux de l'arbre*")    
        st.markdown("   ")
        st.markdown("**Score du modéle :**")
       

        st.markdown('Score sur ensemble train : ')
        st.info(model.score(X_train, y_train))
        st.markdown('Score sur ensemble test : ')
        st.info(model.score(X_test, y_test))
        st.markdown("**Conclusion :**   ")
        st.markdown("Nous constatons un surajustement (overfitting) avec un score partiquement de 1 sur le modéle d'entrenaiment et inférieur à 0,5 sur celui de test.")
        st.markdown("Notre modèle d'entraînement est trop complexe et s'adapte trop étroitement aux données d'entraînement. Il est capable de mémoriser les exemples d'entraînement plutôt que de généraliser les modèles sous-jacents dans les données.   ")   
        st.markdown("Pour éviter cela, nous allons utilisé un modéle de forêt aléatoire (Random Forest), qui permet de combiner plusieurs modèles simples pour obtenir une performance de prédiction plus robuste.   ")
    
    else :
        model = joblib.load('./Modeles/RandomForestRegressor.joblib')  
        st.markdown("**Modéle RandomForestRegressor :**")
        st.markdown("*Il est basé sur un ensemble d'arbres de décision, où chaque arbre est entraîné sur un sous-ensemble aléatoire des données d'entraînement et des caractéristiques. Lors de la prédiction, chaque arbre de décision dans l'ensemble donne une prédiction, puis une moyenne (pour la régression) ou un vote majoritaire (pour la classification) est effectué pour produire la prédiction finale.*   ")
        st.markdown("*Il est capable de traiter des ensembles de données avec des caractéristiques et des classes très nombreuses ou complexes, sans surajustement (overfitting) comme cela a été constaté avec notre modéle DecisionTreeRegressor.*   ")
        st.markdown("   ")
        st.markdown("**Score du modéle :**")
       
        st.markdown('Score sur ensemble train : ')
        st.info(model.score(X_train, y_train))
        st.markdown('Score sur ensemble test : ')
        st.info(model.score(X_test, y_test))
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.markdown(" **Le score R² est :**  ")
        st.info(r2)
        st.markdown(" *Le score R² (R carré) est une mesure de la qualité de l'ajustement d'un modèle de régression aux données.*")
        st.markdown("*Plus le score R² est proche de 1, meilleure est la qualité de l'ajustement du modèle.*   ")
        st.markdown("*Il ne permet pas de déterminer si le modèle est pertinent ou non pour les données.*")
        st.markdown("*Nous allons par la suite évaluer les performances du modéle à l'aide de l'erreur moyenne absolue (MAE).*")
        
      
    
    
    