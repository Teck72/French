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
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.tree import export_graphviz




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

target = salary.SNHM
salary.SNHM = round(salary.SNHM,2)

drop_df = ['%SumMG','TOT']
df2 =df.drop(drop_df, axis = 1 )

drop_df = ['ApESS','Immo','FinAss']
df3 = df2.drop(drop_df, axis = 1 )

X_train, X_test, y_train,y_test = train_test_split(df3, target, test_size = 0.25, random_state=35)


def ML_evaluation():
    
    st.title("Evaluation du modèle Random Forest Regressor")
    
    model = joblib.load('./Modeles/RandomForestRegressor.joblib')
    st.markdown("**Affichage SHAP**")
    st.markdown("*Cette librairie permet d’expliquer les modèles complexes au niveau GLOBAL et LOCAL.*")
    st.markdown("*L’idée ici est d’appliquer la théorie des jeux (game therory) et de redistribuer le gain obtenu durant la partie, à tous les joueurs en fonction de leur implication dans la partie jouer.*")
    st.markdown("Si l’on retranscrit sur nos modèles, l’idée est de calculer ce shape en fonction de l’implication de chaque variable à faire varier notre variable cible.")
    st.markdown("  *- Chaque variable représente un jour de notre partie.*")
    st.markdown("  *- Le gain représente la différence entre la vrai valeur cible et la valeur prédite par le modèle.*")

    with st.echo():
         explainer = shap.TreeExplainer(model)
         shap_values = explainer.shap_values(X_test)

    st.markdown("Expected Value :   ")
    st.text(explainer.expected_value)
    st.markdown("*C'est la valeur moyenne des valeurs SHAP pour toutes les instances du jeux de test.*")
    st.markdown("*Elle permet de comprendre l'importance relative de chaque fonctionnalité pour le modèle de prédiction*")


    st.dataframe(shap_values)
    with st.echo():
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot(fig1)
        st.markdown("**Importance de chaque variable explicative par rapport à la variation de notre variable cible (que ce soit en positif ou en négatif)**")
    with st.echo():
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, X_test)
        st.pyplot(fig1)
    st.markdown("**Il y a deux infos principales :**")
    st.markdown("  **#Le SHAP = plus le chiffre est élevé positivement ou négativement, plus la variable cible explicative à de l’importance dans la valeure de notre variable cible.**")            
    st.markdown("  **#La COULEUR des observations, ici plus elle est rouge plus la valeur dans notre base de donnée est élevée**")
    st.markdown("   ")
    st.markdown("   ")
    st.markdown("**Arbre de Décision :**  ")

    with st.echo():
            fig, ax = plt.subplots()
            plot_tree(model.estimators_[0], feature_names=X_test.columns,
                      filled=True,rounded=True);
            st.pyplot(fig)
            
    with st.container():
        image = st.image("./Images/Tree_RandomForestRegressor.png", use_column_width=True)
        zoom_level = st.slider("Niveau de zoom", min_value=0, max_value=500, step=5, value=0)
        nouvelle_largeur = int(image.width * zoom_level)
        image.width = nouvelle_largeur        
  
           
            

    st.title("**Evaluation des performances du modéle choisi ( RandomForest ) sur le salaire Moyen par département  :**")
    metrics2 = metrics.rename(columns={'Unnamed: 0': 'Modèle'})
    st.dataframe(metrics2.tail(1))
    st.markdown("*MAE : Mesure l'erreur moyenne absolue entre les valeurs réelles et les valeurs prédites par le modèle*   ")
    st.markdown("*MSE :  Mesure la moyenne des carrés des erreurs entre les valeurs réelles et les valeurs prédites par le modèle.*   ")
    st.markdown("*Il est plus approprié pour certains types de problèmes, notamment lorsque les erreurs positives et négatives ont des effets égaux sur le résultat final.*")
    st.markdown("*RMSE : Il est la racine carrée du MSE.*   ")
    st.markdown("*Il mesure la distance moyenne entre les valeurs réelles et les valeurs prédites par le modèle, exprimée dans les mêmes unités que la variable de réponse (ou variable cible).*   ")
    st.markdown("   ")

    st.title("**Evaluation du modéle choisi ( RandomForest ) sur toutes nos variables cibles gràce au MAPE  :**")
    metrics_total2 = metrics_total.rename(columns={'Unnamed: 0': 'Modèle'})
    drop_metrics = ['MAE Moyen','MAE Cadres','MAE Cadres Moyens','MAE Employes','MAE Travailleurs','MAE 18_25 ans','MAE 26-50 ans','MAE + 50 ans']
    metrics_total3 =  metrics_total2.drop(drop_metrics, axis = 1 )
    metrics_total3 = metrics_total3.set_index('Modèle')
    metrics_total3 =  metrics_total3.applymap(lambda x: '{:.2%}'.format(x))
    st.markdown("**Par catégories d'emploi :**     ")
    st.dataframe(metrics_total3.loc[:,['MAPE Moyen', 'MAPE Cadres','MAPE Cadres Moyens','MAPE Employes','MAPE Travailleurs']].tail(1))
    st.markdown("**Par catégories d'age:**     ")
    st.dataframe(metrics_total3.loc[:,['MAPE 18_25 ans', 'MAPE 26-50 ans','MAPE + 50 ans']].tail(1))
    st.markdown("   ")
    st.markdown("*Le MAPE (Mean Absolute Percentage Error) est une mesure de l'erreur de prédiction d'un modèle qui exprime l'erreur absolue moyenne en pourcentage de la valeur réelle.*   ")
    st.markdown("*Il mesure la différence moyenne en pourcentage entre les valeurs réelles et les valeurs prédites. Pour chaque observation dans l'ensemble de données, l'erreur est calculée comme la différence entre la valeur réelle et la valeur prédite, puis elle est divisée par la valeur réelle pour obtenir une erreur en pourcentage. Les erreurs absolues en pourcentage sont ensuite moyennées sur toutes les observations pour obtenir le MAPE.*  ")



