import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import joblib
from sklearn.tree import plot_tree
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
    st.markdown(" Nous allons étudier deux modèles de régression pour la prédiction du salaire moyen d'un département : ")
    st.markdown("       -DecisionTreeRegressor ")
    st.markdown("       -RandomForestRegressor ")
    st.markdown("   ")
    st.markdown("   ")
    st.markdown(" Voici notre Dataset de variables explicatives nettoyé :   ")
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
    st.markdown("**Etude de la valeur cible du salaire moyen : SNHM**")
    with st.echo():
        target = salary.SNHM
        salary.SNHM = round(salary.SNHM,2)
        X_train, X_test, y_train,y_test = train_test_split(df3, target, test_size = 0.25, random_state=35)
    
       
    
    st.markdown("   ")
    modele = st.selectbox('Choix du modèle de régression :',('DecisionTreeRegressor','RandomForestRegressor'))
    if modele == 'DecisionTreeRegressor' :
        model = joblib.load('./Modeles/DecisionTreeRegressor.joblib')
        st.markdown("**Modèle DecisionTreeRegressor :**")
        st.markdown("*Ce modèle est un modèle de régression basé sur un arbre de décision.*")
        st.markdown("*L'arbre de décision est construit en divisant récursivement l'ensemble de données d'entraînement en sous-ensembles en fonction des valeurs des variables d'entrée.*")
        st.markdown("*Le processus de construction de l'arbre commence par la sélection d'une variable d'entrée pour diviser l'ensemble de données en deux sous-ensembles. L'objectif est de choisir une variable qui minimise la somme des erreurs quadratiques (ou tout autre critère de qualité de séparation) des deux sous-ensembles résultants.*")
        st.markdown("*Le processus de division est répété de manière récursive sur chaque sous-ensemble jusqu'à ce qu'un critère d'arrêt soit atteint, tel que le nombre minimum de données dans un sous-ensemble ou le nombre maximum de niveaux de l'arbre*")    
        st.markdown("   ")
        st.markdown("**Score du modèle :**")
       

        st.markdown('Score sur ensemble train : ')
        st.info(model.score(X_train, y_train))
        st.markdown('Score sur ensemble test : ')
        st.info(model.score(X_test, y_test))
        st.markdown("**Conclusion :**   ")
        st.markdown("Nous constatons un surajustement (overfitting) avec un score pratiquement de 1 sur le modèle d'entraînement et un score de 0,55 sur celui de test.")
        st.markdown("Notre modèle d'entraînement est trop complexe et s'adapte trop étroitement aux données d'entraînement. Il est capable de mémoriser les exemples d'entraînement plutôt que de généraliser les modèles sous-jacents aux nouvelles données entrantes.   ")   
        st.markdown("Pour éviter cela, nous allons utiliser un modèle de forêt aléatoire (Random Forest), qui permet de combiner plusieurs modèles simples pour obtenir une performance de prédiction plus robuste.   ")
    
    else :
        model = joblib.load('./Modeles/RandomForestRegressor.joblib')  
        st.markdown("**Modèle RandomForestRegressor :**")
        st.markdown("*Il est basé sur un ensemble d'arbres de décision, où chaque arbre est entraîné sur un sous-ensemble aléatoire de données d'entraînement et des caractéristiques différentes. Lors de la prédiction, chaque arbre de décision donne une prédiction, puis il fait une moyenne (pour la régression) pour produire la prédiction finale.*   ")
        st.markdown("*Il est capable de traiter des ensembles de données avec des caractéristiques et des classes très nombreuses ou complexes, sans surajustement (overfitting) comme cela a été constaté avec notre modéle DecisionTreeRegressor.*   ")
        st.markdown("   ")
        st.markdown("**Score du modèle :**")
       
        st.markdown("Score sur l'ensemble train : ")
        st.info(model.score(X_train, y_train))
        st.markdown("Score sur l'ensemble test : ")
        st.info(model.score(X_test, y_test))
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.markdown(" **Le score R² est :**  ")
        st.info(r2)
        st.markdown(" *Le score R² (R carré) est une mesure de la qualité de l'ajustement d'un modèle de régression aux données.*")
        st.markdown("*Plus le score R² est proche de 1, meilleure est la qualité de l'ajustement du modèle.*   ")
        st.markdown("*Il ne permet pas de déterminer si le modèle est pertinent ou non pour les données, c’est pour cela que, par la suite, nous évaluerons les performances du modèle à l'aide de l'erreur moyenne absolue (MAE).*")
        st.markdown("  ")
        st.markdown("Nos premiers scores de train et de test ainsi que notre R² de 0.83 sont satisfaisants et présagent de bons résultats pour la prédiction de salaire.")
        fig, ax = plt.subplots()
        with st.echo():
            plot_tree(model.estimators_[0], feature_names=X_test.columns,
                     filled=True,rounded=True);
        st.markdown(" ")
        st.markdown("Illustration du premier arbre aléatoire créé par le modèle : ")
        st.pyplot(fig)
        st.markdown("*Ici notre variable cible est le salaire moyen en France.*")
        st.markdown("*Notre arbre est composé d’un ensemble de 44 occurrences, choisies par la machine, avec un salaire moyen de 13,92€ ainsi qu’une MSE de 2.24.*")
        st.markdown("*La MSE est l’erreur quadratique qui est la différence au carrée entre la valeur réelle et la valeur prédite.*")
        st.markdown("*L’objectif final étant de faire des groupes homogènes avec une MSE la plus proche possible de 0.*")
        st.markdown("*Ici l’arbre créé deux groupes avec comme frontière une composition, pour les départements, de 11% d’entreprise de la construction*")
        st.markdown("*Il en ressort donc 2 groupes les plus homogènes possible dont :*")
        st.markdown("*-	Le premier groupe composé de 2 départements ayant plus de 11% d’entreprises de la construction, ayant un salaire moyen de 19.4€/h avec une MSE de 6.2 (ici MSE haute donc moins fiable).*")
        st.markdown("*-	Le second groupe composé de 42 départements ayant moins de 11% d’entreprise de la construction, ayant un salaire moyen de 12.74€/h avec une MSE de 0.90 (très satisfaisante)*")
        st.markdown("*A partir de ces deux nouveaux échantillons, 4 groupes les plus homogènes possibles sont créés en dessous :*")
        st.markdown("*-	Le premier composé d’un département ayant plus de 4% d’entreprises industrielles avec une MSE = 0 ; c’est-à-dire qu’il est représentatif d’une ligne de données réelles de notre dataset, avec une valeur de salaire moyen = 21.90€/h. C’est la fin de cette branche, il représentera une prédiction possible.*")
        st.markdown("*-	Le second composé d’un département ayant moins de 4% d’entreprises industrielles avec une MSE = 0, c’est-à-dire qu’il est représentatif d’une ligne de données réelles de notre dataset, avec une valeur de salaire moyen = 16.91€/h. C’est la fin de cette branche, il représentera une prédiction possible. *")    
        st.markdown("*-	Le troisième composé de 4 départements ayant plus de 30% de masters dans sa population, avec une MSE très satisfaisante de 0.69 et un salaire moyen de 15.35€/h. *")
        st.markdown("*-	Le quatrième groupe est composé de 38 départements ayant moins de 30% de masters dans sa population, avec une MSE très satisfaisante de 0.35 et une valeur de salaire moyen de 12.54€/h. *")
        st.markdown("*Ces troisièmes et quatrièmes groupes sont eux même partagés en quatre nouveaux sous-groupes les plus homogènes possible…. *")
        st.markdown("*Et etcétéra, jusqu’à l’obtention de groupe ayant une MSE la plus proche possible de 0 que notre machine pourra utiliser pour faire les prédictions. *")
      
        
ML_etude()    