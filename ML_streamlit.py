import pandas as pd 
import seaborn as sns 
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np 
import joblib
import shap
from pandas.api.types import is_numeric_dtype
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Custom function
st.cache
df=pd.read_csv("./Data/Data_ML.csv")
salaires = pd.read_csv("./Data/salaires_dp.csv")

def ML_stream():
    st.sidebar.markdown("**Salaire Moyen d'un département :**")
    
    
    st.title('Machine Learning ')
     
    st.markdown('**Sélectionner votre département :**')
    dep = st.selectbox('',
    ('01 : Ain','02 : Aisne','03 : Allier','04 : Alpes-de-Haute-Provence','05 : Hautes-Alpes','06 : Alpes-Maritimes','07 : Ardèche','08 : Ardennes ','09 : Ariège',
'10 : Aube','11 : Aude','12 : Aveyron','13 : Bouches-du-Rhône','14 : Calvados','15 : Cantal','16 : Charente','17 : Charente-Maritime','18 : Cher','19 : Corrèze',
'2A : Corse-du-Sud','2B : Haute-Corse',"21 : Côte-d'Or","22 : Côtes-d'Armor",'23 : Creuse','24 : Dordogne','25 : Doubs','26 : Drôme','27 : Eure','28 : Eure-et-Loir',
'29 : Finistère','30 : Gard','31 : Haute-Garonne','32 : Gers','33 : Gironde','34 : Hérault','35 : Ille-et-Vilaine','36 : Indre','37 : Indre-et-Loire','38 : Isère',
'39 : Jura','40 : Landes','41 : Loir-et-Cher','42 : Loire','43 : Haute-Loire','44 : Loire-Atlantique','45 : Loiret','46 : Lot','47 : Lot-et-Garonne','48 : Lozère',
'49 : Maine-et-Loire','50 : Manche','51 : Marne','52 : Haute-Marne','53 : Mayenne','54 : Meurthe-et-Moselle','55 : Meuse','56 : Morbihan','57 : Moselle','58 : Nièvre',
'59 : Nord','60 : Oise','61 : Orne','62 : Pas-de-Calais','63 : Puy-de-Dôme','64 : Pyrénées-Atlantiques','65 : Hautes-Pyrénées','66 : Pyrénées-Orientales','67 : Bas-Rhin',
'68 : Haut-Rhin','69 : Rhône','70 : Haute-Saône','71 : Saône-et-Loire','72 : Sarthe','73 : Savoie','74 : Haute-Savoie','75 : Paris','76 : Seine-Maritime','77 : Seine-et-Marne',
'78 : Yvelines','79 : Deux-Sèvres','80 : Somme','81 : Tarn','82 : Tarn-et-Garonne','83 : Var','84 : Vaucluse','85 : Vendée','86 : Vienne','87 : Haute-Vienne','88 : Vosges',
'89 : Yonne','90 : Territoire de Belfort','91 : Essonne','92 : Hauts-de-Seine','93 : Seine-St-Denis','94 : Val-de-Marne',"95 : Val-D'Oise"))
    dep = dep[0:2]
    local = df[df['DEP'].isin([dep])]
    local.set_index('DEP', inplace = True)
    st.dataframe(local)
    #modele = st.selectbox('Choix du modéle de régression :',('RandomForestRegressor','DecisionTreeRegressor'))
    modele='RandomForestRegressor'
            
    cible = st.selectbox('Choix de la valeur cible du salaire Moyen :',('Tous', 'Cadre','Cadre Moyen','Travailleur','Employe','18_25ans','26_50ans','>50ans'))
     
    if cible == 'Tous' :
        Median = salaires.SNHM.median()
        if modele == 'DecisionTreeRegressor' :
            regr = joblib.load('./Modeles/DecisionTreeRegressor.joblib')
        else :
            regr = joblib.load('./Modeles/RandomForestRegressor.joblib')
        
    if cible == 'Cadre' :  
         Median = salaires.cadre_SNHM.median()
         if modele == 'DecisionTreeRegressor' :
             regr = joblib.load('./Modeles/DecisionTreeRegressor_cadre.joblib')
         else :
             regr = joblib.load('./Modeles/RandomForestRegressor_cadre.joblib')
             
    if cible == 'Cadre Moyen' :  
          Median = salaires.cadre_moyen_SNHM.median()
          if modele == 'DecisionTreeRegressor' :
              regr = joblib.load('./Modeles/DecisionTreeRegressor_cadre_moyen.joblib')
          else :
              regr = joblib.load('./Modeles/RandomForestRegressor_cadre_moyen.joblib')
    if cible == 'Travailleur' :  
          Median = salaires.travailleur_SNHM.median()
          if modele == 'DecisionTreeRegressor' :
              regr = joblib.load('./Modeles/DecisionTreeRegressor_travailleur.joblib')
          else :
              regr = joblib.load('./Modeles/RandomForestRegressor_travailleur.joblib')          
    if cible == 'Employe' :  
        Median = salaires.employé_SNHM.median()
        if modele == 'DecisionTreeRegressor' :
            regr = joblib.load('./Modeles/DecisionTreeRegressor_employe.joblib')
        else :
            regr = joblib.load('./Modeles/RandomForestRegressor_employe.joblib')   
    if cible == '18_25ans' :  
           Median = salaires.employé_SNHM.median()
           if modele == 'DecisionTreeRegressor' :
               regr = joblib.load('./Modeles/DecisionTreeRegressor_18_25ans.joblib')
           else :
               regr = joblib.load('./Modeles/RandomForestRegressor_18_25ans.joblib')          
    if cible == '26_50ans' :  
             Median = salaires.employé_SNHM.median()
             if modele == 'DecisionTreeRegressor' :
                 regr = joblib.load('./Modeles/DecisionTreeRegressor_26_50ans.joblib')
             else :
                 regr = joblib.load('./Modeles/RandomForestRegressor_26_50ans.joblib')     
    if cible == '>50ans' :  
             Median = salaires.employé_SNHM.median()
             if modele == 'DecisionTreeRegressor' :
                 regr = joblib.load('./Modeles/DecisionTreeRegressor_50ans.joblib')
             else :
                 regr = joblib.load('./Modeles/RandomForestRegressor_50ans.joblib')               
                    
    
    
    col = st.selectbox("Selection du facteur à moduler :", local.columns)
    old_value = local[col].median()
    with st.form(key='my_form'):
     col1,col2 = st.columns(2)
     with col1:
          st.markdown ("Valeur d'Origine en %")
          st.markdown(old_value)
     with col2:
          new_val = st.slider("Nouvelle Valeur")
     if st.form_submit_button("Remplace"):
          local[col]=local[col].replace(old_value,new_val)
          st.dataframe(local) 
    
        
    prediction = regr.predict(local)
    prediction = float(np.round(prediction, 2))
    st.markdown('**Affichage du SHAP Local :**')
    
    
    st.sidebar.markdown('Numpéro du département : ')
    st.sidebar.info(dep)
    st.sidebar.markdown("catégories d'emploi ou d'âge : ")
    st.sidebar.info(cible)
    st.sidebar.metric(label="**En € par Heure :** ", value=prediction, delta=round((prediction-Median),2))
    st.sidebar.markdown('*(Indique la différence avec le salaire médian Français)*')
  
   
    explainer = shap.TreeExplainer(regr)
    shap_values = explainer.shap_values(local)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    f = shap.force_plot(explainer.expected_value, shap_values, local,matplotlib=True, show=False,)
    
    st.pyplot(f,bbox_inches='tight')
    
      
     
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, local)
    st.pyplot(fig1)
    st.markdown("*Importance de chaque variable explicative par rapport à la variation de notre variable cible (que ce soit en positif ou en négatif)*")

   


ML_stream()
