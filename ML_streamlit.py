import pandas as pd 
import seaborn as sns 
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np 
import joblib
import shap
from pandas.api.types import is_numeric_dtype


# Custom function
# st.cache is used to load the function into memory
df=pd.read_csv("./Data/Data_ML.csv")
salaires = pd.read_csv("./Data/salaires_dp.csv")

def ML_stream():
    
    st.title('Machine Learning sur les salaires moyens en France')
    st.markdown('Nous allons utiliser des modéles de Régréssions pour prédir le salaire moyen d un département')
    st.dataframe(df)  
    st.markdown('**Sélectionner votre départements :**')
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
    st.subheader("Prediction du Salaire Moyen : ")
    modele = st.selectbox('Choix du modéle de régression :',('RandomForestRegressor','DecisionTreeRegressor'))
            
    cible = st.selectbox('Choix de la valeur cible du salaire Moyen :',('Tous', 'Cadre','Cadre Moyen','Travailleur','Employe'))
     
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
             
 
                    
    
    
    col = st.selectbox("Selection d'une collone pour modification :", local.columns)
    old_value = local[col].median()
    with st.form(key='my_form'):
     col1,col2 = st.columns(2)
     st_input = st.number_input if is_numeric_dtype(local[col]) else st.text_input
     with col1:
          st.markdown ("Ancienne Valeur")
          st.markdown(old_value)
     with col2:
          new_val = st_input("Nouvelle Valeur")
     if st.form_submit_button("Remplace"):
          local[col]=local[col].replace(old_value,new_val)
          st.dataframe(local) 
    
        
    prediction = regr.predict(local)
    prediction = float(np.round(prediction, 2))
    st.markdown('**Prédiction du salaire moyen :**')
    st.markdown('Rouge : Inférieur au Salaire Médian Français')
    if prediction < Median :
        st.error(prediction)
    else :
        st.success(prediction)    
   
    explainer = shap.TreeExplainer(regr)
    shap_values = explainer.shap_values(local)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.summary_plot(shap_values, local, plot_type="bar"))
    st.pyplot(shap.summary_plot(shap_values, local)) 


ML_stream()
