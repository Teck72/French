import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import time
import altair as alt
import seaborn as sns
from PIL import Image

sns.set_theme()

Popu=pd.read_csv("./Data/Popu_DEP.csv")
dp_salaires=pd.read_csv("./Data/dp_salaires.csv")
base_etablissement_dp=pd.read_csv("./Data/base_etablissement_dp.csv")



Popu = Popu.drop(['Unnamed: 0'], axis=1)
dp_salaires = dp_salaires.drop(['Unnamed: 0'], axis=1)
base_etablissement_dp = base_etablissement_dp.drop(['Unnamed: 0'], axis=1)





def bases_streamlit():
    st.title("Visualisation des Bases de données")
    length = 30000
    bins=500

    st.sidebar.markdown("# Choix de la base")

    choix = st.sidebar.radio("Choix de la base", ("Projet Globale","Populations", "Salaire Moyen","Etablissement"))
 
    st.subheader(choix)
    
    
    if choix == 'Projet Globale':
        st.title("Les Inégalitées Salariale en France selon les terrritoires")
        image = Image.open('./Images/SNHM.png')

        st.image(image)
        st.markdown("Nous pouvons constater une forte inégalité des salaires moyens selon les départements")
        st.markdown("Nous allons étudier l'impacte de plusieurs variables sur celui-ci afin de fournir")
        st.markdown("un outil de machine Learning capable de prédire le salaire Moyen par déparement")
      

    if choix == 'Populations' :
        st.dataframe(Popu)
     
          
      
        col1, col2 = st.columns(2)
      
        original = Image.open('./Images/Populations_Actif.png')
        col1.header("Actif")
        col1.image(original, use_column_width=True)
      
        grayscale = Image.open('./Images/Populations_Non_Actif.png')
        col2.header("Non Actif")
        col2.image(grayscale, use_column_width=True)

          
 
    
    if choix == 'Salaire Moyen' :
        st.dataframe(dp_salaires)
    
    #Affichage des 10 départements ayant les salaires net moyen les plus élevés et bas


        max_col = dp_salaires.head(10)
        min_col = dp_salaires.tail(10)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(25,5), sharey=True)

        sns.barplot(x=max_col['DEP'], y=max_col['SNHM'],ax=ax1);
        ax1.title.set_text("10 départements ayant les salaires net moyen les plus élevés")
        sns.barplot(x=min_col['DEP'], y=min_col['SNHM'], ax=ax2);
        ax2.title.set_text("10 départements ayant les salaires net moyen les plus bas")
    
        st.write(fig)
        
        st.markdown ( "Sur le graphe ci-dessus, on constate que les départements 75,92 et 78 sont ceux ayant les salaires net moyen les plus élevés.")
        st.markdown ( "Et les 10 départements qui ont les salaires net moyens les plus bas ont presque les mêmes salaires net moyens.")
        st.markdown ( " Un peu moins que 13.33 euros/heure (moyen salaire net moyen en France)")
    
        fig, ax = plt.subplots(1, figsize=(15,10))
    
        dp_salaires_age = dp_salaires[['18_25ans_SNHM','26_50ans_SNHM','>50ans_SNHM']]
        sns.boxplot(data=dp_salaires_age);
    
        st.write(fig)
    
        st.markdown ( "1er constat: Le salaire net moyen par heure pour un cadre est supérieur à ceux des autres catégories.") 
        st.markdown ( "2e constat: Pour les employés et les travailleurs, la tranche salariale est approximativement la même bien que les travailleurs ont un salaire plus élevé.")
        st.markdown ( "Conclusion: On ressent une forte inégalité des salaires des cadres par rapport à ceux des autres catégories. Les cadres et les cadres moyens sont privilégiés niveau salariale.")
        
        image = Image.open('./Images/SNHM.png')
        st.image(image)
    
    if choix == 'Etablissement' :
    
        st.dataframe(base_etablissement_dp)
        
bases_streamlit()
        