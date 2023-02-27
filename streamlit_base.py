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
Popu_DEP = pd.read_csv("./Data/Popu_DEP2.csv")
Popu_Actifs = pd.read_csv("./Data/Popu_Actifs.csv")
dep_loyer_app = pd.read_csv("./Data/dep_loyer_app.csv")
te = pd.read_csv("./Data/te.csv")

Popu = Popu.drop(['Unnamed: 0'], axis=1)
dp_salaires = dp_salaires.drop(['Unnamed: 0'], axis=1)
base_etablissement_dp = base_etablissement_dp.drop(['Unnamed: 0'], axis=1)
dep_loyer_app = dep_loyer_app.drop(['Unnamed: 0'], axis=1)
te = te.drop(['Unnamed: 0'], axis=1)





def bases_streamlit():
    st.title("Visualisation des Bases de données")
    length = 30000
    bins=500

    st.sidebar.markdown("# Choix de la base")

    choix = st.sidebar.radio("Choix de la base", ("Projet Globale","Populations", "Salaire Moyen","Etablissement","Loyer Appartement","Type d'entreprise"))
 
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
        
        max_col = Popu_Actifs.head(10)
        min_col = Popu_Actifs.tail(10)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(25,5), sharey=True)

        sns.barplot(x=max_col['DEP'], y=max_col['Actifs'],ax=ax1);
        ax1.title.set_text("10 départements avec le plus d'actifs en Millions")
        sns.barplot(x=min_col['DEP'], y=min_col['Actifs'], ax=ax2);
        ax2.title.set_text("10 départements avec le moins d'Actifs")
    
        st.write(fig)
        
        
        
        plt.figure(figsize = (8, 8))
        x = [Popu_DEP['Ainés'].sum(),Popu_DEP['Enfants'].sum(),Popu_DEP['Juniors'].sum(),Popu_DEP['Masters'].sum(),Popu_DEP['Séniors'].sum()]
        fig1, ax1 = plt.subplots()
        plt.pie(x, labels = ['Ainés','Enfants','Juniors','Masters','Séniors'],
                autopct = lambda x: str(round(x, 2)) + '%',
                pctdistance = 0.7, labeldistance = 1.05,
                shadow = False)
        plt.title('Répartition des catégories de population en France Métropolitaine')
        st.pyplot(fig1)
        
        Popu_DEP['Non_Actifs']=Popu_DEP.Enfants + Popu_DEP.Ainés
        Popu_DEP['Actifs']=Popu_DEP.Juniors + Popu_DEP.Masters + Popu_DEP.Séniors
        Popu_DEP['Total'] = Popu_DEP.Juniors + Popu_DEP.Masters + Popu_DEP.Séniors+Popu_DEP.Enfants + Popu_DEP.Ainés

        Popu_DEP2 = Popu_DEP.sort_values('Total', ascending=False)

        
        fig1, ax1 = plt.subplots()
        max_col = Popu_DEP2.head(10)
        x =  max_col['DEP']
        y1 = max_col['Actifs']
        y2 = max_col['Non_Actifs']


        plt.bar(x, y1, color = "#3ED8C9", label = 'Actifs')

        plt.bar(x, y2,bottom = y1, color = "#EDFF91", label = 'Non Actifs')
        plt.legend()

        plt.title("10 départements les plus peuplés")
        st.pyplot(fig1)

          
 
    
    if choix == 'Salaire Moyen' :
        st.dataframe(dp_salaires)
    
    #Affichage des 10 départements ayant les salaires net moyen les plus élevés et bas

        image = Image.open('./Images/SNHM.png')
        st.image(image)
    
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
     
    
    if choix == 'Etablissement' :
    
        st.dataframe(base_etablissement_dp)
        image = Image.open('./Images/SUMMG.png')
        st.image(image)
        
        
    if choix == "Loyer Appartement" :
        
        st.dataframe(dep_loyer_app)
        
        image = Image.open('./Images/dep_loyer_app.png')
        st.image(image)
        
        max_col = dep_loyer_app.head(10)
        min_col = dep_loyer_app.tail(10)
        fig, ax = plt.subplots()
        ax.barh(max_col['DEP'], max_col['loyerm2'], align='center')
        ax.set_xlabel('Loyer par mètre carré')
        ax.set_ylabel('Département')
        ax.set_title('Top 10 des loyers par mètre carré dans différents départements de France')
        st.write(fig)
        
        fig, ax = plt.subplots()
        ax.barh(min_col['DEP'], min_col['loyerm2'], align='center')
        ax.set_xlabel('Loyer par mètre carré')
        ax.set_ylabel('Département')
        ax.set_title('Top derniers des loyers par mètre carré dans différents départements de France')
        st.write(fig)
        
        st.markdown ("Comme observé sur les graphes ci-dessus, le département du 75 abrite les loyers les plus coûteux en France,")
        st.markdown (" à l'inverse du 53 qui a les loyers les moins coûteux")
        
    if choix == "Type d'entreprise" :
        
        st.dataframe(te)
        
        data = te.head(20)
        fig, ax = plt.subplots()
        plt.bar(data['DEP'], data['indus'], color='blue', label='indus')
        plt.bar(data['DEP'], data['const'], color='orange', label='const')
        plt.bar(data['DEP'], data['CTRH'], color='green', label='CTRH')
        plt.bar(data['DEP'], data['InfoComm'], color='red', label='InfoComm')
        plt.bar(data['DEP'], data['FinAss'], color='purple', label='FinAss')
        plt.bar(data['DEP'], data['Immo'], color='brown', label='Immo')
        plt.bar(data['DEP'], data['STServAdmi'], color='pink', label='STServAdmi')
        plt.bar(data['DEP'], data['ApESS'], color='gray', label='ApESS')
        plt.bar(data['DEP'], data['AutreServ'], color='black', label='AutreServ')


        plt.title('Nombre de type d\'activité par département')
        plt.xlabel('Département')
        plt.ylabel('Nombre d\'activité ')
        st.write(fig)
        
        image = Image.open('./Images/dep_indus.png')
        st.image(image)
        image = Image.open('./Images/dep_CTRH.png')
        st.image(image)   
        image = Image.open('./Images/dep_STServAdmi.png')
        st.image(image) 


plt.legend()
plt.show()
        
         
        
bases_streamlit()
        