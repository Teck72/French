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


Popu=pd.read_csv("Popu_DEP.csv")
Popu_Actifs =pd.read_csv("Popu_Actifs.csv")
Popu_Non_Actifs =pd.read_csv("Popu_Non_Actifs.csv")
dp_salaires=pd.read_csv("dp_salaires.csv")
base_etablissement_dp=pd.read_csv("base_etablissement_dp.csv")



Popu = Popu.drop(['Unnamed: 0'], axis=1)
dp_salaires = dp_salaires.drop(['Unnamed: 0'], axis=1)
base_etablissement_dp = base_etablissement_dp.drop(['Unnamed: 0'], axis=1)




st.title("Visualisation des Bases de données")

length = 30000
bins=500

st.sidebar.markdown("# Choix de la base")

choix = st.sidebar.radio("Choix de la base", ("Populations", "Populations Non Actifs", "Salaire Moyen","Etablissement"))


st.subheader(choix)

if choix == 'Populations' :

      st.dataframe(Popu)
     
     
      image = Image.open('Populations_Actif.png')
      image2 = Image.open('Populations_Non_Actif.png')

      st.image(image)
      st.image(image2)
    
  

if choix == 'Populations Non Actifs' :
     
    max_col = Popu_Non_Actifs.head(10)
    min_col = Popu_Non_Actifs.tail(10)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6), sharey=True)

    sns.barplot(x=max_col['DEP'], y=max_col['Non_Actifs'],  ax=ax1)


    ax1.title.set_text("10 départements avec le plus de Non actifs")

    sns.barplot(x=min_col['DEP'], y=min_col['Non_Actifs'], ax=ax2)

    ax2.title.set_text("10 départements avec le moins de Non Actifs");
    
    image = Image.open('Popu_Non_Actifs.png')


    st.image(image)
    
    st.write(fig)
    
if choix == 'Salaire Moyen' :

    st.dataframe(dp_salaires)
    
if choix == 'Etablissement' :
    
    st.dataframe(base_etablissement_dp)