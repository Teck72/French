import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import time
import altair as alt
import seaborn as sns

sns.set_theme()



Popu_Actifs =pd.read_csv("Popu_Actifs.csv")
Popu_Non_Actifs =pd.read_csv("Popu_Non_Actifs.csv")
dp_salaires=pd.read_csv("dp_salaires.csv")
base_etablissement_dp=pd.read_csv("base_etablissement_dp.csv")





st.title("Visualisation des Bases de données")

length = 30000
bins=500

st.sidebar.markdown("# Choix de la base")

choix = st.sidebar.radio("Choix de la base", ("Populations Actifs", "Populations Non Actifs", "Salaire Moyen","Etablissement"))


st.subheader(choix)

if choix == 'Populations Actifs' :

      print(Popu_Actifs.head(100))
      max_col = Popu_Actifs.head(10)
      min_col = Popu_Actifs.tail(10)

      fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6), sharey=True)

      sns.barplot(x=max_col['DEP'], y=max_col['Actifs'],  ax=ax1)


      ax1.title.set_text("10 départements avec le plus d'actifs")

      sns.barplot(x=min_col['DEP'], y=min_col['Actifs'], ax=ax2)

      ax2.title.set_text("10 départements avec le moins d'Actifs");
    
      st.write(fig)

if choix == 'Populations Populations Non Actifs' :
     
    max_col = Popu_Non_Actifs.head(10)
    min_col = Popu_Non_Actifs.tail(10)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6), sharey=True)

    sns.barplot(x=max_col['DEP'], y=max_col['Non_Actifs'],  ax=ax1)


    ax1.title.set_text("10 départements avec le plus d'actifs")

    sns.barplot(x=min_col['DEP'], y=min_col['Non_Actifs'], ax=ax2)

    ax2.title.set_text("10 départements avec le moins d'Actifs");
    
    st.write(fig)
