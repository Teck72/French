import pandas as pd 
import seaborn as sns 
import streamlit as st

def intro_streamlit():
    st.title("**Prédiction du salaire moyen par catégories d'emploi ou d'âge en France :**") 
    st.markdown(" L'objectif de cette étude est de prédire le salaire moyen par catégories d'emploi ou d'âge ") 
    st.markdown(" Nous avons utilisé une méthode de Machine Learning basée sur un algorithme de régression pour entraîner nos modèles prédictifs.")
    st.markdown("   ")
    st.markdown("La question de la prédiction des salaires moyens par catégories d'emploi ou d'âge est d'une grande importance pour les entreprises, les travailleurs et les décideurs politiques.")
    st.markdown("En effet, la connaissance des salaires moyens peut permettre aux entreprises de mieux comprendre les écarts salariaux et de prendre des décisions éclairées en matière de rémunération.")
    st.markdown(" De même, pour les travailleurs, connaître le salaire moyen peut aider à négocier des salaires équitables.")
    st.markdown("  ")
    st.markdown("Les données utilisées dans cette étude proviennent de sources officielles telles que DATASCIENTEST, l'observatoire des territoires et l'INSEE. ")
    st.markdown("Nous avons utilisé ces données pour entraîner notre modèle de Machine Learning à prédire les salaires moyens par catégories d'emploi ou d'âge en fonction de l'âge de la population, du type d'entreprise, du prix au m2 des locations d'appartement et de la taille des entreprises")
    st.markdown("   ")
    st.markdown("Nous espérons que cette étude permettra de mieux comprendre les facteurs qui influencent les salaires moyens en France et de fournir des informations précieuses aux entreprises, aux travailleurs et aux décideurs politiques pour améliorer la gestion des ressources humaines et la politique économique.")