import pandas as pd 
import seaborn as sns 
import streamlit as st

def intro_streamlit():
    st.title("**Prédiction du salaire moyen par catégories d'emploi ou d'âge en France :**") 
    st.markdown(" L'objectif de cette étude est de prédire le salaire moyen par catégories d'emploi ou d'âge ") 
    st.markdown(" Les Machines Learning utilisées sont basées sur des algorithmes de régression pour entraîner nos modèles prédictifs : Decision TreeRegressor er Random ForestRegressor.")
    st.markdown("   ")
    st.markdown("Nous avons choisi de répondre à la problématique suivante :")
    st.markdown("Au niveau départemental, quel est l’impact de la modulation de certains facteurs sur le salaire moyen ?.")
    st.markdown(" Nos facteurs à moduler :")
    st.markdown(" - La quantité d’entreprise selon leur taille et leur secteur d’activité ")
    st.markdown(" - Le coût du loyer moyen au m² ")
    st.markdown(" - La part de la population selon leur catégorie d’âge ")
    st.markdown("   ")
    st.markdown("Cette problématique nous semble très importante pour certains corps décisionnels comme les entreprises, les instances politique ou encore les personnes actives dans le monde professionnel. ")
    st.markdown("En effet, la connaissance des salaires moyens peut permettre aux entreprises de mieux comprendre les écarts salariaux et de prendre des décisions éclairées en matière de rémunération.")
    st.markdown("De même, les personnes actives dans le monde professionnel (salariés, agence d’intérim, …) pour par exaider à négocier des salaires équitables.   ")
    st.markdown("Et enfin les décideurs politiques en matière de stratégie de développement social et économique. Exemple : promotion d’une région pour l’implantation d’entreprises, décision sur la politique des prix des loyers, le tout basé sur l’évolution du salaire moyen au niveau d’un département français.")
    st.markdown("   ")
    st.markdown("Les données utilisées dans cette étude proviennent de sources officielles telles que DATASCIENTEST, l'observatoire des territoires et l'INSEE.   ")
    st.markdown("Nous avons utilisé ces données pour entraîner notre modèle de Machine Learning à prédire les salaires moyens par catégories d'emploi ou d'âge en fonction de l'âge de la population, du type d'entreprise, du prix au m2 des locations d'appartement et de la taille des entreprises.   ")
    st.markdown("   ")
    st.markdown("Nous espérons que cette étude permettra aux demandeurs de mieux comprendre les facteurs qui influencent les salaires moyens en France.")
    