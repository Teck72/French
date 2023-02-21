import pandas as pd 
import seaborn as sns 
import streamlit as st

def sources():
    st.title("SOURCES DES DONNEES UTILISEES")
    st.markdown(" DATASCIENTEST : ")
    st.markdown("**base_etablissement_par_tranche_effectif.csv :** ")
    st.markdown("Informations sur le nombre d'entreprises dans chaque ville française classées par taille.")
    st.markdown("**name_geographic_information.csv :**")
    st.markdown(" Données géographiques sur les villes françaises")
    st.markdown("**net_salary_per_town_categories.csv :**")
    st.markdown(" Salaires par villes française par catégories d'emploi, âge et sexe")
    st.markdown("**population.csv :**")
    st.markdown(" Informations démographiques par ville, âge, sexe et mode de vie")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown(" GOUVERNEMENT : ")
    st.markdown("  ")
    st.markdown("https://www.observatoire-des-territoires.gouv.fr/nombre-dentreprises-par-secteurs-dactivite ")
    st.markdown("**ent_sect.csv :**")
    st.markdown("Insee, Répertoire des entreprises et des établissements (REE-Sirene), 2021 ")
    st.markdown("  ")
    st.markdown(" https://www.data.gouv.fr/fr/datasets/carte-des-loyers-indicateurs-de-loyers-dannonce-par-commune-en-2018/ ")
    st.markdown("**indicateurs-loyers-appartements.csv :**")
    st.markdown("Indicateurs de loyers d'annonce par commune en 2018")
    st.markdown("  ")
   

