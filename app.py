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


from streamlit_base import bases_streamlit
from ML_streamlit import ML_stream
from ML_etudes import ML_etude
from source import sources
from intro import intro_streamlit
from ML_evaluation import ML_evaluation
from ML_interpretation import ML_inter

def main():

    # List of pages
    liste_menu = ["Introduction","Visualisation Base de donnée", "Etude Machine Learning","Evaluation du modéle","Machine Learning"," Interpretation sur le département 69","Sources"]

    # Sidebar
    menu = st.sidebar.selectbox("selectionner votre activité :", liste_menu)

    # Page navigation
    if menu == liste_menu[0] :
        intro_streamlit()
    if menu == liste_menu[1]:
        bases_streamlit()
    if menu == liste_menu[2] :
        ML_etude()
     
    if menu == liste_menu[3] :
        ML_evaluation()
    if menu == liste_menu[4] :
        ML_stream()
        
    if menu == liste_menu[5] :
        ML_inter()       
      
    if menu == liste_menu[6] :
        sources()

if __name__ == '__main__':
    main()