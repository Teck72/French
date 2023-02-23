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

def main():

    # List of pages
    liste_menu = ["Visualisation Base de donnée", "Machine Learning","Etude Machine Learning","Sources"]

    # Sidebar
    menu = st.sidebar.selectbox("selectionner votre activité", liste_menu)

    # Page navigation
    if menu == liste_menu[0]:
        bases_streamlit()
    if menu == liste_menu[1] :
        ML_stream()
    if menu == liste_menu[2] :
        ML_etude()
   
    if menu == liste_menu[3] :
        sources()

if __name__ == '__main__':
    main()