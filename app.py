import streamlit as st
import seaborn as sns


sns.set_theme()


from streamlit_base import bases_streamlit
from ML_streamlit import ML_stream
from ML_etudes import ML_etude
from source import sources
from intro import intro_streamlit
from ML_evaluation import ML_evaluation
from ML_predictions import ML_predi
from ML_local import local
from API import ML_api
from PIL import Image


st.set_page_config(page_title='French Industry', page_icon=":guardsman:", 
                   layout="wide", initial_sidebar_state="expanded")

# Ajouter un logo à l'en-tête de votre application
header = st.container()
with header:
    logo_image = Image.open("./images/Logo.png")
    st.image(logo_image, width=200)



def main():
    
    liste_menu = ["Introduction","Visualisation Base de donnée", "Etude Machine Learning","Evaluation du modéle","Machine Learning"," Evaluation de la prédiction","Etude Local du département 33","Démonstration API","Sources"]

  
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
        ML_predi()       
      
    if menu == liste_menu[6] :
        local()
        
    if menu == liste_menu[7] :
        ML_api()    

    if menu == liste_menu[8] :
        sources()
        
if __name__ == '__main__':
    main()
  