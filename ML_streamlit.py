import pandas as pd 
import seaborn as sns 
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np 
import joblib


# Custom function
# st.cache is used to load the function into memory
df=pd.read_csv("./Data/Data_ML.csv")



def ML_stream():
    
    st.title('Machine Learning sur les salaires moyens en France')
    st.markdown('Nous allons utiliser des modéles de Régréssions pour prédir le salaire moyen d un département')
    regr = joblib.load('./Modeles/RandomForestRegressor.joblib')
    st.dataframe(df)    
    st.subheader("Entrer le numéro de départements")
    dep = st.text_input('', 0,95)
    local = df[df['DEP'].isin([dep])]
    local.set_index('DEP', inplace = True)
    st.dataframe(local)
    st.subheader("Prediction du Salaire Moyen : ")
      
    st.code(regr.predict(local))


ML_stream()
