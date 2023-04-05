import streamlit as st
import requests
import json


def ML_api():
    
    st.write("Saisir les variables à envoyées à l'API :")
    petite = st.number_input('%Petite', min_value=0, max_value=100, value=0)
    moyenne = st.number_input('%Moyenne', min_value=0, max_value=100, value=0)
    grande = st.number_input('%Grande', min_value=0, max_value=100, value=0)
    aines = st.number_input('%Ainés', min_value=0, max_value=100, value=0)
    enfants = st.number_input('%Enfants', min_value=0, max_value=100, value=0)
    juniors = st.number_input('%Juniors', min_value=0, max_value=100, value=0)
    masters = st.number_input('%Masters', min_value=0, max_value=100, value=0)
    seniors = st.number_input('%Séniors', min_value=0, max_value=100, value=0)
    indus = st.number_input('%indus', min_value=0, max_value=100, value=0)
    const = st.number_input('%const', min_value=0, max_value=100, value=0)
    ctrh = st.number_input('%CTRH', min_value=0, max_value=100, value=0)
    infocomm = st.number_input('%InfoComm', min_value=0, max_value=100, value=0)
    stservadmi = st.number_input('%STServAdmi', min_value=0, max_value=100, value=0)
    autreserv = st.number_input('%AutreServ', min_value=0, max_value=100, value=0)
    loyerm2 = st.number_input('Loyer/m2', min_value=0, max_value=100, value=0)


    if st.button('Envoyer'):

        data = {'%Petite': petite, '%Moyenne': moyenne, '%Grande': grande, '%Ainés': aines, '%Enfants': enfants,
                '%Juniors': juniors, '%Masters': masters, '%Séniors': seniors, '%indus': indus, '%const': const,
                '%CTRH': ctrh, '%InfoComm': infocomm, '%STServAdmi': stservadmi, '%AutreServ': autreserv, 'loyerm2': loyerm2}

        
        
        with st.echo():
             
             url = 'https://teck72.pythonanywhere.com:443'
             response = requests.post(url, json=data)

        if response.status_code == 200:
            try:
                predictions = json.loads(response.content.decode('utf-8'))
                for model_name, prediction in predictions['predictions'].items():
                    st.write('Prédiction du ', model_name, ':', prediction)
            except json.JSONDecodeError as e:
                st.write('Erreur lors de l\'analyse de la réponse JSON :', e.msg)
                st.write('Contenu de la réponse :', response.content)
        else:
            st.write('La requête a échoué avec le code', response.status_code)
            st.write('Contenu de la réponse :', response.content)

ML_api()