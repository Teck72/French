import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


Popu=pd.read_csv("./Data/Popu_DEP.csv")
dp_salaires=pd.read_csv("./Data/dp_salaires.csv")
base_etablissement_dp=pd.read_csv("./Data/base_etablissement.csv")
Popu_DEP = pd.read_csv("./Data/Popu_DEP2.csv")
Popu_Actifs = pd.read_csv("./Data/Popu_Actifs.csv")
dep_loyer_app = pd.read_csv("./Data/dep_loyer_app.csv")
te = pd.read_csv("./Data/te.csv")
te = pd.read_csv("./Data/te_100.csv")

Popu = Popu.drop(['Unnamed: 0'], axis=1)
dp_salaires = dp_salaires.drop(['Unnamed: 0'], axis=1)
base_etablissement_dp = base_etablissement_dp.drop(['Unnamed: 0'], axis=1)
dep_loyer_app = dep_loyer_app.drop(['Unnamed: 0'], axis=1)
te = te.drop(['Unnamed: 0'], axis=1)
te.set_index('DEP',inplace = True)
#base_etablissement_dp.set_index('DEP',inplace = True) 


Popu.set_index('DEP',inplace = True) 


def bases_streamlit():
    st.title("Visualisation des Bases de données")
  
    st.sidebar.markdown("# Choix de la base")

    choix = st.sidebar.radio("Choix de la base", ("Populations", "Salaire Moyen","Etablissement","Loyer Appartement","Type d'entreprise"))
 
    st.subheader(choix)
    
    
   

    if choix == 'Populations' :
        
        st.markdown("Les informations de cette base de données venant de L’INSEE nous indiquaient, par ville/village, le nombre de personne par tranche d’âge de 5 ans.")
        st.markdown("Nous avons travaillé cette base de données afin d’obtenir par département le nombre de personne par catégorie suivante : ")
        st.markdown("Enfants : jusqu’à 15 ans (Non actifs) ")
        st.markdown("Juniors : de 16 ans à 29 ans (Actifs) ")
        st.markdown("Séniors : 30 ans à 44 ans (Actifs) ")
        st.markdown("Masters : 45 ans à 64 ans (Actifs) ")  
        st.markdown("Ainés : 65 ans et plus. (Non Actifs) ")              
        st.dataframe(Popu)
        variable = st.multiselect("Visulation de la distribution des données :", Popu.columns)
       
        # Visualisation de la distribution de la ou des colonnes choisis précédement
        fig = px.box(Popu, y=variable)
        st.plotly_chart(fig) 
        
         
        
        # Créer un DataFrame avec les données de population par catégorie pour chaque département
        df = pd.DataFrame({
            'Catégorie': ['Ainés', 'Enfants', 'Juniors', 'Masters', 'Séniors'],
            'Population': [Popu_DEP['Ainés'].sum(), Popu_DEP['Enfants'].sum(), 
                   Popu_DEP['Juniors'].sum(), Popu_DEP['Masters'].sum(), Popu_DEP['Séniors'].sum()]
            })

        # Créer un graphique circulaire avec Plotly
        fig = px.pie(df, values='Population', names='Catégorie', 
             title='Répartition des catégories de population en France Métropolitaine',
             labels={'Catégorie': 'Catégories'})

        # Afficher le graphique dans l'application web Streamlit
        st.plotly_chart(fig)
        st.markdown("La plus grande proportion de populations est potentiellement active et a entre 30 et 44 ans, on remarque aussi que la population d’enfants est supérieure au nombre d’ainés, ce qui est plutôt encourageant d’un point de vue économique.")
        
        # Calcule la proportion de la population active et non active
        Popu_DEP['Non_Actifs']=Popu_DEP.Enfants + Popu_DEP.Ainés
        Popu_DEP['Actifs']=Popu_DEP.Juniors + Popu_DEP.Masters + Popu_DEP.Séniors
        Popu_DEP['Total'] = Popu_DEP.Juniors + Popu_DEP.Masters + Popu_DEP.Séniors+Popu_DEP.Enfants + Popu_DEP.Ainés

        Popu_DEP2 = Popu_DEP.sort_values('Total', ascending=False)

        
          
        
        # Affichage des cartes de France sauvegardées lors de la préparation des données
        col1, col2 = st.columns(2)
        
        original = Image.open('./Images/Populations_Actif.png')
        col1.header("Actif")
        col1.image(original, use_column_width=True)
      
        grayscale = Image.open('./Images/Populations_Non_Actif.png')
        col2.header("Non Actif")
        col2.image(grayscale, use_column_width=True)
        
        st.markdown("Plus la couleur est foncée plus la population décrite est présente.")
        st.markdown("Cette visualisation est en en corrélation avec les graphiques suivants qui indique les 10 départements les plus peuplés et les moins peuplés de personnes Actives.")
        st.markdown("Les populations (actives et non actives) sont concentrés autours villes et des gros « pôles » économiques français (Paris, Lille, Dunkerque, Bordeaux, Lyon, Marseille, …).")
        
        max_col = Popu_DEP2.head(10)

        # Créer un graphique à barres empilées avec Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(x=max_col['DEP'], y=max_col['Actifs'], name='Actifs', marker_color='#3ED8C9'))
        fig.add_trace(go.Bar(x=max_col['DEP'], y=max_col['Non_Actifs'], name='Non Actifs', marker_color='#EDFF91'))

        fig.update_xaxes(visible=True, type='category', categoryorder='array',
                 categoryarray=max_col[max_col[['Actifs', 'Non_Actifs']].ne(0).any(axis=1)]['DEP'])
        fig.update_layout(
            title='Répartition des populations actives et non actives sur les 10 départements les plus peuplés de France',
            xaxis_title='Départements',
            yaxis_title='Population',
            barmode='stack'
            )


        st.plotly_chart(fig)
        
        
        st.markdown("Nous avons ici la répartition des populations actives et non actives sur les 10 département les plus peuplés de France.")
        st.markdown("On retrouve le Nord avec Lille, Tourcoing, Roubaix, Dunkerque, le département de Paris,  les Bouches du Rhône avec Marseille, le Rhône avec Lyon, la région parisienne (92,93), la Gironde avec Bordeaux, le Pas-de-Calais, la région parisienne (78,77).")  
 
    
    if choix == 'Salaire Moyen' :
        
        st.markdown("Cette base de données nous indique tous les salaires moyens par ville/Village par catégorie suivantes : ")
        st.markdown("  - Moyenne de tous les salaires")
        st.markdown("  - 18-25 ans ")
        st.markdown("  - 26-50 ans")
        st.markdown("  - Plus de 50 ans")
        st.markdown("  - Cadre")
        st.markdown("  - Cadre Moyen")
        st.markdown("  - Employé")
        st.markdown("  - Travailleur (travailleurs indépendant, artisan, …)")
        st.markdown("  ")
        st.markdown("Nous avons travaillé cette base pour enlever toutes les colonnes faisant référence au sexe et nous avons décidé de regrouper toutes ces données par département.")
        
              
        
        
        st.dataframe(dp_salaires)
        
        variable = st.multiselect("Visulation de la distribution des données :", dp_salaires.columns[1:])
        
        # Visualisation de la distribution de la ou des colonnes choisis précédement
        fig = px.box(dp_salaires, y=variable)
        st.plotly_chart(fig)    
    
        # Affichage de la carte de France
        image = Image.open('./Images/SNHM.png')
        st.image(image)
    
        st.markdown("*Ici on remarque que les salaires les plus élevés sont regroupés autour des grandes villes de France et non pas forcément dans les zones les plus peuplés.*")
        st.markdown("*On remarque notamment que toute la zone Nord et Nord-Pas-de-Calais est plutôt claire alors que très peuplés.*")
    
        # Graphiques à barres qui montrent les 10 départements ayant les salaires net moyen les plus élevés et les plus bas.
        max_col = dp_salaires.head(10)
        min_col = dp_salaires.tail(10)

        fig1 = px.bar(max_col, x="DEP", y="SNHM", color="DEP", title="10 départements ayant les salaires net moyen les plus élevés")
        fig1.update_layout(xaxis=dict(type='category'))

        fig2 = px.bar(min_col, x="DEP", y="SNHM", color="DEP", title="10 départements ayant les salaires net moyen les plus bas")
        fig2.update_layout(xaxis=dict(type='category'))
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        
        st.markdown ( "*Sur le graphe ci-dessus, on constate que les départements 75,92 et 78 sont ceux ayant les salaires net moyen les plus élevés.*")
        st.markdown ( "*Les 10 départements qui ont les salaires net moyens les plus bas ont presque le même niveau de salaire net moyen égal à un peu moins que 13.33 euros/heure (salaire net horaire moyen en France)*")
         
        
        # Graphique en boîte qui montre la comparaison des salaires par type d’employé et par département.
        fig1 = px.box(dp_salaires, y='cadre_SNHM')
        fig2 = px.box(dp_salaires, y='cadre_moyen_SNHM')
        fig3 = px.box(dp_salaires, y='employé_SNHM')
        fig4 = px.box(dp_salaires, y = 'travailleur_SNHM')
        y_range = [0, dp_salaires[['cadre_SNHM', 'cadre_moyen_SNHM', 'employé_SNHM', 'travailleur_SNHM']].max().max()]
        fig = make_subplots(rows=1, cols=4, subplot_titles=("Cadres", "Cadres moyens", "Employés", "Travailleurs"))


        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)
        fig.add_trace(fig3.data[0], row=1, col=3)
        fig.add_trace(fig4.data[0], row=1, col=4)
        
        fig.update_yaxes(range=y_range, row=1, col=1)
        fig.update_yaxes(range=y_range, row=1, col=2)
        fig.update_yaxes(range=y_range, row=1, col=3)
        fig.update_yaxes(range=y_range, row=1, col=4)

        fig.update_layout(title="Comparaison des salaires par type d'employé et par département")


        st.plotly_chart(fig)
  
    
    
            
        st.markdown ( "*Pour les employés et les travailleurs, la tranche salariale est approximativement la même bien que les travailleurs aient un salaire plus élevé.*") 
        st.markdown ( "*Nous constatons une forte inégalité des salaires des cadres par rapport à ceux des autres catégories. Les cadres et les cadres moyens ont un salaire nettement plus élevé que les autres catégories de personnes en emploi.*")
     
    
    if choix == 'Etablissement' :
        
       
        st.dataframe(base_etablissement_dp)
        image = Image.open('./Images/SUMMG.png')
        variable = st.multiselect("Visulation de la distribution des données :", base_etablissement_dp.columns)
        # Visualisation de la distribution de la ou des colonnes choisis précédement
        fig = px.box(base_etablissement_dp, y=variable)
        st.plotly_chart(fig)
        
        
        # Graphique circulaire qui montre les 10 départements ayant le plus de moyennes et grandes entreprises.
        top = base_etablissement_dp.sort_values(by='SumMG', ascending=False).head(10)
        fig = px.bar(top, x='DEP', y='SumMG', text='SumMG')
        
        fig.update_layout(xaxis=dict(type='category'),margin=dict(l=100, r=50, t=70, b=50))
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title='Département', yaxis_title='Nbre Moyennes et Grandes Entreprises')
        st.plotly_chart(fig)
        st.markdown("*Ce graphique nous permet d’identifier les 10 départements ayant le plus de moyennes et grandes entreprises.*")
        st.markdown("*Nous constatons une grande différence entre les départements du 75 (paris) et  du 33 (Gironde avec Bordeaux), le somme des moyennes et grandes entreprises du 75 est environ 2 fois plus élevé que celle du 33 (le dixième département). Cependant nous pouvons nous poser la question de la véracité de ces données concernant Paris puisque peut être s’agit-il d’adresse postale uniquement.*")
        
        
    if choix == "Loyer Appartement" :
        
        st.markdown("Nous avons décidé de rajouter cette base de données pour ajouter une variable explicative non corrélée aux autres afin d’augmenter nos scores de Machine Learning.")
        st.markdown("Après étude de cette base de données comprenant différents indicateurs, nous décidons de garder uniquement 2 colonnes : le département et le loyer moyen par m² des appartements.")
        
        st.dataframe(dep_loyer_app)
        # Visualisation de la distribution
        
        st.markdown("Distribution de la variable")
        st.markdown("  ")
        fig = px.box(dep_loyer_app, y=dep_loyer_app['loyerm2'])
        st.plotly_chart(fig) 
        
        # Affichage carte de France
        image = Image.open('./Images/dep_loyer_app.png')
        st.image(image)
        
        st.markdown("*On remarque ici que les loyers les plus élevés se situent autour de Paris, de l’Iles de France, de Marseille, dans les Alpes maritime et en Haute-Savoie.*")
        st.markdown("*On retrouve le détail de cette carte dans les graphes ci-dessous montrant le top des 10 départements avec les loyers par m² les plus élevés et le plus faibles.*")
        
        #  Graphiques à barres horizontales qui montrent les 10 départements ayant les loyers par mètre carré les plus élevés et les moins élevés
        max_col = dep_loyer_app.head(10)
        

        fig1 = px.bar(max_col, x="DEP", y="loyerm2", color="DEP", orientation='v', title="Top 10 des prix des loyers par mètre carré dans différents départements de France")
        fig1.update_layout(xaxis=dict(type='category',title="", tickfont=dict(size=12), tickmode='array', tickvals=max_col['DEP'], ticktext=max_col['DEP']),
                   yaxis=dict(title="Loyer par mètre carré", tickprefix="€", ticksuffix="/m²", nticks=10),
                   margin=dict(l=100, r=50, t=70, b=50))

        min_col = dep_loyer_app.tail(10)
       

        fig2 = px.bar(min_col, x="DEP", y="loyerm2", color="DEP", orientation='v', title="Flop 10 des prix des loyers par mètre carré dans différents départements de France")
        fig2.update_layout(xaxis=dict(type='category',title="", tickfont=dict(size=12), tickmode='array', tickvals=min_col['DEP'], ticktext=min_col['DEP']),
                   yaxis=dict(title="Loyer par mètre carré", tickprefix="€", ticksuffix="/m²", nticks=10),
                   margin=dict(l=100, r=50, t=70, b=50))

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)


    if choix == "Type d'entreprise" :
        
        st.markdown("Pour la même raison que l’ajout de la base de données précédente, nous avons décidé d’ajouter une base de données gouvernementales nous indiquant le nombre d’entreprise par département selon leur secteur d’activité : ")
        st.markdown("-	indus = Industrie ")
        st.markdown("-	const = Construction  ")
        st.markdown("-	CTRH = Commerce, transports, hébergement et restauration ")
        st.markdown("-	InfoComm = Information et communication ")
        st.markdown("-	FinAss = Activités financières et d'assurance ")
        st.markdown("-	Immo = Activités immobilières  ")
        st.markdown("-	STServAdmi = Activités spécialisées, scient et techn, et activités de service et administratif ")
        st.markdown("-	ApESS = Administration publique, enseignement, santé et action sociale ")
        st.markdown("-	AutreServ = Autres activités de services")
        
               
     
        st.dataframe(te)
        
        variable = st.multiselect("Visulation de la distribution des données :", te.columns)
        # Visualisation de la distribution de la ou des colonnes choisis précédement
        fig = px.box(te, y=variable)
        st.plotly_chart(fig)    
        
        st.markdown("Nous vous proposons de regarder la composition des entreprises par région avec un détail au département.")
        st.markdown( "  ")
        
        modele = st.selectbox("Choix de la région :",
                              ("Auvergne-Rhône-Alpes","Bourgogne-Franche-Comté","Bretagne","Centre-Val de Loire","Corse",
                               "Grand Est","Hauts-de-France","Île-de-France","Normandie","Nouvelle-Aquitaine",
                               "Occitanie","Pays de la Loire","Provence-Alpes-Côte d'Azur"))
        if modele == 'Auvergne-Rhône-Alpes' :
            data = te.filter(items=['01','03','07','15','26','38','42','43','63','69','73','74'], axis =0)
        if modele == 'Bourgogne-Franche-Comté' :
            data = te.filter(items=['39','58','70','71'], axis =0)    
        if modele == 'Bretagne' :
            data = te.filter(items=['22','29','35','56'], axis =0) 
        if modele == 'Centre-Val de Loire' :
            data = te.filter(items=['18','28','36','37','41','45'], axis =0)    
        if modele == 'Corse' :
            data = te.filter(items=['2A','2B'], axis =0)            
        if modele == 'Grand Est' :
            data = te.filter(items=['08','10','51','52','54','55','57','67','68','88'], axis =0)
        if modele == 'Hauts-de-France' :
            data = te.filter(items=['02','59','60','62','80'], axis =0)
        if modele == 'Île-de-France' :
            data = te.filter(items=['75','78','77','91','92','93','94','95'], axis =0)
        if modele == 'Normandie' :
            data = te.filter(items=['14','27','50','61','76'], axis =0)
        if modele == 'Nouvelle-Aquitaine' :
            data = te.filter(items=['33','40','47','64'], axis =0)    
        if modele == 'Occitanie' :
            data = te.filter(items=['09','11','12','30','31','32','34','46','48','65','66','81','82'], axis =0)
        if modele == 'Pays de la Loire' :
            data = te.filter(items=['44','49','53','72','85'], axis =0)
        if modele == "Provence-Alpes-Côte d'Azur" :
            data = te.filter(items=['06','13','83','84'], axis =0)

        
        
        data=data.reset_index()
        
        filtered_data = data[(data[['indus', 'const', 'CTRH', 'InfoComm', 'FinAss', 'Immo', 'STServAdmi', 'ApESS', 'AutreServ']] > 0).any(axis=1)]
       

        fig = px.bar(filtered_data, x='DEP', y=['indus', 'const', 'CTRH', 'InfoComm', 'FinAss', 'Immo', 'STServAdmi', 'ApESS', 'AutreServ'],
             color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(title='Nombre de type d\'activité par département', xaxis_title='Département', yaxis_title='Nombre d\'activité')
        st.plotly_chart(fig)
        st.markdown("Nous vous proposons enfin de regarder la répartition des entreprises sur la toute la France selon leur secteur d’activité que vous pouvez choisir.")
        
        modele = st.selectbox("Choix du type d'entreprise pour la visualisation :",
                              ('Industriel','CTRH','STServAdmi','ApESS','AutreServ','Const','FinAss','Immo','InfoComm'))
        if modele == 'Industriel' :
            image = Image.open('./Images/dep_indus.png')
            st.image(image)
            
        if modele == 'CTRH' :   
            image = Image.open('./Images/dep_CTRH.png')
            st.image(image)   
                
        if modele == 'STServAdmi' :
            image = Image.open('./Images/dep_STServAdmi.png')
            st.image(image) 
            
        if modele == 'ApESS' :
            image = Image.open('./Images/dep_ApESS.png')
            st.image(image) 
            
        if modele == 'AutreServ' :
            image = Image.open('./Images/dep_AutreServ.png')
            st.image(image)     
              
        if modele == 'Const' :
             image = Image.open('./Images/dep_const.png')
             st.image(image)    
             
        if modele == 'FinAss' :
             image = Image.open('./Images/dep_FinAss.png')
             st.image(image)      
        
        if modele == 'Immo' :
             image = Image.open('./Images/dep_Immo.png')
             st.image(image)           
   
        if modele == 'InfoComm' :
             image = Image.open('./Images/dep_InfoComm.png')
             st.image(image)  
             
        


     
         
        
bases_streamlit()
        