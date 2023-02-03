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



base_etablissement=pd.read_csv("base_etablissement_par_tranche_effectif.csv")
geo=pd.read_csv("name_geographic_information.csv")
salary=pd.read_csv("net_salary_per_town_categories.csv")
population=pd.read_csv("population.csv")





st.title("Visualisation des Bases de données")

length = 30000
bins=500

st.sidebar.markdown("# Choix de la base")

choix = st.sidebar.radio("Choix de la base", ("Populations", "Salaire", "Base Etablissement"))


st.subheader(choix)

if choix == 'Populations' :
    
    population.info()

#"""on remarque la base ne contient que CODGEO et pas d'information sur le département ou la région
#le codgeo est un object, il faut comprendre pourquoi afin de transformer en int comme le code_insee
#"""

    population.AGEQ80_17.unique()

#"""On voit ici que l'age de la population est coupé par tranche de 5 ans"""

    population.NIVGEO.unique()

#il y a donc une seule valeur unique qui est COM. je peux donc la supprimer
    population.drop('NIVGEO',axis=1, inplace=True)
#population total de la base


    population.CODGEO.unique()

#"""On a du mal à voir ce qu'il se passe sur CODGEO et pourquoi c'est un objet. Nous allons essayer de selection une ligne avec comme exemple Ajaccio, une ville de corse."""

    population[population.LIBGEO=='Ajaccio']

#"""On remarque ici que le CODE GEO contient une lettre, pas extension c'est la même chose pour le département 2B.
#Nous décidons alors qu'il faudra les remplacer par le code_insee de la base geo.
#"""

 
#la notion de mode cohabitation nous importe Il faut donc supprimer MOCO mais attention à ne pas perdre de données.
#IDEM pour la colonne SEXE

#groupby
    population = population.drop(['MOCO','SEXE'],axis=1)


    
#la base de population est prête, je peux maintenant regrouper les catégories d'ages.
    population=population.assign(CAT_AGE='a')
    population.loc[(population.AGEQ80_17<=15),'CAT_AGE']='Enfants'
    population.loc[(population.AGEQ80_17>15) & (population.AGEQ80_17<30),'CAT_AGE']='Juniors'
    population.loc[(population.AGEQ80_17>=30) & (population.AGEQ80_17<45),'CAT_AGE']='Séniors'
    population.loc[(population.AGEQ80_17>=45) & (population.AGEQ80_17<=65),'CAT_AGE']='Masters'
    population.loc[(population.AGEQ80_17>65),'CAT_AGE']='Ainés'
    


#je positionne la nouvelle variable CAT_AGE avant nombre

    col = population.pop('CAT_AGE')
    population.insert(loc= 3 , column= 'CAT_AGE', value= col)
 
#je peux maintenant supprimer la colonne AGEQ80_17
    population=population.drop(columns =['AGEQ80_17'])


#Nous pouvons de nouveau sommer les CAT_AGE
    population = population.groupby(['CODGEO','LIBGEO','CAT_AGE']).sum()

    population = population.reset_index()
  


#nous devons maintenant obtenir 1 ligne par CODGEO avec en colonne la CAT_AGE et en ligne le Nombre
    population = population.pivot(index = ['CODGEO','LIBGEO'], columns = 'CAT_AGE', values = 'NB')
    
    population['Ainés'].sum()+population['Enfants'].sum()+population['Juniors'].sum()+population['Masters'].sum()+population['Séniors'].sum()


#"""Les populations en âge de travailler sont les populations Juniors, Masters et Séniors. Les enfants étants trop jeunes et les ainés étants des retraité.
#Nous crééons ainsi 2 nouvelles catégories : Actifs (personnes en âge de travailler) et les NON Actifs (personnes PAS ou PLUS en âge de travailler)

#"""

    population['Actifs'] = population.Juniors + population.Masters + population.Séniors
    population['Non_Actifs']=population.Ainés + population.Enfants

    population = population.reset_index()

     
#"""Nous souhaitons connaitre la population des DOM TOM c'est à dire des CODGEO supérieur à 95999, pour cela nous devons pouvoir convertir les CODGEO en INT.

#"""

#créons un df geocorse avec uniquement les données de la corse que nous voulons utiliser pour le CODGEO de notre base population
    geocorse = geo[geo.nom_région=='Corse']

#dans geocorse je n'ai besoin que des nom_de ville et du code_insee
    geocorse = geocorse.drop(['EU_circo','code_région','nom_région','chef.lieu_région','numéro_département','nom_département','préfecture','numéro_circonscription','codes_postaux'], axis=1)

#je remplace nom_commune par LIBGEO
    geocorse = geocorse.rename(columns = {'nom_commune' : 'LIBGEO'})

#dans la base populationtest nous mettons le LIBGEO en dernier pour faire la jointure
    populationtest = population.copy()
  
    
    col = populationtest.pop('LIBGEO')
    populationtest.insert(loc= 8 , column= 'LIBGEO', value= col)

  
#"""un premier run des quelques lignes de code nous a permis de nous rendre compte que le merge sur des nom_propre était sensible et il y avait 4 NAN qui ressortait, nous décidons donc de remplacer les 4 orthographe différents par ceux de la base geo"""

#je relance le code en plaçant ses replace avant le merge
    populationtest.replace({
        "L'Île-Rousse" : "L'Ile-Rousse" , "Évisa" : "Evisa" , "Pianottoli-Caldarello" : "Pianotolli-Caldarello", "Érone" : "Erone"}, inplace = True)

#je peux maintenant merger la base population en populationcorse 
    populationcorse = populationtest.merge(geocorse, on = "LIBGEO", how = 'left')



#on remplace les NAN par le CODGEO
    populationcorse['code_insee']= populationcorse['code_insee'].fillna(populationcorse.CODGEO)

   
#nous pouvons donc convertir en INT nos code_insee
    populationcorse['code_insee'] = populationcorse['code_insee'].astype(int)

    Populations = populationcorse.copy()


#notre étude se focalise uniquement sur la france métropolitaine + la corse, nous décidons de supprimer les DOMTOM
    Populations.drop(Populations[(Populations['code_insee'] >95999)].index, inplace=True)
    

#nous allons rajouter les numéro de département qui se situe dans la base géo. pour cela nous allons garder unique les colonnes code_insee et numéro de département et supprimer tous les doublons
    geodep = geo.drop(['EU_circo','code_région','nom_région','chef.lieu_région','nom_commune','nom_département','préfecture','numéro_circonscription','codes_postaux'], axis=1)

   
#j'inverse les colonnes numéro_département et code_insee
    col = geodep.pop('numéro_département')

    geodep.insert(loc= 1 , column= 'numéro_département', value= col)

#supression des doublons
    geodep.drop_duplicates(keep = 'first', inplace=True)

#je peux maintenant ajouter le département que je vais appeler DEP à partir de la base geo et du code_insee qui est juste pour toutes les communes de la france metropolitaine + corse
    Popu = Populations.merge(geodep, on = 'code_insee', how = 'left')

 
#"""nous décidons de remplacer les valeurs manquantes (cela veut dire que la base géo n'était pas complète)par les code_insee puis nous remplaceront par les 2 premiers caractère ==ERREUR DE CODE, NOU SUPPRIMON CES LIGNES NAN!"""

    Popu['numéro_département']= Popu['numéro_département'].fillna(Popu.code_insee)

    Popu['numéro_département'].replace([14472,26020,31300,35317,52033,52124,52266,52278,52465,55138,61483,62847,88106,89326,55039,55050,55139,55189,55239,55307],[14,26,31,35,52,52,52,52,52,55,61,62,88,89,55,55,55,55,55,55], inplace = True )

   

#je renome les numéro de département par DEP
    Popu.rename(columns = {'numéro_département': 'DEP'},inplace = True)

#je peux supprimer la colonne CODGEO
    Popu = Popu.drop('CODGEO', axis = 1)


#Somme des Actifs par département
#je positionne la variable LIBGEO en premier puis code_insee puis DEP

    col = Popu.pop('LIBGEO')
    Popu.insert(loc= 0 , column= 'LIBGEO', value= col)
    col1 = Popu.pop('code_insee')
    Popu.insert(loc= 1 , column= 'code_insee', value= col1)
    col2 = Popu.pop('DEP')
    Popu.insert(loc= 2 , column= 'DEP', value= col2)
    Popu.reset_index()

#je regroupe par DEP,le nombre d'Actifs
    Popu_Actifs = Popu[['DEP','Actifs']].groupby('DEP').sum().sort_values('Actifs', ascending=False).reset_index()

#Popu_Actifs.drop(Popu_Actifs.tail(10).index,inplace = True)

    Popu_Actifs.head(1000)

#Visualisation des 10 département avec le + et le moins d'actifs
    max_col = Popu_Actifs.head(10)
    min_col = Popu_Actifs.tail(10)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6), sharey=True)

    sns.barplot(x=max_col['DEP'], y=max_col['Actifs'],  ax=ax1)


    ax1.title.set_text("10 départements avec le plus d'actifs")

    sns.barplot(x=min_col['DEP'], y=min_col['Actifs'], ax=ax2)

    ax2.title.set_text("10 départements avec le moins d'Actifs");
    
    st.write(fig)


  
#je regroupe par DEP,le nombre d'Inactifs
    Popu_Non_Actifs = Popu[['DEP','Non_Actifs']].groupby('DEP').sum().sort_values('Non_Actifs', ascending=False).reset_index()

    Popu_Non_Actifs.drop(Popu_Non_Actifs.tail(10).index,
                         inplace = True)

    Popu_Non_Actifs.head(100)

  

    
if choix == 'Base Etablissement' :
    
# regroupement des entreprises en 4 catégories : Micro ( <10 ), petite ( 10 à 50 ), moyenne ( >50 à 100 ) et grande ( +100 )

    base_etablissement['Micro'] = base_etablissement['E14TS1'] + base_etablissement['E14TS6']
    base_etablissement['Petite'] = base_etablissement['E14TS10'] + base_etablissement['E14TS20']    
    base_etablissement['Moyenne'] = base_etablissement['E14TS50'] + base_etablissement['E14TS100']
    base_etablissement['Grande'] = base_etablissement['E14TS200'] + base_etablissement['E14TS500']

# Somme de toutes les entreprises
    base_etablissement['Sum'] = base_etablissement['E14TS10'] + base_etablissement['E14TS20'] + base_etablissement['E14TS50'] + base_etablissement['E14TS100'] + base_etablissement['E14TS200'] + base_etablissement['E14TS500']

# Somme des moyennes et grandes entreprises 
    base_etablissement['SumMG'] =   base_etablissement['E14TS50'] + base_etablissement['E14TS100'] + base_etablissement['E14TS200'] + base_etablissement['E14TS500']

# Elimination des colonnes non utiles : E14TST remplacé par la colonne Sum qui ne compte pas les entreprises de taille inconnue

    Colonne = [
        'CODGEO',
        'LIBGEO', 'REG', 'DEP',
        'Micro', 'Petite', 'Moyenne', 'Grande',
        'Sum','SumMG']

    base_etablissement = base_etablissement[Colonne]



    print(base_etablissement.head(2000))

#"""Nous allons travailler au niveau départementale :

#1.   Regroupement par département
#2.   Suppression des départements d'Outre Mer qui ne sont pas représentatif au vue du nombre de moyenne et grande entreprise



#"""

# Regroupment par département des moyennes et grandes entretprises 


    dp = base_etablissement[['DEP', 'SumMG','Micro','Petite','Moyenne','Grande','Sum']].groupby('DEP').sum().sort_values('SumMG', ascending=False).reset_index()



    dp_om = dp[(dp['DEP'] > '95')]




# Suppression des département d'outre mer : Notre étude porte sur la Fance métropolitaine, le nombre de moyenne et grande entreprise sont petites.

    dp.drop(dp[(dp['DEP'] > '95')].index, inplace=True)


    print(dp.head(20))

    max_col = dp.head(10)
    min_col = dp.tail(10)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6), sharey=True)

    sns.barplot(x=max_col['DEP'], y=max_col['SumMG'],  ax=ax1)


    ax1.title.set_text("10 départements avec le plus de Grande et Moyenne entreprise")

    sns.barplot(x=min_col['DEP'], y=min_col['SumMG'], ax=ax2)

    ax2.title.set_text("10 départements avec le moins de Grande et Moyenne entreprise");
    
    st.write(fig)

