# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""

import numpy as np
import streamlit as st
import pandas as pd
import configparser
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from  src import Bibli_DataScience_3 as ds


def load_data() :
    if 'df_target' not in st.session_state :
        print("load data()")
        st.session_state.df_target = pd.read_csv(MY_PATH+'Y_train_CVw08PX.csv')
        st.session_state.df = df_feats.merge(st.session_state.df_target,on='Unnamed: 0',how='inner')
        st.session_state.catdict = nomenclature.to_dict()['definition']
        st.session_state.df_langue = pd.read_csv(MY_PATH+'df_langue.csv')
        st.session_state.wc = WordCloud(background_color="black", max_words=100, stopwords=stopwordFR['MOT'].tolist(), max_font_size=50, random_state=42)
        st.session_state.df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
        st.session_state.df = st.session_state.df.merge(
        st.session_state.df_langue.drop(['Unnamed: 0', 'prdtypecode'], axis=1), on='Id', how='inner')



def relabel_titles(*args, **kwargs):
    col_val = kwargs.get('col_val', '')
    return st.session_state.catdict.get(int(col_val), col_val)

MY_PATH = ds.get_RACINE_DOSSIER()
df_feats = pd.read_csv(MY_PATH + 'X_train_update.csv')
nomenclature = pd.read_csv(MY_PATH + 'NOMENCLATURE.csv', header=0, encoding='utf-8', sep=';', index_col=0)
stopwordFR = pd.read_csv(MY_PATH + "stopwords_FR_02.csv")
load_data()

print(st.session_state.df.info())


def show():
    load_data()
    st.write("### Exploration des donnees")
    col1, col2 = st.columns(2)
    print("show()")
    # Premier combo box dans la première colonne
    option1 = col1.selectbox('Choisissez un paragraphe', ['Distribution des categories', 'Le champ désignation',
    'Le champ déscription','langues par categorie','Nuages de mots','Histogramme RGB','Histogramme des contours',
    'Moyenne des images'])
    if option1 == 'Distribution des categories' :
        st.write("### Nombre de produits par catégorie")       
        st.markdown("""
            Ce graphique représente le nombre de produits par catégorie dans le jeu d'entrainement.  Nous voyons que la distribution n'est pas uniforme et que certaines catégories sont très peu présentes.""")              
        fig1=plt.figure(figsize=(18, 10))  # Ajustez la taille selon vos besoins
        sns.countplot(data=st.session_state.df_target, x='prdtypecode', order = st.session_state.df_target['prdtypecode'].value_counts().index)
        plt.xticks(rotation=90)  # Rotation des labels de l'axe x pour une meilleure lisibilité
        plt.title("Distribution des prdtypecode")
        plt.xlabel("Code produit (prdtypecode)")
        plt.ylabel("Nombre d'occurrences")
        st.pyplot(fig1) 
        fig1 = None
    if option1 == 'Le champ désignation' :            
        st.markdown("""
            Histogramme des langues les plus utilisées pour la *désignation* de l'objet  
            et distribution de la taille du libéllé en nombre de caractères :""")    
        print(st.session_state.df.info())
        fig2, axs = plt.subplots(2, 1, figsize=(15,20))
        new_labels = ['Avec', 'Sans']
        pays_principaux = st.session_state.df['pays_design'].value_counts()[:10]
        sns.countplot(x = st.session_state.df[st.session_state.df['pays_design'].isin(pays_principaux.index)]['pays_design'],hue=st.session_state.df_langue['descr_NaN'],ax=axs[0])
        axs[0].set_xlabel('Langue')
        axs[0].set_ylabel("Nombre d'occurences")
        axs[0].legend(title='Description', labels=new_labels)
        plt.subplots_adjust( wspace=0.1,hspace=0.1)  
        sns.countplot(x = st.session_state.df_langue['design_long'],ax=axs[1])
        axs[1].set_xticks(np.linspace(0,150, 10))
        axs[1].set_xlabel('Longueur du libellé')
        axs[1].set_ylabel("Nb de produits")
        st.pyplot(fig2) 
        fig2 = None
        del axs, new_labels, pays_principaux
    if option1 == 'Le champ déscription' :          
        st.markdown("""
            Histogramme des langues les plus utilisées pour le *description* de l'objet  
            et distribution de la taille du libéllé en nombre de caractères :""")                
        fig3, axs = plt.subplots(2, 1, figsize=(15,20))
        new_labels = ['Avec', 'Sans']
        pays_principaux = st.session_state.df['pays_descr'].value_counts()[:10]
        sns.countplot(x = st.session_state.df[st.session_state.df['pays_descr'].isin(pays_principaux.index)]['pays_descr'],ax=axs[0])
        axs[0].set_xlabel('Langue')
        axs[0].set_ylabel("Nombre d'occurences")
        axs[0].legend(title='Description', labels=new_labels)
        plt.subplots_adjust( wspace=0.01,hspace=0.1)  
        sns.countplot(x = st.session_state.df_langue[(st.session_state.df_langue['descrip_long']  < 2000) ]['descrip_long'],ax=axs[1])
        axs[1].set_xticks(np.linspace(0,2000, 10))
        axs[1].set_xlabel('Longueur du libellé')
        axs[1].set_xticklabels(np.linspace(0, 2000, 10).astype(int))
        axs[1].set_ylabel("Nb de produits")
        st.pyplot(fig3) 
        fig3 = None
        del axs, new_labels, pays_principaux
    if option1 == 'langues par categorie' :         
        st.markdown("""
            Histogramme des langues les plus utilisées par catégorie:""")        
        df2 = st.session_state.df.copy()
        for prt in df2['prdtypecode'].unique():
            filtre = df2['prdtypecode']==prt
            pays_principaux=df2[filtre]['PAYS_LANGUE'].value_counts()[:8]
            df2.loc[(filtre) & (~df2['PAYS_LANGUE'].isin(pays_principaux.index)),'PAYS_LANGUE']="XX"   
        g = sns.FacetGrid(data=df2, col='prdtypecode', col_wrap=3,sharex=False, sharey=False,height=5, aspect=1.5)
        g.map_dataframe(sns.countplot, 'PAYS_LANGUE')
        for ax in g.axes.flat:
            col_val = ax.get_title().split('=')[1]  # Récupérer la valeur après le signe égal dans le titre original
            new_title = st.session_state.catdict.get(int(col_val.strip()), col_val.strip())  # Chercher le nouveau titre dans le dictionnaire
            ax.set_title(new_title)
        g.set_xticklabels( rotation=90)
        g.set_xlabels('PAYS')
        g.add_legend()
        st.pyplot(g.fig) 
        g = None
        del filtre, pays_principaux, df2
    if option1 == 'Nuages de mots' :         
        st.markdown("""
            Mots les plus fréquemment utilisés par catégorie:""")       
        df_top_40=pd.read_csv(MY_PATH+'Top40.csv')
        Lcat=st.session_state.df_target.sort_values(by = ['prdtypecode'])['prdtypecode'].unique()
        fig4, axs = plt.subplots(9, 3, figsize=(15,23))
        for c,ax in zip(Lcat,axs.flat):      
            df_cat = df_top_40[df_top_40['prdtypecode']==c]
            # Définir la variable text
            text = ""
            for mot in df_cat['mot'] : 
                text += mot + " "
            print(c,"Catégorie ",st.session_state.catdict[c] )    
            st.session_state.wc.generate(text)           # "Calcul" du wordcloud
            ax.imshow(st.session_state.wc) # Affichage
            ax.set_title( st.session_state.catdict[c][:30])
        plt.subplots_adjust( wspace=0.1,hspace=0.5)    
        st.pyplot(fig4) 
        del df_top_40, Lcat, axs, df_cat, text
        fig4 = None
    if option1 == 'Histogramme RGB' :          
        st.markdown("""
            Moyennes des valeurs des couleurs dans l'espace RGB Rouge(Red), Vert(Green) et Bleu(Blue) des images par catégorie.  
            On représente la distribution des inténsités de chaque couleur dans l'image :""")      
        DfhistoMean = pd.read_json(MY_PATH+'DfhistoMean.json', orient='records', lines=True)
        cat=st.session_state.df_target['prdtypecode'].sort_values().unique()
        fig5, axs = plt.subplots(9, 3, figsize=(15,23))
        for c,ax in zip(cat,axs.flat):
            print("c = ",c)
            dfred=DfhistoMean[(DfhistoMean['prdtypecode']==c) & (DfhistoMean['color']=='red')]
            dfgreen=DfhistoMean[(DfhistoMean['prdtypecode']==c) & (DfhistoMean['color']=='green')]
            dfblue=DfhistoMean[(DfhistoMean['prdtypecode']==c) & (DfhistoMean['color']=='blue')]
            
            red_values = np.hstack(dfred['histo'].values)
            green_values = np.hstack(dfgreen['histo'].values)
            blue_values = np.hstack(dfblue['histo'].values)
               
            ax.plot(red_values, color='red')
            ax.set_xlim([0, 256])
            ax.plot(green_values, color='green')
            ax.set_xlim([0, 256])
            ax.plot(blue_values, color='blue')
            ax.set_xlim([0, 256])
            ax.set_title( st.session_state.catdict[c][:30])
        
        plt.subplots_adjust( wspace=0.1,hspace=0.5)    
        st.pyplot(fig5) 
        del DfhistoMean, cat, axs, dfred, dfgreen, dfblue, red_values, green_values, blue_values
        fig5 = None 

    if option1 == 'Histogramme des contours' :      
        st.markdown("""
            Analyse des caractéristiques des formes présentes dans une image, en particuliers les contours et les coins des objets dans l'image.  
            On représente la fréquence d'occurence des différents nombres de coins dans les images par catégorie :""")     
        Dfcontour=pd.read_csv(MY_PATH+'Dfcontour.csv')
        cat=st.session_state.df_target['prdtypecode'].sort_values().unique()
        fig6, axs = plt.subplots(9, 3, figsize=(15, 30))
        for c, ax in zip(cat, axs.flat):     
            sns.countplot(x=Dfcontour[Dfcontour['prdtypecode'] == c]['corners'], ax=ax)   
            # Modifier cette partie pour régler le problème
            ticks = ax.get_xticks()[::2]  
            labels = [label.get_text() for label in ax.get_xticklabels()][::2]  
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_title(st.session_state.catdict[c])  # Ajouter un titre à chaque subplot 
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        st.pyplot(fig6)
        del Dfcontour, cat, axs, labels
        fig6 = None    
        
    if option1 == 'Moyenne des images' :    
        st.markdown("""
            Une image moyenne en noir et blanc est calculée pour chaque catégorie en fusionnant plusieurs images.  
            Chaque pixel a une valeur d'intensité entre 0 (noir) et 255 (blanc) :""")       
        DfColorMean = pd.read_json(MY_PATH+'DfColorMean.json', orient='records', lines=True)
        cat = st.session_state.df_target['prdtypecode'].sort_values().unique()
        fig6, axs = plt.subplots(9, 3, figsize=(15, 30))
        for index, row in DfColorMean.iterrows():
            moyenne_image = row['moyenne_image']
            c = row['prdtypecode']
            ax = axs.flatten()[index]
            # Afficher l'image moyenne
            ax.set_title(st.session_state.catdict[c][:30])
            ax.imshow(moyenne_image)
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        st.pyplot(fig6) 
        del DfColorMean, cat, axs, moyenne, image, ax 
        fig6 = None
                 