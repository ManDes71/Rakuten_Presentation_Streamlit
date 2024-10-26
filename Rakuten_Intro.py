# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""

import streamlit as st
import os
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from  src import Bibli_DataScience_3 as ds


def format_number(x):
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x).replace(",", " ")
    return x


def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def file_selector(folder_path=None):
    if folder_path is None:
        folder_path = os.path.join('images', 'image_test')
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Choisissez une images à classer ', filenames)
    return os.path.join(folder_path, selected_filename)


DOSSIER_CSV = ds.get_RACINE_DOSSIER()

file_path = ds.get_RACINE_IMAGES()



def show():
    # Titre principal
    st.title('PROJET RAKUTEN')
    col1, col2 = st.columns(2)

    # Premier combo box dans la première colonne
    option1 = col1.selectbox('Choisissez un paragraphe',
                             ['Le contexte', 'Les catégories', 'les produits', 'Les images'])

    if option1 == 'Le contexte':

        # Description du projet
        st.header('1) Description du projet')

        # Sous-titre et texte pour la description du problème
        st.subheader('Description du problème')
        st.markdown("""
            L'objectif de ce défi est la classification à grande échelle des données de 
            produits multimodales (texte et image) en type de produit.  
            Par exemple, dans le catalogue de <span style='color:blue'>Rakuten France</span>, un produit avec une désignation ou un titre 
             *'Klarstein Présentoir 2 Montres Optique Fibre'* est associé à une image et 
            parfois à une description supplémentaire.  
            Ce produit est catégorisé sous le type de produit 1500.  
            Il existe d'autres produits avec des titres différents, des images différentes et éventuellement des descriptions,
            qui appartiennent au même code de produit.""", unsafe_allow_html=True)
        st.markdown("""
            Cette présentation a été construite avec le framework python **<span style='color:red'>Streamlit</span>**.  
            Streamlit est un mini serveur web permettant d'exécuter
            du code python et de manipuler des dataframes, c'est donc un framework idéal pour la datascience.  
            Ce projet nécessitant des bibliothèques bien particulières, j'ai utilisé Docker afin d'isoler ce projet dans son environnement spécifique.  
            Certaines pages du projet nécessitent un peu de calcul de la part de la bibliothèque *TensorFlow* (réseaux de neurones), j'ai donc 
            opté pour deployer ce conteneur **Docker** sur **Amazon Fargate** avec une configuration de <span style='color:blue'>*2 cpu*</span> et <span style='color:blue'>*16 Go*</span> de mémoire et en configurant l'**Auto Scaling** pour ajuster les performances aux pics de demandes.""", unsafe_allow_html=True)    

        # Introduction
        st.header('2) Introduction')
        st.markdown("""Rakuten est une société japonaise de services internet créée en février 1997.
                Rakuten France, à la suite du rachat de PriceMinister, est un site internet de vente
                en ligne pour le marché français.    
                Le but du projet est de prédire le type de chaque produit tel que défini dans le
                catalogue de Rakuten France. La catégorisation des annonces de produits se fait par
                le biais de la désignation, de la description (quand elle est présente) et des images.""")

        # Détails sur les fichiers de données
        st.markdown("""
        - `X_train_update.csv` : fichier d'entrée d'entraînement
        - `Y_train_CVw08PX.csv` : fichier de sortie d'entraînement
        - `X_test_update.csv` : fichier d'entrée de test
        - `Un fichier images.zip` est également fourni, contenant toutes les images. La décompression de ce fichier fournira un dossier nommé "images" avec deux sous-dossiers nommés "image_training" et "image_test", contenant respectivement les images d'entraînement et de test.
        """)

        
        st.markdown("""
        **Note:** Le fichier de sortie de test n'est bien sûr pas disponible. C'est en appliquant le modèle obtenu au fichier d'entrée qu'il sera généré et pourra être comparé au fichier réel.  
             Pour cela il faut participer au challenge Rakuten et envoyer son fichier. Nous devrons donc piocher dans le jeu d’entraînement pour réaliser notre jeu de test.
        """)

        # Exemple de formatage de fichier
        st.markdown("""
        #### X_train_update.csv : fichier d'entrée d'entraînement
        La première ligne des fichiers d'entrée contient l'en-tête et les colonnes sont séparées par des virgules (","). Les colonnes sont les suivantes :
        - Un identifiant entier pour le produit.
        - Désignation - Le titre du produit, un court texte résumant le produit.
        - Description - Un texte plus détaillé décrivant le produit.
        - `productid` - Un identifiant unique pour le produit.
        - `imageid` - Un identifiant unique pour l'image associée au produit.
        """)

        st.markdown("""
        #### Y_train_CVw08PX.csv : fichier de sortie d'entraînement
        Les colonnes sont les suivantes :
        - Un identifiant entier pour le produit.
        - `prdtypecode` – Catégorie dans laquelle le produit est classé.
        """)

        st.markdown("""
        **Jointure:** La liaison entre les fichiers se fait par une jointure sur l’identifiant entier présent dans les deux fichiers. """)
        st.markdown("""
        #### le fichier images.zip
        Pour un produit donné, le nom du fichier image est : 
        - image_imageid_product_productid.jpg  """)
        st.markdown("""ex : image_1263597046_product_3804725264.jpg""")



    elif option1 == 'Les catégories':
        st.markdown("""
            Il y a 27 catégories représentées par un nombre.  Voici une proposition de libellés pour chaque catégorie :""")    
        nomenclature = pd.read_csv(DOSSIER_CSV + 'NOMENCLATURE.csv', header=0, encoding='utf-8', sep=';', index_col=0)
        catdict = nomenclature.to_dict()['definition']
        # cat=df_target['prdtypecode'].unique()
        markdown_text = "## Catégories présentes\n"
        for k, v in catdict.items():
            markdown_text += f"- **{k}**: {v}\n"
        st.markdown(markdown_text)
        del nomenclature, catdict, markdown_text
    elif option1 == 'les produits':
        st.markdown("""
            Voici un echantillon des enregistrements présents dans les fichiers d'entrainement en entrée et en sortie  :""")  
        st.markdown("<p style='color: blue;'>X_train_update.csv</p>", unsafe_allow_html=True)
        df = pd.read_csv(DOSSIER_CSV + 'X_train_update.csv')
        df_formatte = df.applymap(format_number)
        st.dataframe(df_formatte.head(10))
        st.markdown("<p style='color: blue;'>Y_train_CVw08PX.csv</p>", unsafe_allow_html=True)
        df2 = pd.read_csv(DOSSIER_CSV + 'Y_train_CVw08PX.csv')
        df2_formatte = df2.applymap(format_number)
        st.dataframe(df2_formatte.head(10))
        del df, df_formatte, df2, df2_formatte
    elif option1 == 'Les images':
        st.markdown("""
        Vous pouvez selectionner une images pour vous faire une idée plus précise du jeu de données  :""")  
        #uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
        uploadFile = file_selector()
        st.write('You selected `%s`' % uploadFile)
        # Checking the Format of the page
        if uploadFile is not None:
            # Perform your Manupilations (In my Case applying Filters)
            img = load_image(uploadFile)
            st.image(img)
            nom_image = uploadFile.split('/')[2]
            print(nom_image)
            nom_image_sans_jpg = nom_image.split('.')[0]
            print(nom_image_sans_jpg)
            # Diviser la chaîne en utilisant le caractère de soulignement ('_') comme séparateur
            parties = nom_image_sans_jpg.split('_')
            print(parties)
            numero_image = int(parties[1])
            numero_produit = int(parties[3])
            
            df_test = pd.read_csv(DOSSIER_CSV + 'X_test_update.csv')
            subset = df_test[(df_test.imageid == numero_image) & (df_test.productid == numero_produit)][
                ['designation', 'description']].iloc[0]
            designation = subset[0]
            description = subset[1]
            st.markdown(f"<span style='color: blue;'>designation</span>: {designation}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: blue;'>description</span>: {description}", unsafe_allow_html=True)
        else:
            st.write("Make sure you image is in JPG/PNG Format.")
        del img, df_test, subset
        del nom_image, nom_image_sans_jpg, numero_image, numero_produit, description, designation        
        uploadFile = None
