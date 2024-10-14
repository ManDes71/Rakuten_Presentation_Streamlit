# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""

import streamlit as st
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from  src import Bibli_DataScience_3 as ds
from  src import CNN_DataScience_2 as cnn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix
import itertools

DOSSIER_CSV = ds.get_RACINE_DOSSIER()
print("DOSSIER_CSV = ",DOSSIER_CSV)

def format_number(x):
    if isinstance(x, float):
        return "{:,.2f}".format(x).replace(",", " ").replace(".00", "")
    elif isinstance(x, int):
        return "{:,.0f}".format(x).replace(",", " ")
    return x


def plot_fit(train_acc,val_acc,tloss,tvalloss) :
    fig1 = plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(tloss)
    plt.plot(tvalloss)
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model acc by epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    st.pyplot(fig1)

def st_show_confusion_matrix(y_orig, y_pred):
    
    
    cnf_matrix = confusion_matrix(y_orig, y_pred,labels=sorted(list(set(y_orig))))
    
    #classes = [10,2280,2403,2705,40,50,2462,1280,1281]
    classes=sorted(list(set(y_orig)))
    b=list(set(y_orig))

    fig2 = plt.figure(figsize=(15,15))

    plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')
    plt.title("Matrice de confusion")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=90)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment = "center",
                 color = "white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")

    plt.ylabel('Vrais labels')
    plt.xlabel('Labels prédits')
    st.pyplot(fig2)  

def show():
    
    st.write("### Analyse de donnees (CNN)")
    col1, col2 = st.columns(2)

    # Premier combo box dans la première colonne
    option1 = col1.selectbox('Choisissez un modèle', ['Modèle EfficientNetB1', 'Modèle InceptionV3', 'Modèle VGG16',
                                    'Modèle ResNet50','Modèle VGG19','Modèle Xception'])
    maintenant = datetime.now()
    
    # ici seul la classe DS_EfficientNetB1 est chargée pour les 6 modèles.
    # En effet dans cette page aucun calcul d'entrainement n'est fait. On utilise les classes d'objets pour utiliser les fonctions communes d'affichage.
    
    if option1 == 'Modèle EfficientNetB1' :
        Modele_cnn = cnn.DS_EfficientNetB1("EfficientNetB1")
         #deb 3        
        st.markdown("""
            Le modèle **EfficientNetB1** est un modèle de réseau de neuronnes convolutifs (CNN) conçu pour la classification d'images.  
            Il fait partie de la famille **EfficientNet** qui se distingue par son efficacité à atteindre un bon équilibre entre la performance et la consommation de ressources.  
            **EfficientNetB1** est une version plus petite que d'autres variantes (comme **B7**) mais reste suffisamment puissante pour des taches de classification d'images.  
            **EfficientNet** utilise un mecanisme de **scaling uniforme** qui améliore simultanément le profondeur, la largeur et la résolution du réseau de manière équilibrée.""")    
        #fin 3  
    elif   option1 == 'Modèle InceptionV3' :  
        Modele_cnn = cnn.DS_EfficientNetB1("Mon_Modele_Inception")
        #deb 3        
        st.markdown("""
            Le modèle **InceptionV3** utilise des blocs de convolutions **Inception** qui permettent d'appliquer pluieurs convolutions avec differentes tailles de filtres (1X, 3X3, 5X5) en parallèle.  
            Cela permet au modèle de capturer à la fois des caractéristiques locales et globales dans les images de produits.  
            Cette approche modulaire réduit le nombre de paramètres nécessaire ce qui fait d'InceptionV3 un modèle puissant tout en étant plus léger que les modèles comme VGG16.  
            **InceptionV3** est un modèle très profond avec environ 42 couches.""")    
        #fin 3  
    elif   option1 == 'Modèle VGG16' :  
        Modele_cnn = cnn.DS_EfficientNetB1("Mon_Modele_VGG16")
        #deb 3 
        st.markdown("""
            Le modèle **VGG16** se distingue par simplicité et sa profondeur (16 couches), ce qui en fait un modèle bien adapté à la classification d'image complexes.  
            VGG16 est un modèle profond avec 16 couches d'opérations convolutives entièrement connectées.  
            VGG16 a montré de très bonne performances sur des tâches de classification d'images et est pré-entrainé sur ImageNet.  
            **VGG16** est plus grand que **EfficientNet**. Il contient environ 138 millions de paramètres, ce qui signifie qu'il nécissite plus de mémoire et plus de temps d'entraînement.""")    
        #fin 3 
    elif   option1 == 'Modèle ResNet50' :  
       Modele_cnn = cnn.DS_EfficientNetB1("RESNET50")
       #deb 3 
       st.markdown("""
            **Resnet50** est un modèle de réseau de neurones convolutifs (CNN) de la famille ResNet (Residual Networks). Il se distingue par sa capacité à gérer des réseaux très profonds tout en évitant les problèmes de dégradation des performances.  
            ResNet50 utilise une architecture basée sur des locs résiduels. Ces blocs permettent au modele de contourner certains problèmes de dégradation qui surviennent souvent dans des reseaux très profonds.  
            En effet ResNet introduit des **SKIP connections** qui permettent aux informations de passer directement à travers certaines couches sans midifications facilitant l'entraînement de resaux très profonds.  
            **ResNet50** a 50 couches.""")    
        #fin 3 
    elif   option1 == 'Modèle VGG19' :  
       Modele_cnn = cnn.DS_EfficientNetB1("VGG19") 
       #deb 3 
       st.markdown("""
            **VGG19** est un modèle de réseau de neurones convolutifs (CNN) de la famille VGG,  qui est une extension de VGG16 avec une profondeur supplementaire.  
            VGG19 a une architecture similaire à VGG16 mais avec 19 couches au total et utilise des filtres 3X3 ce qui permet de capturer des informations fines à différentes echelles .  
            VGG19 utilise 144 millions de paramètres. Il est plus profond que VGG16 mais est plus coûteux en terme de mémoire et de temps.  
            En résumé **VGG19** est un modèle très puissant mais plus lourd que les alternatives comme **ResNet50** ou **EfficientNetB1**.""")    
        #fin 3    
    elif   option1 == 'Modèle Xception' :  
       Modele_cnn = cnn.DS_EfficientNetB1("Xception")       
       #deb 3 
       st.markdown("""
            **Xception (Extreme Inception)** est un modèle de réseau de neurones convolutifs (CNN) qui repose sur une version optimisée de l'architecture **Inception**.  
            Il remplace les convolutions classiques d'Inception par des **convolutions séparables en profondeur** (depthwise separable convolutions) rendant le modèle à la fois plus performant et plus efficace en terme de calcul.  
            Sur des images contenant des variations visuelles complexes (textures, motifs, formes), **Xception** est bien positionné pour capturer des informations.  
            **Xception** est composé de 36 couches convolutives entièrement basées sur les convolutions séparables en profondeur.""")    
        #fin 3  
    
    st.write(option1)
    
   
    train_acc,val_acc,tloss,tvalloss = Modele_cnn.restore_fit_arrays()
    y_orig,y_pred = Modele_cnn.restore_predict_arrays()
    f1 = f1_score(y_orig, y_pred, average='weighted')
    
    df_pred = Modele_cnn.restore_predict_dataframe()
    
    df_pred_formatte = df_pred.applymap(format_number)
    
    st.markdown("""**Catégories prédites par ordre d'importance** :""")
    st.write("Voici un échantillon de produits avec pour chaque code produit , les 3 meilleurs prédictions de codes produit trouvées par ce modèle (codes et pourcentages)")

    st.dataframe( df_pred_formatte.head(27))
    
    st.write("Evolution de la perte et de l'accuracy lors de l'entrainement du modèle : ")
    
    plot_fit(train_acc,val_acc,tloss,tvalloss)
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.markdown("""**Performance du modèle** :""")
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)
    st.markdown("""**Rapport de classification** : """)
    st.markdown(f"```\n{classif}\n```")
    st.markdown("""**Matrice de confusion** : """)
    st_show_confusion_matrix(y_orig, y_pred)
    
    nomenclature = pd.read_csv(DOSSIER_CSV+'NOMENCLATURE.csv',header=0,encoding='utf-8',sep=';',index_col=0)
    catdict = nomenclature.to_dict()['definition']
    #cat=df_target['prdtypecode'].unique()
    markdown_text = "## Catégories présentes\n"
    for k, v in catdict.items():
        markdown_text += f"- **{k}**: {v}\n"
    st.markdown(markdown_text)
    del Modele_cnn, train_acc,val_acc,tloss,tvalloss, y_orig, y_pred
    del f1, df_pred, df_pred_formatte, nomenclature, catdict, markdown_text    
    gc.collect()