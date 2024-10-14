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
from  src import RNN_DataScience as rnn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix
import itertools

DOSSIER_CSV = ds.get_RACINE_DOSSIER()

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
    
    st.write("## Analyse de donnees (RNN)")
    
    col1, col2 = st.columns(2)

    # Premier combo box dans la première colonne
    option1 = col1.selectbox('Choisissez un modèle', ['Modèle EMBEDDING SIMPLE', 'Modèle EMBEDDING + Lemmatisation',
                            'Modèle EMBEDDING + stemming','Modèle EMBEDDING + SPACY'])
    
    if option1 == 'Modèle EMBEDDING SIMPLE' :
        emb = rnn.RNN_EMBEDDING("EMBEDDING")
        st.markdown("""
            ## Les étapes du préprocessing

            <div style="font-size: 14px;">
            <h4>Étape 1 : Passage en minuscule</h4>  
            Nous transformons les majuscules en minuscules afin d'uniformiser le texte, car les étapes suivantes sont sensibles à la casse.

            <h4>Étape 2 : Tokenisation</h4>  
            La tokenisation consiste à décomposer un texte en "tokens", c'est-à-dire des mots ou des ponctuations. Cependant, des cas complexes comme les mots avec traits d’union, les dates, heures ou caractères spéciaux, nécessitent une attention particulière.

            <h4>Étape 3 : Retrait des stopwords</h4>  
            Nous retirons les "stopwords", qui sont des mots courants sans valeur ajoutée (ex : "je", "nous"). Cela permet de ne conserver que les termes significatifs, afin d’améliorer la représentation du texte.

            <h4>Étape 4 : Word Embedding</h4>  
            Le word embedding transforme les mots en vecteurs numériques représentant leur contexte sémantique et syntaxique. Par exemple, des mots proches comme "chien" et "chat" seront représentés par des vecteurs proches dans l'espace vectoriel.
            </div>   """, unsafe_allow_html=True)




    elif   option1 == 'Modèle EMBEDDING + Lemmatisation' :
        emb = rnn.RNN_LEMMER("EMBEDDING LEMMER")
        st.markdown("""
            ## Les étapes du préprocessing

            <div style="font-size: 14px;"> 
            <h4>Étape 1 : Passage en minuscule</h4>  
            Dans un premier temps, nous transformons les majuscules en minuscules car les étapes suivantes sont sensibles à la casse.

            <h4>Étape 2 : Tokenisation</h4>  
            Il s’agit de décomposer une phrase, et donc un document, en tokens. Un token est un élément correspondant à un mot ou une ponctuation.<br/>
            Cependant, de nombreux cas ne sont pas triviaux à traiter :<br/>  
            - Les mots avec un trait d’union, exemple : « peut être » et « peut-être » qui ont des significations très différentes  
            - Les dates et heures qui peuvent être séparées par des points, des slashs, des deux points  
            - Les apostrophes  
            - Les caractères spéciaux : émoticônes, formules mathématiques.

            <h4>Étape 3 : Retrait des stopwords</h4>  
            Ensuite, nous retirons les mots appartenant aux stopwords. Il s’agit de listes de mots définies au préalable, soit par l’utilisateur, soit dans des librairies existantes.<br/>
            Ces listes se composent de mots qui n’apportent aucune information, souvent très courants et présents dans la plupart des documents (ex. : je, nous, avoir).<br/>
            La suppression de ces stopwords permet de ne pas polluer la représentation des documents, en ne conservant que les mots représentatifs. Ce "nettoyage" du texte peut aussi inclure la suppression des nombres, dates, et ponctuation.

            <h4>Étape 4 : Groupement sémantique</h4>  
            Une fois les documents "nettoyés", nous avons une liste de mots porteurs de sens, séparés en tokens. Il est nécessaire de réduire les variations grammaticales en trouvant une forme commune pour chaque mot (ex. : pluriel, singulier, accords verbaux). Deux méthodes sont utilisées :<br/>  
            - La stemmatisation, qui ne prend pas en compte le contexte<br/>  
            - La lemmatisation, qui prend en compte le contexte.<br/>  
            
            <h4> La Lemmatisation :</h4> 
            La lemmatisation cherche à trouver la forme canonique d’un mot, appelée "lemme", en tenant compte du contexte dans lequel il est utilisé. Par exemple, elle différencie "nous avions" (verbe avoir) de "les avions" (pluriel du nom "avion"). Le lemme correspond à l’infinitif des verbes et à la forme au masculin singulier des noms, adjectifs et articles.

            <h4> Étape 5 : Word Embedding</h4>  
            Le word embedding représente les mots par des vecteurs de nombres réels, permettant de capturer leur contexte sémantique et syntaxique (genre, synonymes, etc.).<br/>
            Par exemple, "chien" et "chat" seront représentés par des vecteurs proches. Le modèle apprend à sélectionner les caractéristiques pertinentes, comme la notion "d’être vivant" pour rapprocher "chien" et "chat", et éloigner "chien" de "ordinateur".
            </div>   """, unsafe_allow_html=True)
    elif   option1 == 'Modèle EMBEDDING + stemming' :
        emb = rnn.RNN_STEMMER("EMBEDDING STEMMER")
        st.markdown("""
            ## Les étapes du préprocessing

            <div style="font-size: 14px;"> 
            <h4>Étape 1 : Passage en minuscule</h4>  
            Dans un premier temps, nous transformons les majuscules en minuscules car les étapes suivantes sont sensibles à la casse.

            <h4>Étape 2 : Tokenisation</h4>  
            Il s’agit de décomposer une phrase, et donc un document, en tokens. Un token est un élément correspondant à un mot ou une ponctuation. Cependant, de nombreux cas ne sont pas triviaux à traiter :<br/>  
            - Les mots avec un trait d’union, exemple : « peut être » et « peut-être » qui ont des significations très différentes  
            - Les dates et heures qui peuvent être séparées par des points, des slashs, des deux points  
            - Les apostrophes  
            - Les caractères spéciaux : émoticônes, formules mathématiques.

            <h4>Étape 3 : Retrait des stopwords</h4>  
            Ensuite, nous retirons les mots appartenant aux stopwords. Il s’agit de listes de mots définies au préalable, soit par l’utilisateur, soit dans des librairies existantes. Ces listes se composent de mots qui n’apportent aucune information, souvent très courants et présents dans la plupart des documents (ex. : je, nous, avoir).<br/>
            La suppression de ces stopwords permet de ne pas polluer la représentation des documents, en ne conservant que les mots représentatifs.<br/>
            Ce "nettoyage" du texte peut aussi inclure la suppression des nombres, dates, et ponctuation.

            <h4>Étape 4 : Groupement sémantique</h4>  
            Une fois les documents "nettoyés", nous avons une liste de mots porteurs de sens, séparés en tokens.<br/>
            Il est nécessaire de réduire les variations grammaticales en trouvant une forme commune pour chaque mot (ex. : pluriel, singulier, accords verbaux). Deux méthodes sont utilisées :<br/>  
            - La stemmatisation, qui ne prend pas en compte le contexte<br/>  
            - La lemmatisation, qui prend en compte le contexte.<br/>  
            
            <h4>La stemmatisation :</h4> 
            La stemmatisation (ou racinisation) réduit les mots à leur radical ou racine.

            <h4> Étape 5 : Word Embedding</h4>  
            Le word embedding représente les mots par des vecteurs de nombres réels, permettant de capturer leur contexte sémantique et syntaxique (genre, synonymes, etc.).<br/>
            Par exemple, "chien" et "chat" seront représentés par des vecteurs proches. Le modèle apprend à sélectionner les caractéristiques pertinentes, comme la notion "d’être vivant" pour rapprocher "chien" et "chat", et éloigner "chien" de "ordinateur".
            </div>   """, unsafe_allow_html=True)
    elif   option1 == 'Modèle EMBEDDING + SPACY' :  
        emb = rnn.RNN_SPACY("EMBEDDING SPACY")
        st.markdown("""
            ## Les étapes du préprocessing avec spaCy

            <div style="font-size: 14px;"> 
            <h4>Étape 1 : Passage en minuscule</h4>  
            La première étape consiste à transformer toutes les majuscules en minuscules, car certaines des étapes suivantes, comme la tokenisation et le retrait des stopwords, sont sensibles à la casse.

            <h4>Étape 2 : Tokenisation</h4>  
            Nous utilisons spaCy pour décomposer une phrase en tokens, c'est-à-dire en mots ou en ponctuations.<br/>
            SpaCy gère efficacement des cas complexes comme les mots avec des traits d'union ("peut-être"), les dates, les heures, les apostrophes, et les caractères spéciaux (émoticônes, formules mathématiques, etc.).

            <h4>Étape 3 : Retrait des stopwords</h4>  
            Grâce à spaCy, nous pouvons retirer les stopwords, qui sont des mots très courants ne portant pas de signification importante, comme "je", "nous" ou "avoir".<br/>
            SpaCy fournit des listes de stopwords pré-intégrées, que nous pouvons ajuster selon nos besoins. <br/>
            Cette étape permet également de supprimer la ponctuation, les nombres et les dates pour affiner la représentation du texte.

            
            <h4> Étape 5 : Word Embedding</h4>  
            Enfin, nous utilisons les capacités de spaCy pour générer des embeddings, c’est-à-dire des représentations vectorielles des mots.<br/>
            Cela permet de capturer les relations sémantiques et syntaxiques entre les mots, comme la proximité entre "chien" et "chat", et la distance entre "chien" et "ordinateur".<br/>
            SpaCy facilite cette étape avec des modèles d’embeddings pré-entraînés comme **Word2Vec** ou **GloVe**.
            </div>   """, unsafe_allow_html=True)
           
    
    
    st.write(option1)
   
    train_acc,val_acc,tloss,tvalloss = emb.restore_fit_arrays()
   
    y_orig,y_pred = emb.restore_predict_arrays()
    df_pred = emb.restore_predict_dataframe()
    df_pred_formatte = df_pred.applymap(format_number)
    st.markdown("""**Catégories prédites par ordre d'importance** :""")
    st.write("Voici un échantillon de produits avec pour chaque code produit , les 3 meilleurs prédictions de codes produit trouvées par ce modèle (codes et pourcentages)")
    
    st.dataframe( df_pred_formatte.head(27))
    
    st.write("Evolution de la perte et de l'accuracy lors de l'entrainement du modèle : ")
   
    f1 = f1_score(y_orig, y_pred, average='weighted')
    plot_fit(train_acc,val_acc,tloss,tvalloss)
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.markdown("""**Performance du modèle** :""")
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)
    st.markdown("""**Rapport de classification** : """)
    st.markdown(f"```\n{classif}\n```")
    st.markdown("""**Matrice de confusion** : """)
    st_show_confusion_matrix(y_orig, y_pred)
    
    nomenclature=pd.read_csv(DOSSIER_CSV+'NOMENCLATURE.csv',header=0,encoding='utf-8',sep=';',index_col=0)
    catdict=nomenclature.to_dict()['definition']
    #cat=df_target['prdtypecode'].unique()
    markdown_text = "## Catégories présentes\n"
    for k, v in catdict.items():
        markdown_text += f"- **{k}**: {v}\n"
    st.markdown(markdown_text)
    del emb, train_acc,val_acc,tloss,tvalloss, y_orig, y_pred
    del f1, df_pred, df_pred_formatte, nomenclature, catdict, markdown_text    
    gc.collect()