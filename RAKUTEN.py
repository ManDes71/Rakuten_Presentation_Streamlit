# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:03:53 2023

Python 3.10
"""
#streamlit run /home/manuel/PycharmProjects/Streamlit/Rakuten/RAKUTEN.py
#streamlit run D:\Manuel\PROJET\STREAMLIT_DOSSIERS\Rakuten\RAKUTEN.py



import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import nltk
import spacy
import gc
import sklearn as sk
import sys

from sklearn.metrics import f1_score,confusion_matrix
import itertools

import matplotlib.pyplot as plt
 

import src
from  src import Bibli_DataScience_3 as ds

from  src  import ML_DataScience as ml
from  src import CNN_DataScience_2 as cnn
from  src import RNN_DataScience as rnn

from datetime import datetime

import tensorflow as tf

import Rakuten_Intro,Rakuten_Exploration,Rakuten_ML,Rakuten_CNN
import Rakuten_RNN,Rakuten_Test,Rakuten_Proba,Rakuten_3Modeles,Rakuten_2Modeles

# Point de vérification de santé
def health_check():
    st.write("Healthy")

#



    
    
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image    
        

def cleanup_exploration_data() :
    if 'df_target' in st.session_state :
        del st.session_state.df_target
    print("del st.session_state.df_target")
    if 'df' in st.session_state :
        del st.session_state.df
    if 'catdict' in st.session_state :
        del st.session_state.catdict
    if 'df_langue' in st.session_state :
        del st.session_state.df_langue       
    if 'wc' in st.session_state :
        del st.session_state.wc
    gc.collect()


#Si le chemin est la racine (/) ou inclut un paramètre spécifique
query_params = st.query_params

# Vérifiez si le chemin est le point de santé (/) ou inclut un paramètre spécifique
if st.query_params.get("health") == ["true"] :
    health_check()
else:
    st.sidebar.title("Sommaire")
    pages=["Contexte du projet","Exploration des donnees","Analyse de donnees (ML)"
       ,"Analyse de donnees (CNN)","Analyse de donnees (RNN)","Tester une image","Modèle probabiliste (ML)",
         "Modèle concaténation 3 modèles","Modèle concaténation LM er RNN"]
    page= st.sidebar.radio("Aller vers la page :",pages)


    if page == pages[0] :
        cleanup_exploration_data()
        Rakuten_Intro.show()
    elif page == pages[1]:
        print("Rakuten_Exploration.show()")
        Rakuten_Exploration.show()
    elif page == pages[2]:
        cleanup_exploration_data()
        Rakuten_ML.show()
    elif page == pages[3]:
        cleanup_exploration_data()
        Rakuten_CNN.show()
    elif page == pages[4]:
        cleanup_exploration_data()
        Rakuten_RNN.show()
    elif page == pages[5]:
        cleanup_exploration_data()
        Rakuten_Test.show()
    elif page == pages[6]:
        Rakuten_Proba.show()
    elif page == pages[7]:
        cleanup_exploration_data()
        Rakuten_3Modeles.show()
    elif page == pages[8]:
        cleanup_exploration_data()
        Rakuten_2Modeles.show()


# Ajout du lien "Retour au blog" en bas de la page
    st.sidebar.markdown("[Retour au blog](https://aventuresdata.com)")
   
    
    
