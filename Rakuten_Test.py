# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""

import streamlit as st
import os
import pandas as pd
import gc
from  src import Bibli_DataScience_3 as ds
from  src import ML_DataScience as ml
from  src import CNN_DataScience_2 as cnn
from  src import RNN_DataScience as rnn
import numpy as np
from PIL import Image
import tensorflow as tf

class RESEAU_NEURONNES:
    def __init__(self, reseau,nom_reseau):
        self.__reseau = reseau
        self.__nom_reseau = nom_reseau
        self.__model = None
        
        
    def prediction(self,objet_a_predire):
        self.__model = self.__reseau.create_modele()
        self.__model.build((None, 400, 400, 3))
        print(self.__model.summary())
        print(self.__nom_reseau+"_weight chargement ...")
        ds.load_model(self.__model,self.__nom_reseau+"_weight")
        print(self.__nom_reseau+"_weight chargé !")
        predictions = self.__model.predict(objet_a_predire)
        pred = np.argmax(predictions, axis=1)
        return pred
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

def show():

    # Objectif
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px'>
        <h2 style='color: #333;'>🎯 Objectif</h2>
        <p style='font-size: 15px;'>Il est possible de combiner les modèles <b>CNN</b> et <b>SVC</b> pour prédire des catégories à partir de deux types d'entrées : des images et des descriptions textuelles.</p>
    </div>
    """, unsafe_allow_html=True)

    # Modèles utilisés
    st.markdown("""
    <div style='background-color: #e8f4fc; padding: 15px; border-radius: 10px;'>
        <h2 style='color: #007acc;'>🧠 Modèles utilisés</h2>
        <ul style='font-size: 15px;'>
            <li><b>CNN EfficientNetB1</b> : Ce modèle est utilisé pour extraire les caractéristiques des images.</li>
            <li><b>Machine Learning (SVC)</b> : Ce modèle est utilisé pour faire une prédiction basée sur les informations textuelles.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Fonctionnalités principales
    st.markdown("""
    <div style='background-color: #f4f4e8; padding: 15px; border-radius: 10px;'>
        <h2 style='color: #444;'>📋 Fonctionnalités principales</h2>
        <ul style='font-size: 15px;'>
            <li><b>Sélection d'images :</b> L'utilisateur peut sélectionner une image dans un répertoire au format JPG/PNG. L'image est affichée et redimensionnée à 400x400 pixels avant d'être passée au modèle CNN. Ce répertpoire ne fait pas partie du jeu d'entrainement.</li>
            <li><b>Prédiction avec CNN (EfficientNetB1) :</b> Un modèle CNN basé sur EfficientNetB1 est chargé et appliqué à l'image. Les poids du modèle sont chargés avant la prédiction. La prédiction de la catégorie est ensuite décodée à l'aide d'un label encoder, et la catégorie prédite est affichée.</li>
            <li><b>Prédiction avec Machine Learning (SVC) :</b> La description associée à l'image est récupérée (désignation et description). Un modèle de machine learning (SVC) est utilisé pour faire une prédiction basée sur ces informations textuelles. La catégorie prédite par le modèle est affichée.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    uploadFile = file_selector()
    st.write('Vous avez choisi  `%s`' % uploadFile)


    # Checking the Format of the page
    if uploadFile is not None:
        img = load_image(uploadFile)
        st.image(img)
        resized_image = tf.image.resize(img, (400, 400))
        reshaped_image = tf.expand_dims(resized_image, 0)
        st.markdown("<span style='color:blue'>Prédiction par l'image (Modèle CNN)</span>", unsafe_allow_html=True)
        # CNN
        EffB1 = cnn.DS_EfficientNetB1("EfficientNetB1")
        CNN = RESEAU_NEURONNES(EffB1,"EfficientNetB1")
        y_pred = CNN.prediction(reshaped_image)
        print(y_pred)
        y_train = ds.load_ndarray('EfficientNetB1_y_train')
        label_encoder=EffB1.get_labelencoder()
        label_encoder.fit(y_train)
        pred=label_encoder.inverse_transform(y_pred)
        st.markdown("""**Code produit déterminé par le CNN à partir de l'image** : """)
        st.write("Catégorie ",pred[0])
        Lcat=EffB1.get_cat()
        catdict = EffB1.get_catdict()
        st.write(catdict[pred[0]])
        st.write("*********************************")
        st.markdown("<span style='color:red'>Prédiction par la description (Modèle Machine learning)</span>", unsafe_allow_html=True)
        # ML
        svc = ml.ML_SVC("Mon_Modele_SVC",process=False)
        #designation,description = svc.get_DF_TEST_DESCRIPTION(uploadFile.split('/')[2]) # unix
        designation,description = svc.get_DF_TEST_DESCRIPTION(uploadFile.split('\\')[2]) # windows
        st.write("designation",designation)
        st.write("description",description)
        pred=svc.predire_phrases(designation,description)
        st.markdown("""**Code produit déterminé par le modèle SCV à partir du texte** : """)
        st.write("Catégorie ",pred[0])
        st.write(catdict[pred[0]])
        CNN = None
        svc = None
        del y_pred, y_train, label_encoder, pred, description, designation
        gc.collect()
        
    else:
        st.write("Make sure you image is in JPG/PNG Format.")