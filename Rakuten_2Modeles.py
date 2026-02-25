# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""
import numpy as np
import streamlit as st
import gc
from datetime import datetime
import random
import time
from  src import Bibli_DataScience_3 as ds
from  src import ML_DataScience as ml
from  src import CNN_DataScience_2 as cnn
from  src import RNN_DataScience as rnn
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import itertools
#import gc

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from PIL import Image

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate, Flatten
from tensorflow.keras import regularizers



class RESEAU_NEURONNES_CAT:
    def __init__(self, reseau,nom_reseau):
        self.__reseau = reseau
        self.__nom_reseau = nom_reseau
        self.__model = None
        self.__model_cat = None
        
        
    def prediction(self,objet_a_predire):
        self.__model = self.__reseau.create_modele()
        self.__model_cat = Model(inputs=self.__model.input, outputs=self.__model.layers[-2].output)
        print(self.__nom_reseau+"_weight_cat")
        ds.load_model(self.__model_cat,self.__nom_reseau+"_weight_cat")
        predictions = self.__model_cat.predict(objet_a_predire)
        #pred = np.argmax(predictions, axis=1)
        return predictions

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

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image        
 

def show():
    
    st.write("### Modèle concaténation LM er RNN")
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px'>
        <h2 style='color: #333;'>🎯 Objectif</h2>
        <p style='font-size: 15px;'>La page présente un modèle qui combine un réseau de neurones récurrent (RNN) de type <b>GRU</b> avec un modèle de machine learning <b>LinearSVC</b>, en utilisant la fonction de <b>concaténation</b> de Keras. Le modèle concatène les résultats des deux approches pour produire une prédiction finale sur plusieurs classes.</p>
    </div>
    """, unsafe_allow_html=True)

    # Modèles utilisés
    st.markdown("""
    <div style='background-color: #e8f4fc; padding: 15px; border-radius: 10px'>
        <h2 style='color: #007acc;'>🧠 Modèles utilisés</h2>
        <ul style='font-size: 15px;'>
            <li><b>RNN (GRU)</b> : Ce modèle traite les séquences textuelles et utilise les couches GRU pour effectuer des prédictions. Les résultats sont évalués avec des métriques telles que le <b>F1 Score</b> et l'<b>accuracy</b>.</li>
            <li><b>LinearSVC</b> : Ce modèle de machine learning traite également des données textuelles et est évalué de manière similaire avec des <b>scores F1</b> et <b>accuracy</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Combinaison des modèles
    st.markdown("""
    <div style='background-color: #f4f4e8; padding: 15px; border-radius: 10px'>
        <h2 style='color: #444;'>🔗 Combinaison des modèles</h2>
        <p style='font-size: 15px;'>Les deux modèles, <b>GRU</b> et <b>LinearSVC</b>, sont concaténés juste avant la couche de prédiction, permettant de combiner les avantages des réseaux de neurones récurrents et des modèles de machine learning linéaires.</p>
    </div>
    """, unsafe_allow_html=True)

    # Évaluation du modèle
    st.markdown("""
    <div style='background-color: #f9e9e9; padding: 15px; border-radius: 10px'>
        <h2 style='color: #d9534f;'>📊 Évaluation du modèle</h2>
        <p style='font-size: 15px;'>Le modèle final est compilé avec l'optimiseur <b>Adam</b> et la fonction de perte <b>categorical_crossentropy</b>. 
        Un entraînement est réalisé sur plusieurs époques, avec un suivi des performances (perte, précision) via une barre de progression interactive.</p>
        <p style='font-size: 15px;'>Les performances finales sont évaluées en termes de <b>loss</b>, <b>accuracy</b>, et <b>F1 Score</b>. Les résultats sont également affichés via une <b>matrice de confusion</b> pour visualiser les prédictions du modèle.</p>
    </div>
    """, unsafe_allow_html=True)

    # Prédiction finale
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px'>
        <h2 style='color: #333;'>🔍 Prédiction finale</h2>
        <p style='font-size: 15px;'>Une prédiction est effectuée sur l'ensemble de test en combinant les données des deux modèles (<b>GRU</b> et <b>LinearSVC</b>). La <b>classe prédite</b> est ensuite affichée, et des métriques supplémentaires comme l'<b>accuracy</b> et le <b>F1 Score</b> sont calculées pour évaluer la qualité des prédictions.</p>
    </div>
    """, unsafe_allow_html=True)


        
    st.markdown("""**rappel des modèles** : """)      
   
    st.write("Modèle RNN EMBEDDING GRU") 
    gru = rnn.RNN_GRU("EMBEDDING GRU")
    with st.spinner("Chargement EMBEDDING GRU..."):
        y_orig,y_pred = gru.restore_predict_arrays()
    y_orig = np.array(y_orig).ravel().astype(int)
    y_pred = np.array(y_pred).ravel().astype(int)
    f1 = f1_score(y_orig, y_pred, average='weighted')
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)

    del y_orig, y_pred, f1, acc_score,classif
    gru = None
    
    train_X_gru = ds.load_ndarray('EMBEDDING_GRU_CONCAT2_X_train') 
    test_X_gru = ds.load_ndarray('EMBEDDING GRU_CONCAT2_X_test') 
    #train_y_gru = ds.load_ndarray('EMBEDDING GRU_y_train')
    #test_y_gru = ds.load_ndarray('EMBEDDING GRU_y_test')
   
  
    st.write("Modèle LinearSVC") 
    
    
    #lsvc =  ml.ML_LinearSVCFromModel("LinearSVCFromModel",process=False)
    #lsvc_mod=lsvc.load_modele()
    
    
    #train_X_svc = ds.load_ndarray('LinearSVCFromModel_CONCAT2_X_train') 
    #test_X_svc = ds.load_ndarray('LinearSVCFromModel_CONCAT2_X_test') 
    #train_y_svc = ds.load_ndarray('LinearSVCFromModel_CONCAT2_y_train') 
    #test_y_svc = ds.load_ndarray('LinearSVCFromModel_CONCAT2_y_test') 

   
    #lsvc =  ml.ML_LinearSVCFromModel("LinearSVC",process=False)
    #lsvc_mod=lsvc.load_modele()
    
    
    train_X_svc = ds.load_ndarray('LinearSVC_CONCAT2_X_train') 
    test_X_svc = ds.load_ndarray('LinearSVC_CONCAT2_X_test') 
    train_y_svc = ds.load_ndarray('LinearSVC_CONCAT2_y_train')
    test_y_svc = ds.load_ndarray('LinearSVC_CONCAT2_y_test')

    lsvc = None
    gc.collect()
    lr = ml.ML_LinearSVC("LinearSVC",process=False)
    with st.spinner("Chargement du modèle LinearSVC..."):
        lr_mod = lr.load_modele()
    y_orig = np.array(lr.get_y_orig()).ravel().astype(int)
    y_pred = np.array(lr.get_y_pred()).ravel().astype(int)

    f1 = f1_score(y_orig, y_pred, average='weighted')
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)

    del y_orig, y_pred, f1, acc_score, classif
    lr_mod = None


    st.markdown("""**Entrainement du modèle commun** (agrégation des 2 modèles par la fonction **Concatenate** de TensorFlow)  : environ 20 s, veuillez patientez ... """)

    print("train_X_svc.shape = ",train_X_svc.shape)
    print("train_X_gru.shape = ",train_X_gru.shape)


    [nSamp,inpShape_svc] = train_X_svc.shape
    print("nSamp = ",nSamp)
    [nSamp,inpShape_gru] = train_X_gru.shape
    print("nSamp = ",nSamp)
    
    print("inpShape_svc = ",inpShape_svc)
    print("inpShape_gru = ",inpShape_gru)
   
    
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    label_encoder = LabelEncoder()
    
    
    y_classes_converted = label_encoder.fit_transform(np.array(train_y_svc).copy())
    y_train_Network = to_categorical(y_classes_converted)
    y_classes_converted = label_encoder.transform(np.array(test_y_svc).copy())
    y_test_Network = to_categorical(y_classes_converted)

    del train_y_svc
    gc.collect()

    num_classes=27


    seq_modelsvc = Sequential()     #   svc
    seq_modelsvc.add(Dense(128, activation='relu',input_shape = (inpShape_svc,), name = "Input1"))
    seq_modelsvc.add(Dense(64, activation='relu'))
    seq_modelsvc.add(Dense(27,activation='softmax'))  # 27 neurones pour 27 classes
    
    
    
    
    seq_modelgru = Sequential()    #  gru
    seq_modelgru.add(Dense(128, activation='relu',input_shape = (inpShape_gru,), name = "Input2"))
    seq_modelgru.add(Dense(64, activation='relu'))
    seq_modelgru.add(Dense(27,activation='softmax'))  # 27 neurones pour 27 classes
    del y_classes_converted
    
    # Définir les couches d'entrée explicitement
    input_svc = Input(shape=(inpShape_svc,))
    input_gru = Input(shape=(inpShape_gru,))

    # Appeler les modèles avec les données d'entrée correspondantes
    svc_output = seq_modelsvc(input_svc)  # Utiliser train_X_svc comme entrée pour seq_modelsvc
    del input_svc
    gc.collect()
    gru_output = seq_modelgru(input_gru)  # Utiliser train_X_gru comme entrée pour seq_modelgru
    del input_gru
    gc.collect()
    # Concaténer les deux modèles
    concat_layer = Concatenate()([svc_output, gru_output])
    del svc_output, gru_output
    gc.collect()
    
    normalized_layer = BatchNormalization()(concat_layer)
    # Ajoutez des couches supplémentaires si nécessaire
    final_output = Dense(num_classes, activation='softmax')(normalized_layer)
    
    # Créer le modèle final
    final_model = Model(inputs=[seq_modelsvc.input, seq_modelgru.input], outputs=final_output)
    
    # Résumé du modèle
    final_model.summary()

    seq_modelsvc = None
    seq_modelgru = None
    #del train_X_svc, train_X_gru
    #gc.collect()  # Libération manuelle de la mémoire

    print("avant compile")
    
    # Compilation du modèle
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("apres compile")
    #print(train_X_svc.shape)
    #print(train_X_gru.shape)
    


    print("Entraînement du modèle...")

    progress_bar = st.progress(0)
    status_text = st.empty()

   

    st.write("Entraînement du modèle...")
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 3)  # 3 époques dans votre cas
            status_text.text(f"Époque actuelle : {epoch + 1}, Perte : {logs['loss']:.4f}, Précision : {logs['accuracy']:.4f}")


    # Entraînement du modèle
    with tf.device('/CPU:0'):
        final_model.fit([train_X_svc,train_X_gru], y=y_train_Network, epochs=3, batch_size=32,
                    callbacks=[ProgressCallback()])

    print("modèle entrainné")
    #ds.save_model(final_model,"Rakuten_2M_weight.weights") 
    #ds.save_ndarray(label_encoder,"Rakuten_2M_label_encoder")
    
    #ds.load_model(final_model,"Rakuten_2M_weight.weights")
    #print("modele chargé")
    #label_encoder = ds.load_ndarray("Rakuten_2M_label_encoder")
    #Évaluation du modèle
    st.write("Appliquons le modèle sur le jeu de test :")
    loss, accuracy = final_model.evaluate([ test_X_svc,test_X_gru], y=y_test_Network)
    st.write(f'Loss: {loss}, Accuracy: {accuracy}')
        
   
    
    #ds.save_model(final_model,"Rakuten_2M_weight") 
    #ds.save_ndarray(label_encoder,"Rakuten_2M_label_encoder")
    
    prediction = final_model.predict([ test_X_svc,test_X_gru])
    #print("prediction : ",prediction)
    predicted_class = np.argmax(prediction, axis=1)
    y_pred = label_encoder.inverse_transform(predicted_class)

    #print(f"La classe prédite est : {predicted_class}")
    y_orig = np.array(test_y_svc).ravel().astype(int)
    y_pred = np.array(y_pred).ravel().astype(int)
    accuracy = accuracy_score(y_orig,y_pred)
    f1 = f1_score(y_orig, y_pred,average='weighted')
    
    # Affichage des résultats
    st.markdown("""**Performance du modèle** :""")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
 
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)
    st.markdown("""**Rapport de classification** : """)
    st.markdown(f"```\n{classif}\n```")
    st.markdown("""**Matrice de confusion** : """)
    st_show_confusion_matrix(y_orig, y_pred)
    del  test_X_svc,  test_X_gru
    del y_orig, y_pred
    del acc_score, classif,f1
    del  y_train_Network, y_test_Network
    del loss, accuracy, prediction, predicted_class
    final_model = None
    gc.collect()  # Libération manuelle de la mémoire

    

