# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""
import random
import time
import numpy as np
import streamlit as st
import gc
from datetime import datetime
from  src import Bibli_DataScience_3 as ds
from  src import ML_DataScience as ml
from  src import CNN_DataScience_2 as cnn
from  src import RNN_DataScience as rnn
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import itertools

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
        print(self.__nom_reseau+"_weight_3cat")
        ds.load_model(self.__model_cat,self.__nom_reseau+"_weight_3cat")
        predictions = self.__model_cat.predict(objet_a_predire)
        pred = np.argmax(predictions, axis=1)
        return pred

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
    
    st.write("### Modèle concaténation de 3 modèles")
    # Objectif
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px'>
        <h2 style='color: #333;'>🎯 Objectif</h2>
        <p style='font-size: 15px;'>Il est possible d'arrêter les réseaux de neurones avant la dernière couche. 
        Les modèles sont concaténés juste avant la prédiction grâce à la fonction <b>Concatenate()</b> de Keras. 
        Un modèle de machine learning peut également être ajouté s'il peut délivrer un modèle sous forme de combinaisons linéaires.</p>
    </div>
    """, unsafe_allow_html=True)

    # Modèles utilisés
    st.markdown("""
    <div style='background-color: #e8f4fc; padding: 15px; border-radius: 10px'>
        <h2 style='color: #007acc;'>🧠 Modèles utilisés</h2>
        <ul style='font-size: 15px;'>
            <li><b>CNN EfficientNetB1</b> : Utilisé pour extraire les caractéristiques des images, avec des performances évaluées par des <b>scores F1</b> et <b>accuracy</b>.</li>
            <li><b>RNN EMBEDDING</b> : Traite les séquences textuelles, avec des résultats également affichés en <b>métriques de performance</b>.</li>
            <li><b>LinearSVC</b> : Utilisé pour traiter les données textuelles avec des <b>scores F1</b> et <b>accuracy</b> calculés.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Combinaison des modèles
    st.markdown("""
    <div style='background-color: #f4f4e8; padding: 15px; border-radius: 10px'>
        <h2 style='color: #444;'>🔗 Combinaison des modèles</h2>
        <p style='font-size: 15px;'>Les trois modèles sont combinés à l'aide de la fonction <b>Concatenate()</b> de Keras, 
        permettant ainsi de produire une <b>prédiction unique</b> à partir des trois sources d'information.</p>
    </div>
    """, unsafe_allow_html=True)

    # Évaluation
    st.markdown("""
    <div style='background-color: #f9e9e9; padding: 15px; border-radius: 10px'>
        <h2 style='color: #d9534f;'>📊 Évaluation</h2>
        <p style='font-size: 18px;'>Les performances du modèle final sont affichées sous forme de <b>métriques</b> telles que <b>F1 Score</b> et <b>accuracy</b>. 
        Une <b>matrice de confusion</b> est générée pour visualiser les résultats des prédictions.</p>
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("""**rappel des modèles** : """)      
      
    st.write("Modèle CNN EfficientNetB1")  
    Modele_cnn = cnn.DS_EfficientNetB1("EfficientNetB1")
    with st.spinner("Chargement EfficientNetB1..."):
        train_acc,val_acc,tloss,tvalloss = Modele_cnn.restore_fit_arrays()
        y_orig,y_pred = Modele_cnn.restore_predict_arrays()
    y_orig = np.array(y_orig).ravel().astype(int)
    y_pred = np.array(y_pred).ravel().astype(int)
    f1 = f1_score(y_orig, y_pred, average='weighted')
    st.write("F1 Score: ", f1)
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.write("Accuracy: ", acc_score/100)
    
   
    Modele_cnn = None
    gc.collect()


    train_X_cnn = ds.load_ndarray('EfficientNetB1_CONCAT2_X_train') 
    test_X_cnn = ds.load_ndarray('EfficientNetB1_CONCAT2_X_test') 
    train_y_cnn = ds.load_ndarray('EfficientNetB1_CONCAT2_y_train') 
    test_y_cnn = ds.load_ndarray('EfficientNetB1_CONCAT2_y_test') 
   
    


    st.write("Modèle RNN EMBEDDING") 
    emb = rnn.RNN_EMBEDDING("EMBEDDING")
    with st.spinner("Chargement EMBEDDING..."):
        y_orig,y_pred = emb.restore_predict_arrays()
    y_orig = np.array(y_orig).ravel().astype(int)
    y_pred = np.array(y_pred).ravel().astype(int)
    f1 = f1_score(y_orig, y_pred, average='weighted')
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)

   
    emb = None
    gc.collect()
    
    train_X_gru = ds.load_ndarray('EMBEDDING_CONCAT2_X_train') 
    test_X_gru = ds.load_ndarray('EMBEDDING_CONCAT2_X_test') 
    train_y_gru = ds.load_ndarray('EMBEDDING_y_train') 
    test_y_gru = ds.load_ndarray('EMBEDDING_y_test') 
   
    st.write("Modèle LinearSVC") 
    #lsvc =  ml.ML_LinearSVCFromModel("LinearSVC",process=False)
    #lsvc_mod=lsvc.load_modele()


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
    lr = None
    gc.collect()

    train_X_svc = ds.load_ndarray('LinearSVC_CONCAT2_X_train')

    train_y_svc = ds.load_ndarray('LinearSVC_CONCAT2_y_train')



    st.markdown("""**Entrainement du modèle commun** (agrégation des 3 modèles par la fonction **Concatenate** de TensorFlow)  : environ 30 s, veuillez patientez ...   : """)


    
    [nSamp,inpShape_svc] = train_X_svc.shape
    gc.collect()
    print(nSamp)
    [nSamp,inpShape_gru] = train_X_gru.shape
    gc.collect()
    print(nSamp)
    [nSamp,inpShape_cnn] = train_X_cnn.shape
    gc.collect()
    print(nSamp)
    #[nSamp_TdIdf,inpShape_TdIdf]=X_train_TdIdf.shape
    #[nSamp_SVC,inpShape_SVC]=X_train_SVC.shape
    #print(inpShape_svc)
    #print(inpShape_gru)
    #print(inpShape_cnn)
    #print("Indices max dans X_train_pad : ", np.max(X_train_pad))
    #print(len(tokens))
    #print(X_train_seq.shape)
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    label_encoder = LabelEncoder()
    
    
    y_classes_converted = label_encoder.fit_transform(np.array(train_y_svc).copy())
    del train_y_svc
    gc.collect()
    test_X_svc = ds.load_ndarray('LinearSVC_CONCAT2_X_test')
    test_y_svc = ds.load_ndarray('LinearSVC_CONCAT2_y_test')
    y_train_Network = to_categorical(y_classes_converted)
    y_classes_converted = label_encoder.transform(np.array(test_y_svc).copy())
    y_test_Network = to_categorical(y_classes_converted)

    del  y_classes_converted
    gc.collect()
    num_classes=27


    seq_modelsvc = Sequential()     #   emp
    seq_modelsvc.add(Dense(128, activation='relu',input_shape = (inpShape_svc,), name = "Input1"))
    seq_modelsvc.add(Dense(64, activation='relu'))
    seq_modelsvc.add(Dense(27,activation='softmax'))  # 27 neurones pour 27 classes


    
    seq_modelgru = Sequential()    #  gru
    seq_modelgru.add(Dense(128, activation='relu',input_shape = (inpShape_gru,), name = "Input2"))
    seq_modelgru.add(Dense(64, activation='relu'))
    seq_modelgru.add(Dense(27,activation='softmax'))  # 27 neurones pour 27 classes
    
    seq_modelcnn = Sequential()    #  cnn
    seq_modelcnn.add(Dense(128, activation='relu',input_shape = (inpShape_cnn,), name = "Input3"))
    seq_modelcnn.add(Dense(64, activation='relu'))
    seq_modelcnn.add(Dense(27,activation='softmax'))  # 27 neurones pour 27 classes


     # Définir les couches d'entrée explicitement
    input_svc = Input(shape=(inpShape_svc,))
    input_gru = Input(shape=(inpShape_gru,))
    input_cnn = Input(shape=(inpShape_cnn,))
    

    # Appeler les modèles avec les données d'entrée correspondantes
    svc_output = seq_modelsvc(input_svc)  # Utiliser train_X_svc comme entrée pour seq_modelsvc
    gru_output = seq_modelgru(input_gru)  # Utiliser train_X_gru comme entrée pour seq_modelgru
    cnn_output = seq_modelcnn(input_cnn)  # Utiliser train_X_cnn comme entrée pour seq_modelcnn

    del input_cnn, input_gru; input_svc 
    gc.collect()

    # Concaténer les deux modèles
    concat_layer = Concatenate()([svc_output, gru_output, cnn_output])

    del svc_output, gru_output, cnn_output
    gc.collect()
    
    normalized_layer = BatchNormalization()(concat_layer)
    # Ajoutez des couches supplémentaires si nécessaire
    final_output = Dense(num_classes, activation='softmax')(normalized_layer)
    
    # Créer le modèle final
    final_model = Model(inputs=[seq_modelsvc.input, seq_modelgru.input,seq_modelcnn.input], outputs=final_output)
    
    seq_modelsvc = None
    seq_modelgru = None
    seq_modelcnn = None
    gc.collect()

    # Résumé du modèle
    final_model.summary()
    
    # Compilation du modèle
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #print(train_X_svc.shape)
    #print(train_X_gru.shape)
    #print(train_X_cnn.shape)
     #  debut
    
    # fin
    # Indiquer que l'entraînement est terminé
    #st.success("Entraînement terminé avec succès ! 🎉")

    #del res_precision, res_loss

    progress_bar = st.progress(0)
    status_text = st.empty()

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 5)  # 10 époques dans votre cas
            status_text.text(f"Époque actuelle : {epoch + 1}, Perte : {logs['loss']:.4f}, Précision : {logs['accuracy']:.4f}")


    
    # Entraînement du modèle
    with tf.device('/CPU:0'):
        final_model.fit([train_X_svc,train_X_gru,train_X_cnn], y=y_train_Network, epochs=5, batch_size=32,
                        callbacks=[ProgressCallback()])
    
    
    #ds.load_model(final_model,"Rakuten_3M_weights.weights")
    #label_encoder = ds.load_ndarray("Rakuten_3M_label_encoder")
    
    st.write("Appliquons le modèle sur le jeu de test :")
    # Évaluation du modèle
    loss, accuracy = final_model.evaluate([ test_X_svc,test_X_gru,test_X_cnn], y=y_test_Network)
    st.write(f'Loss: {loss}, Accuracy: {accuracy}')

    
    
    #ds.save_model(final_model,"Rakuten_3M_weights.weights") 
    #ds.save_ndarray(label_encoder,"Rakuten_3M_label_encoder")


    prediction = final_model.predict([ test_X_svc,test_X_gru,test_X_cnn])
    predicted_class = np.argmax(prediction, axis=1)
    y_pred = label_encoder.inverse_transform(predicted_class)

    print(f"La classe prédite est : {predicted_class}")
    y_orig = np.array(test_y_svc).ravel().astype(int)
    y_pred = np.array(y_pred).ravel().astype(int)
    accuracy = accuracy_score(y_orig,y_pred)
    f1 = f1_score(y_orig, y_pred,average='weighted')
    
    # Affichage des résultats
    #st.write(f"Accuracy: {accuracy:.4f}")
    #st.write(f"F1 Score: {f1:.4f}")
    acc_score,classif=ds.get_classification_report(y_orig, y_pred)
    st.markdown("""**Performance du modèle** :""")
    st.write("Accuracy: ", acc_score/100)
    st.write("F1 Score: ", f1)
    st.markdown("""**Rapport de classification** : """)
    st.markdown(f"```\n{classif}\n```")
    st.markdown("""**Matrice de confusion** : """)
    st_show_confusion_matrix(y_orig, y_pred)
    del  test_X_svc,  test_X_gru,  test_X_cnn
    del y_orig, y_pred
    del acc_score, classif,f1
    del  y_train_Network, y_test_Network
    del loss, accuracy, prediction, predicted_class
    final_model = None
    gc.collect()  # Libération manuelle de la mémoire

    
"""

    st.markdown("<span style='color:blue'>Prédiction par l'image (Modèle CNN)</span>", unsafe_allow_html=True)
      
    uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
    
    # Checking the Format of the page
    if uploadFile is not None:
        # Perform your Manupilations (In my Case applying Filters)
        img = load_image(uploadFile)
        st.image(img)
        st.write(uploadFile.name,"Image Uploaded Successfully")
        resized_image = tf.image.resize(img, (400, 400))
        reshaped_image = tf.expand_dims(resized_image, 0)
        print(resized_image.shape)
        st.markdown("<span style='color:blue'>Prédiction par l'image (Modèle CNN)</span>", unsafe_allow_html=True)
        designation,description = lsvc.get_DF_TEST_DESCRIPTION(uploadFile.name)
        st.write("designation",designation)
        st.write("description",description)
        X_test=lsvc.traiter_phrases(designation,description)
        mots=gru.traiter_phrases(designation,description)
        st.write(mots)
        st.write(X_test)
        #ds.load_model(model,"EfficientNetB1_weight")
        y_pred = final_model.predict([ X_test,mots,reshaped_image])
        #print("predictions",predictions)
        #y_pred = np.argmax(predictions, axis=1)
        print(y_pred)
        
        
        
    else:
        st.write("Make sure you image is in JPG/PNG Format.")
        """