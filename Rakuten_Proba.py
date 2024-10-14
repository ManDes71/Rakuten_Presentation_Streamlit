# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""

import streamlit as st
import os
import gc
import pandas as pd
import numpy as np
from datetime import datetime
from  src import Bibli_DataScience_3 as ds
from  src import ML_DataScience as ml
from sklearn.metrics import f1_score,accuracy_score
from PIL import Image

def format_number(x):
    if isinstance(x, float):
        return "{:,.2f}".format(x).replace(",", " ").replace(".00", "")
    elif isinstance(x, int):
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

def combiner_dataframe(df1,df2,df3):
    all_dfs = pd.concat([df1, df2, df3], axis=1)

    # Renommer les colonnes pour les distinguer
    col_names = []
    for i, df in enumerate([df1, df2, df3]):
        for col in df.columns:
            col_names.append(f"{col}_{i}")

    all_dfs.columns = col_names

    max_prob_col = all_dfs.filter(like='ProbMax').idxmax(axis=1)
    #print(max_prob_col)
    predicted_class_col = max_prob_col.str.replace('ProbMax_', 'predicted_class_')
    #print(predicted_class_col)

    #max_prob_values = all_dfs.lookup(all_dfs.index, max_prob_col)
    #predicted_class_values = all_dfs.lookup(all_dfs.index, predicted_class_col)

    max_prob_values = all_dfs.reindex(columns=max_prob_col).to_numpy()[np.arange(len(all_dfs)), np.arange(len(max_prob_col))]
    predicted_class_values = all_dfs.reindex(columns=predicted_class_col).to_numpy()[np.arange(len(all_dfs)), np.arange(len(predicted_class_col))]



    result_df = pd.DataFrame({
        'max_classe': predicted_class_values,
        'max_prob': max_prob_values
    })
    return result_df
    
def show():
    
    st.write("### Modèle probabiliste (ML)")
    st.write("############################")        
      
    # Objectif principal
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px'>
        <h3 style='color: #333;'>🎯 Objectif principal</h3>
        <p style='font-size: 15px;'>L'objectif de cette page est d'illustrer l'approche de prédiction par 
        modèle de machine learning, en combinant plusieurs modèles pour améliorer la précision des prédictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Explication des modèles
    st.markdown("""
    <div style='background-color: #e8f4fc; padding: 15px; border-radius: 10px;'>
        <h3 style='color: #007acc;'>🧠 Modèles Utilisés</h3>
        <ul style='font-size: 15px;'>
            <li><b>SVC (Support Vector Classifier)</b></li>
            <li><b>Régression Logistique</b></li>
            <li><b>Random Forest</b></li>
        </ul>
        <p style='font-size: 15px;'>Ces modèles peuvent prédire la probabilité d'appartenance d'une observation à différentes classes.</p>
    </div>
    """, unsafe_allow_html=True)

    # Explication des métriques
    st.markdown("""
    <div style='background-color: #f4f4e8; padding: 15px; border-radius: 10px;'>
        <h3 style='color: #444;'>📊 Évaluation des Modèles</h3>
        <p style='font-size: 15px;'>Pour chacun de ces modèles, les scores de <b>F1</b> et d'<b>accuracy</b> sont affichés pour évaluer la performance des prédictions.</p>
        <p style='font-size: 15px;'>Les probabilités de prédiction pour les différents modèles sont présentées sous forme de tableaux formatés.</p>
    </div>
    """, unsafe_allow_html=True)

    # Combinaison des résultats
    st.markdown("""
    <div style='background-color: #f9e9e9; padding: 15px; border-radius: 10px;'>
        <h3 style='color: #d9534f;'>🔗 Combinaison des Prédictions</h3>
        <p style='font-size: 15px;'>Une fonction combine les résultats des trois modèles pour chaque observation et 
        détermine la classe avec la probabilité maximale, assurant la meilleure prédiction possible.</p>
    </div>
    """, unsafe_allow_html=True)
    # Ligne de séparation visuelle
    st.markdown("---")

    st.write("Examinons 5 produits du jeu de test et pour chaque cas, la classe majoritaire prédite avec sa probabilité de survenance :")
    
    st.markdown("""**Modèle_SVC** : """)
    lr1 = ml.ML_SVC("Mon_Modele_SVC",process=False)
    lr_mod_1=lr1.load_modele()
    y_orig = lr1.get_y_orig()
    y_pred = lr1.get_y_pred()
    y_test =  ds.load_ndarray('y_test')
    f1 = f1_score(y_orig, y_pred, average='weighted')
    st.write("F1 Score: ", f1)
    accuracy = accuracy_score(y_test,y_pred)
    st.write("Accuracy: ", accuracy)
    f1 = f1_score(y_test, y_pred,average='weighted')
    df_prob_svc = ds.load_dataframe('Mon_Modele_SVC_prob.csv')
    del lr_mod_1, y_orig, y_pred
    gc.collect()
    
    df_prob_svc_formatte = df_prob_svc.applymap(format_number)
    
    st.write("df_prob_svc.head()")
    st.write( df_prob_svc_formatte.head())

    st.markdown("""**Modèle_LogisticRegression** : """)
    st.write("Examinons 5 produits du jeu de test et pour chaque cas, la classe majoritaire prédite avec sa probabilité de survenance :")
    lr2 = ml.ML_LogisticRegression("LogisticRegression",process=False)
    lr_mod_2 = lr2.load_modele()
    y_orig_2 = lr2.get_y_orig()
    y_pred_2 = lr2.get_y_pred()
    X_test =  ds.load_ndarray('X_test')
   
    f1 = f1_score(y_orig_2, y_pred_2, average='weighted')
    st.write("F1 Score: ", f1)
    accuracy = lr_mod_2.score(X_test, y_orig_2)
    st.write("Accuracy: ", accuracy)
    df_prob_LR = ds.load_dataframe('LogisticRegression_prob.csv')
    del lr_mod_2, y_orig_2, y_pred_2
    gc.collect()
    df_prob_LR_formatte = df_prob_LR.applymap(format_number)
    st.write("df_prob_LR.head()")
    st.write( df_prob_LR_formatte.head())
    
      
    st.markdown("""**Modèle_RandomForestClassifier** : """)
    st.write("Examinons 5 produits du jeu de test et pour chaque cas, la classe majoritaire prédite avec sa probabilité de survenance :")
    lr3 = ml.ML_RandomForest("RandomForestClassifier",process=False)
    lr_mod_3=lr3.load_modele()
    y_orig_3 = lr3.get_y_orig()
    y_pred_3 = lr3.get_y_pred()
   
    f1 = f1_score(y_orig_3, y_pred_3, average='weighted')
    st.write("F1 Score: ", f1)
    accuracy = lr_mod_3.score(X_test, y_orig_3.values)
    st.write("Accuracy: ", accuracy)
    del lr_mod_3, y_orig_3, y_pred_3
    gc.collect()
    df_prob_RF = ds.load_dataframe('RandomForestClassifier_prob.csv')
    df_prob_RF = df_prob_RF / 100 if df_prob_RF.max().max() > 1 else df_prob_RF
    df_prob_RF_formatte = df_prob_RF.applymap(format_number)
    st.write("df_prob_RF.head()")
    st.write( df_prob_RF_formatte.head())

    df_result = combiner_dataframe(df_prob_svc,df_prob_LR,df_prob_RF)
   
    y_pred = df_result['max_classe'].astype(int)
    
    df_result_formatte = df_result.applymap(format_number)
    
    st.write("df_result.head()")
    st.write( df_result_formatte.head())
    
    #st.write("df_prob_RF.head()")
    
    st.markdown("<span style='color:red'>Prédiction par la probabilité maximale (la meilleure des 3 modèles) </span>", unsafe_allow_html=True)
    
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred,average='weighted')

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

      
    #uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
    uploadFile = file_selector()
    st.write('Vous avez choisi `%s`' % uploadFile)
    
    if uploadFile is not None:
       # Perform your Manupilations (In my Case applying Filters)
       img = load_image(uploadFile)
       st.image(img)
       # designation,description = lr1.get_DF_TEST_DESCRIPTION(uploadFile.split('/')[2])   #unix
       designation,description = lr1.get_DF_TEST_DESCRIPTION(uploadFile.split('\\')[2])   #windows
       st.write("designation",designation)
       st.write("description",description)
       y_prob1=lr1.proba_phrases(designation,description)
       y_pred1=lr1.predire_phrases(designation,description)
       df_prob1 = lr1.Calculer_df_prob(y_pred1,y_prob1)
       st.write("Probabilité SVC",y_prob1)
       del y_prob1, y_pred1
       gc.collect()
       y_prob2=lr2.proba_phrases(designation,description)
       y_pred2=lr2.predire_phrases(designation,description)
       df_prob2 = lr1.Calculer_df_prob(y_pred2,y_prob2)
       st.write("Probabilité LogisticRegression",y_prob2)
       del y_prob2, y_pred2
       gc.collect()
       y_prob3=lr3.proba_phrases(designation,description)
       y_prob3 = y_prob3 / 100 if y_prob3.max().max() > 1 else y_prob3
       y_pred3=lr3.predire_phrases(designation,description)
       df_prob3 = lr3.Calculer_df_prob(y_pred3,y_prob3)
       st.write("Probabilité RandomForestClassifier",y_prob3)
       del y_prob3, y_pred3
       gc.collect()
       st.markdown("<span style='color:red'>Résultat (code produit ayant obtenu la probabilité maximale): </span>", unsafe_allow_html=True)
       df_result= combiner_dataframe(df_prob1,df_prob2,df_prob3)
       del df_prob1,df_prob2,df_prob3
       gc.collect()
       df_result_formatte = df_result.applymap(format_number)
       st.write(df_result_formatte)
       catdict = lr1.get_catdict()
       st.write(catdict[df_result['max_classe'][0]])
    else:
       st.write("Make sure you image is in JPG/PNG Format.")   