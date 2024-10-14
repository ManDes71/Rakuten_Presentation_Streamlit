# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:45:04 2023

@author: Manuel desplanches
"""

import streamlit as st
import numpy as np
import gc
import pandas as pd
from collections import Counter
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from src import Bibli_DataScience_3 as ds
from src  import ML_DataScience as ml
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import itertools

DOSSIER_CSV = ds.get_RACINE_DOSSIER()

def format_number(x):
    if isinstance(x, float):
        return "{:,.2f}".format(x).replace(",", " ").replace(".00", "")
    elif isinstance(x, int):
        return "{:,.0f}".format(x).replace(",", " ")
    return x


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
    
    st.write("### Analyse de donnees (ML)")
    col1, col2 = st.columns(2)

    # Premier combo box dans la première colonne
    option1 = col1.selectbox('Choisissez un modèle', ['SVC', 'LogisticRegression', 'RandomForestClassifier',
                            'GradientBoosting','XGBClassifier','DecisionTreeClassifier','MultinomialNB','LinearSVC'])
    
    # Deuxième combo box dans la deuxième colonne
    option2 = col2.selectbox('Choisissez une visualiation', ['Matrice confusion', 'Répartition par catégorie','Catégories proches'])
    
    # Afficher les options sélectionnées
    st.write(f"Vous avez choisi {option1} et {option2}")
    maintenant = datetime.now()
    print("t1 : ",maintenant)
    st.write(f"Modèle {option1}")
    
    if option1 == 'SVC' :
        lr = ml.ML_SVC("Mon_Modele_SVC",process=False)
        #deb 3        
        st.markdown("""
            Le modèle **SVC** (Support Vector Classifier) est un algorithme de classification supervisée basé sur le **Support Vector Machine (SVM)**.  
            il cherche à trouver une frontière optimale entre différentes classes dans un espace de caractéristique.  
            Le SVC trouve un hyperplan qui sépare les données de différentes classes avec la plus grande marge possible.  
            Cette marge correspond à la distance entre le plus proche point de chaque classe et l'yperplan.""")    
        #fin 3  
    elif   option1 == 'LogisticRegression' :  
        lr = ml.ML_LogisticRegression("LogisticRegression",process=False)
        #deb 3        
        st.markdown("""
            La **régression logistique multinomiale** est une méthode linéaire qui cherche à établir une relation entre les variables d'entrée (caractéristiques)
            et la probabilité d'appartenir à une classe.
            Le modèle est basé sur la fonction *Softmax* qui garantit que la somme des probabilités pour toutes les classes est égale à 1.  
            Elle prédit la probalité qu'une observation appartienne à l'une des 27 classes.  
            La régression logistique est un modèle simple qui fonctionne bien pour les problèmes de classification où les classes peuvent être séparées linéairement.""")    
        #fin 3  
    elif   option1 == 'RandomForestClassifier' :  
        lr = ml.ML_RandomForest("RandomForestClassifier",process=False)
        #deb 3
        st.markdown("""
            Le **RandomForestClassifier** est un algorithme d'apprentissage supervisé basé sur une **forêt d'arbre de décision**.  
            Il fonctionne bien pour les problèmes multi-classes et est capable de gérer la complexité liée à la classification de 27 catégories.  
            Le Random Forest fournit également une estimation de l'importance des caractéristiques dans le modèle.  
            Le modèle est très performant est très performant pour les problèmes complexes et non linéaires.""")    
        #fin 3  
    elif   option1 == 'GradientBoosting' :  
        lr = ml.ML_GradientBoosting("GradientBoosting",process=False)  
        #deb 3
        st.markdown("""
            Le **GradientBoostingClassifier** est un algorithme d'apprentissage supervisé basé sur une technique d'ensemble appelée **boosting**.  
            Contrairement au RandomForestClassifier qui entraine des arbres de décision de manière indépendante, le GradientBoosting crée des arbres séquentiellemnt, chaque nouvel arbre cherchant à corriger les erreurs des arbres précédents.  
            Il est particulièrement puissant pour des données complexes et hétérogènes.""")    
        #fin 3  
    elif   option1 == 'XGBClassifier' :  
        lr = ml.ML_XGBClassifier("XGBClassifier",process=False) 
        #deb 3         
        st.markdown("""
            Comme le **GradientBoosting**, **XGBoost** entraine des arbres de décision séquentiellement.  
            Chaque nouvel arbre corrige les erreurs faites par les arbres précédents, en se concentrant sur les exemples mal classés.    
            Le **XGBClassifier** combine précision, rapidité et capacité à gérer des ensembles complexes et volumineux.""")  
        #fin 3          
    elif   option1 == 'DecisionTreeClassifier' :  
        lr = ml.ML_DecisionTreeClassifier("DecisionTreeClassifier",process=False)  
        #deb 3         
        st.markdown("""
            Le **DecisionTreeClassifier**, est un modèle basé sur *un seul arbre de décision* où les décisions sint prises à chaque noeud en fontion des caracteristiques de l'observation.  
            Chaque chemin dans l'arbre mène à une decision de classe.    
            L'arbre va diviser les données de façon itérative pour aboutir à 27 classes de sortie possibles, une pour chaque catégorie.  
            Le modèle est simple à comprendre et à mettre en oeuvre.""")  
        #fin 3       
    elif   option1 == 'MultinomialNB' :  
        lr = ml.ML_MultinomialNB("MultinomialNB",process=False)   
        #deb 3         
        st.markdown("""
            Le modèle **MultinomialNB (Multinomial Naive Bayes)**, est un modèle probabiliste basé sur *le théorème de Bayes*.  
            Il utilise les probabilités des caractéristiques conditionnellement aux classes pour prédire la classe d'une nouvelle donnée.    
            Il est particulièrement adapté aux *données discrètes*, comme les occurences de mots dans les documents.""")  
        #fin 3               
    elif   option1 == 'LinearSVC' :  
        lr = ml.ML_LinearSVC("LinearSVC",process=False)   
        #deb 3         
        st.markdown("""
            Le modèle **LinearSVC**, utilise un noyau linéaire.  
            Il est efficace lorsque les classes peuvent être séparées par une frontière linéaire.    
            LinearSVC est généralement plus rapide et plus efficace pour les grandes données que SVC.""")  
        #fin 3     
        
    maintenant = datetime.now()
   
    lr_mod = lr.load_modele()
    y_orig = lr.get_y_orig()
    y_pred = lr.get_y_pred()
   
   
    X_test =  ds.load_ndarray('X_test')
   
    st.markdown("""**Performance du modèle** :""")
    f1 = f1_score(y_orig.values, y_pred, average='weighted')
    st.write("F1 Score: ", f1)
    
    if option1 == 'SVC' :
        y_test =  ds.load_ndarray('y_test')
        accuracy = accuracy_score(y_test.values,y_pred)
        st.write("Accuracy: ", accuracy)
    else:    
        if option1 == 'XGBClassifier' :
            label_encoder = LabelEncoder()
            label_encoder = ds.load_ndarray('XGBClassifier_label_encoder')
            y_test_encoded = label_encoder.transform(y_orig)
            accuracy = lr_mod.score(X_test, y_test_encoded)
        else  :   
            accuracy = lr_mod.score(X_test, y_orig.values)
        st.write("Accuracy: ", accuracy)
                                                # Matrice confusion
    if option2 == 'Matrice confusion' :
        st.markdown("""**Catégories prédites par ordre d'importance** :""")
        st.write("Voici un échantillon de produits avec pour chaque code produit , les 3 meilleurs prédictions de codes produit trouvées par ce modèle (codes et pourcentages)")
        df_pred = lr.get_df_pred()
        df_pred_formatte = df_pred.applymap(format_number)
        
        st.dataframe( df_pred_formatte.head(27))
        maintenant = datetime.now()
        st.markdown("""**Rapport de classification** : """)
        acc_score,classif=ds.get_classification_report(y_orig.values, y_pred)
        st.markdown(f"```\n{classif}\n```")
        st.markdown("""**Matrice de confusion** : """)
        st_show_confusion_matrix(y_orig.values, y_pred)
        nomenclature=pd.read_csv(DOSSIER_CSV+'NOMENCLATURE.csv',header=0,encoding='utf-8',sep=';',index_col=0)
        catdict=nomenclature.to_dict()['definition']
        st.write("#### Groupe 10,2280,2403 et 2705")
        st.write("10 : ",catdict[10])
        st.write("2280 : ",catdict[2280])
        st.write("2403 : ",catdict[2403])
        st.write("2705 : ",catdict[2705])
        st.write("#### Groupe 40,50 et 2462")
        st.write("40 : ",catdict[40])
        st.write("50 : ",catdict[50])
        st.write("2462 : ",catdict[2462])
        st.write("#### Groupe 1280 et 1281")
        st.write("1280 : ",catdict[1280])
        st.write("1281 : ",catdict[1281])
        
        del lr
        del lr_mod, y_orig, y_pred, X_test, f1, y_test, accuracy
        del df_pred, df_pred_formatte, acc_score, classif, nomenclature, catdict
        
                                            # Répartition par catégorie
    elif option2 == 'Répartition par catégorie' :  
        
        st.markdown("""**Répartition par catégorie** : """)  
        st.write("Liste pour chaque catégorie de produit les cinq codes produits les plus pronostiqués par le modèle")

        df_cross = lr.get_df_cross()
        Lcat=lr.get_cat()
        catdict = lr.get_catdict()
        for c in Lcat:
            st.markdown(f"<p style='color: red;'>{c} ------ {catdict[c]}</p>", unsafe_allow_html=True)   
            s=df_cross.loc[c].sort_values(ascending=False)[:5]
            for index, value in s.items(): 
                st.write(f"  : {index}  : {np.round(value*100,2)} % , {catdict[index]}")   
        del lr  
        del lr_mod, y_orig, y_pred, X_test, f1, y_test, accuracy       
        del df_cross, Lcat, catdict, s    
                                #  Catégories proches
    elif option2 == 'Catégories proches' : 
        if option1 == 'SVC' :
            st.markdown("""**Catégories proches** : """)  
            st.write("Certaines catégories sont mal différenciées. ")
            st.write("Les 20 mots les plus fréquents par groupe de catégories")
            col1, col2, col3 = st.columns(3)
            X_test = ds.load_ndarray("X_test")
            y_test = ds.load_ndarray("y_test")
            df_test = pd.concat([X_test,y_test],axis=1)
        
            with col1:
                selected_categories = [10, 2705, 2280, 2403]
                filtered_df = df_test[df_test['prdtypecode'].isin(selected_categories)]
            
                all_text = ' '.join(filtered_df['phrases'])
            
            
                word_count = Counter(all_text.split())
                occurrences_triees = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:40]
            
                Dic_words={}
                for item in occurrences_triees:
                    Dic_words[item[0]]=item[1] 
                df_words=pd.DataFrame.from_dict(Dic_words, orient='index',columns=['Total'])
                df_total_livres=df_words
                for c in [10,2705,2280,2403]:
                    filtered_df = df_test[df_test['prdtypecode']==c]
                    all_text = ' '.join(filtered_df['phrases'])
              
                    word_count = Counter(all_text.split())
                    occurrences_triees = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:40]
                    Dic_words={}
                    for item in occurrences_triees:
                        Dic_words[item[0]]=item[1]
                    df_words_X=pd.DataFrame.from_dict(Dic_words, orient='index',columns=[str(c)])    
                    df_total_livres=df_total_livres.join(df_words_X)
               
                fig1, ax = plt.subplots(figsize=(5,5))    
                cax = ax.matshow(df_total_livres.iloc[0:20], cmap = 'coolwarm')
                plt.xticks(range(5),('total','10','2705','2280','2403'),rotation=45)
                plt.yticks(range(20),df_total_livres.iloc[0:20].index,rotation=0)
                st.pyplot(fig1)
            with col2:     
                selected_categories = [40,50,2462]
                filtered_df = df_test[df_test['prdtypecode'].isin(selected_categories)]
            
                all_text = ' '.join(filtered_df['phrases'])
            
            
                word_count = Counter(all_text.split())
                occurrences_triees = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:40]
            
                Dic_words={}
                for item in occurrences_triees:
                    Dic_words[item[0]]=item[1] 
                df_words=pd.DataFrame.from_dict(Dic_words, orient='index',columns=['Total'])
                df_total_livres=df_words    
                for c in [40,50,2462]:
                    filtered_df = df_test[df_test['prdtypecode']==c]
                    all_text = ' '.join(filtered_df['phrases'])
               
                    word_count = Counter(all_text.split())
                    occurrences_triees = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:40]
                    Dic_words={}
                    for item in occurrences_triees:
                        Dic_words[item[0]]=item[1]
                    df_words_X=pd.DataFrame.from_dict(Dic_words, orient='index',columns=[str(c)])    
                    df_total_livres=df_total_livres.join(df_words_X) 
            
                fig2, ax = plt.subplots(figsize=(5,4.9))    
                cax = ax.matshow(df_total_livres.iloc[0:20], cmap = 'coolwarm')
                plt.xticks(range(4),('total','40','50','2462'),rotation=45)
                plt.yticks(range(20),df_total_livres.iloc[0:20].index,rotation=0)
                st.pyplot(fig2) 
                 
            with col3:   
                selected_categories = [1280,1281]
                filtered_df = df_test[df_test['prdtypecode'].isin(selected_categories)]
            
                all_text = ' '.join(filtered_df['phrases'])
            
            
                word_count = Counter(all_text.split())
                occurrences_triees = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:40]
            
                Dic_words={}
                for item in occurrences_triees:
                    Dic_words[item[0]]=item[1] 
                df_words=pd.DataFrame.from_dict(Dic_words, orient='index',columns=['Total'])
                df_total_livres=df_words     
                for c in [1280,1281]:
                    filtered_df = df_test[df_test['prdtypecode']==c]
                    all_text = ' '.join(filtered_df['phrases'])
               
                    word_count = Counter(all_text.split())
                    occurrences_triees = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:40]
                    Dic_words={}
                    for item in occurrences_triees:
                        Dic_words[item[0]]=item[1]
                    df_words_X=pd.DataFrame.from_dict(Dic_words, orient='index',columns=[str(c)])    
                    df_total_livres=df_total_livres.join(df_words_X) 
                fig3, ax = plt.subplots(figsize=(5.5,3.5))    
                cax = ax.matshow(df_total_livres.iloc[0:20], cmap = 'coolwarm')
                plt.xticks(range(3),('total','1280','1281'),rotation=45)
                plt.yticks(range(20),df_total_livres.iloc[0:20].index,rotation=0)
                st.pyplot(fig3) 
            nomenclature=pd.read_csv(DOSSIER_CSV+'NOMENCLATURE.csv',header=0,encoding='utf-8',sep=';',index_col=0)
            catdict=nomenclature.to_dict()['definition']
            st.write("#### Groupe 10,2280,2403 et 2705")
            st.write("10 : ",catdict[10])
            st.write("2280 : ",catdict[2280])
            st.write("2403 : ",catdict[2403])
            st.write("2705 : ",catdict[2705])
            st.write("#### Groupe 40,50 et 2462")
            st.write("40 : ",catdict[40])
            st.write("50 : ",catdict[50])
            st.write("2462 : ",catdict[2462])
            st.write("#### Groupe 1280 et 1281")
            st.write("1280 : ",catdict[1280])
            st.write("1281 : ",catdict[1281])
        else:
            st.write("Cette visualisation n'est disponible que pour le modèle SVC .")
        del lr
        del col1, col2, col3, df_test, selected_categories, filtered_df, all_text
        del word_count, occurrences_triees, Dic_words, df_words_X, df_total_livres
        del ax, cax, nomenclature, catdict
        fig1 = None
        fig2 = None
        fig3 = None
        gc.collect()