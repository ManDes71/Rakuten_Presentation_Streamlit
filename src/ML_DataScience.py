import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.metrics import f1_score


import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pickle
import re
from datetime import datetime



import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from  src import Bibli_DataScience_3 as ds




class DS_ML(ds.DS_Model):
    
     def __init__(self, nom_modele,process=True):
     
        super().__init__(nom_modele)
            
        
        self.__nom_modele = nom_modele
        
        
        self.__df_pred = pd.DataFrame()
        self.__df_cross = pd.DataFrame()
        self.__df_feats = pd.DataFrame()
        self.__df_target = pd.DataFrame()
        self.__df_prob = pd.DataFrame()
        self.__y_orig =[]
        self.__y_pred = []
        self.__y_prob = []
        self.__X = np.array([])
        self.__y = np.array([])
        self.__report_ID = "SVM1"
        self.__report_MODELE = nom_modele
        self.__report_LIBELLE = "SVM AVEC TDIDF"
        self.recuperer_dataframes()
     
        nltk.download('punkt')
        
        if process == True:
            print("preprocessing ...")
            self.traiter_dataframes()
      
     def get_df_feats(self):
        return self.__df_feats
        
     def get_df_target(self):
        return self.__df_target

     def get_df_pred(self):
        return self.__df_pred

     def set_df_pred(self,pred):
        self.__df_pred = pred
        
     def get_df_cross(self):
        return self.__df_cross

     def set_df_cross(self,cross):
        self.__df_cross = cross   
        
     def get_df_prob(self):
        return self.__df_prob

     def set_df_prob(self,prob):
        self.__df_prob = prob  

     def get_labelencoder(self):
        return self.__label_encoder

     def set_labelencoder(self,label):
        self.__label_encoder = label          
        
     def set_y_orig(self,orig):
        self.__y_orig = orig 
        
     def set_y_prob(self,prob):
        self.__y_prob = prob    

     def set_y_pred(self,pred):
        self.__y_pred = pred        
        
     def get_y_orig(self) :
        return self.__y_orig
        
     def get_y_pred(self) :
        return self.__y_pred
        
     def get_y_prob(self) :
        return self.__y_prob   
        
     def get_REPORT_ID(self) :
        return self.__report_ID
     def set_REPORT_ID(self,id):
        self.__report_ID = id           
     def get_REPORT_MODELE(self) :
        return self.__report_MODELE
     def set_REPORT_MODELE(self,modele):
        self.__report_MODELE = modele          
     def get_REPORT_LIBELLE(self) :
        return self.__report_LIBELLE  
     def set_REPORT_LIBELLE(self,libelle):
        self.__report_LIBELLE = libelle      
        
     def clean_sentence(self,sentence,langue):
        if langue == 'en':
            SW = stopwords.words('english')
        elif langue == 'fr':
            SW = stopwords.words('french')
        elif langue == 'de':
            SW = stopwords.words('german')
        elif langue == 'ca':
            SW = stopwords.words('french')
        elif langue == 'nl':
            SW = stopwords.words('dutch')
        elif langue == 'it':
            SW = stopwords.words('italian')
        elif langue == 'es':
            SW = stopwords.words('spanish')
        else:
            SW = stopwords.words('french')
         # Pour chaque mot de la phrase (dans l'ordre inverse)
        for i, word in reversed(list(enumerate(sentence))):
                # Si le mot est un stopword
                if word in SW :
                    # On l'enlève de l'artikle
                    sentence.pop(i)
        return sentence    
       
     def dictionarize(self,article,langue):
        r = re.compile(r"[a-zA-Z0-9âéè°]{2,}") 
        #dico ={}
        ListMots =[]
        ## Etape 1:  Découper l'article en une liste de phrase
        #article = sent_tokenize(article)
        #sent = re.sub(liste_caracteres_speciaux, ' ', article)
        sent = ' '.join(r.findall(article))
                
        ## Etape 3: Découper chaque phrase en une liste de mots
        sent = word_tokenize(sent)
                        
        ## Etape 4: Mettre tous les mots de la phrase en minuscule
        sent_lower = [word.lower() for word in sent]
                
        ## Etape 5: Retirer les stopwords de chaque liste.
        sent_clean = self.clean_sentence(sent_lower,langue)
              
            
        #dico[i]=sent_clean
        ListMots.append(sent_clean)
        
        resultat = []

        for sous_liste in ListMots:
            resultat.extend(sous_liste)
            
        return resultat
        
     def decomposition(self,article,langue):
        stop_words = self.get_stopwordFR().to_numpy()
        #print(type(stop_words))
        #print(stopwordFR[:1])
        artikle = self.dictionarize(article,langue)
        artikle = [mot for mot in artikle if mot not in stop_words]
        
        return artikle       
        
     def recuperer_dataframes(self):
        df = self.get_DF()
        self.__df_feats = df[['designation','description','productid','imageid','PAYS_LANGUE']].copy()
        self.__df_target = df['prdtypecode'].copy()
        
     def traiter_dataframes(self):
        DESCRIP = []
        df  = self.__df_feats
        for design, descrip in zip( df['designation'],  df['description']):
            partie_design = design if type(design) == str else ''
            partie_descrip = descrip if type(descrip) == str else ''
            s = (partie_design + ' ' + partie_descrip) if len(partie_descrip) > 0 else partie_design
            DESCRIP.append(s)
        
        #self.__df_feats['designation'] = pd.Series(DESCRIP)
        #print(pd.Series(DESCRIP).values)
        self.__df_feats.loc[:,'designation'] = pd.Series(DESCRIP)
        self.__df_feats.loc[:,'phrases'] = self.__df_feats.apply(lambda x : self.decomposition(str(x.designation),str(x.PAYS_LANGUE)),axis=1)
        df2=self.get_DF()
        df2['phrases'] = self.__df_feats['phrases']
     
     def traiter_phrases(self,design,descrip):
        DESCRIP = []
        partie_design = design if type(design) == str else ''
        partie_descrip = descrip if type(descrip) == str else ''
        s = (partie_design + ' ' + partie_descrip) if len(partie_descrip) > 0 else partie_design
        langue=ds.detection_langue(s)
        print("langue = ",langue)
        print("decomposition",self.decomposition(str(s),str(langue)))
        print("type decomposition",type(self.decomposition(str(s),str(langue))))
        print("prepoccessing_x",self.preprossessing_X(self.decomposition(str(s),str(langue))))
             
        return self.preprossessing_X(self.decomposition(str(s),str(langue)))
        
        
     def preprossessing_X(self,text):
        # Utilisez PunktSentenceTokenizer pour tokeniser les phrases
        return ' '.join(text)
       
     #     Train       : "None" , "Save"  , "Load"  =>   "Save" : on enregistre les données d'entrainement
     #                                              =>   "Load" : on charge les données d'entrainement                 
     def Train_Test_Split_(self,train_size=0.8, random_state=1234,  RandUnderSampl = True,  RandomOverSampl = True,fic="None"):    
        
        
        if fic == "Load" :
            X_train = ds.load_ndarray('X_train')
            X_test = ds.load_ndarray('X_test')
            y_train_avant = ds.load_ndarray('y_train')
            y_test_avant = ds.load_ndarray('y_test')
            
            return X_train, X_test, y_train_avant, y_test_avant
        
        X_train_avant, X_test_avant, y_train_avant, y_test_avant = super().Train_Test_Split_(train_size, random_state)
        
        

        #X_train = X_train_avant['designation'].apply(self.preprossessing_X)
        #X_test = X_test_avant['designation'].apply(self.preprossessing_X)
        
        X_train = X_train_avant['phrases'].apply(self.preprossessing_X)
        X_test = X_test_avant['phrases'].apply(self.preprossessing_X)
        
        if fic == "Save" :
            ds.save_ndarray(X_train,'X_train')
            ds.save_ndarray(X_test,'X_test')
            ds.save_ndarray( y_train_avant,'y_train')
            ds.save_ndarray(y_test_avant,'y_test')
        
        return X_train, X_test, y_train_avant, y_test_avant
        
     def Calculer_df_prob(self,y_pred,y_prob):
        # =========================================================
        #    probabilite par enregistrement (16984)
        # =========================================================
        # Transformez y_prob et y_test en DataFrames
        y_prob_df = pd.DataFrame(y_prob, columns=[f"prob_class_{i}" for i in range(y_prob.shape[1])])
        # Reset de l'index pour qu'il corresponde à celui de y_prob_df
        y_pred_series = pd.Series(y_pred, name='predicted_class')
        # Concaténez les deux DataFrames le long de l'axe des colonnes
        combined_df = pd.concat([y_pred_series, y_prob_df], axis=1)

        
        # Obtenir la colonne avec la probabilité maximale
        max_prob_col = combined_df.iloc[:, 1:].max(axis=1)

        # Ajouter la colonne 'ProbMax' au DataFrame
        combined_df['ProbMax'] = max_prob_col

        # Créer un nouveau DataFrame avec les colonnes 'predicted_class' et 'ProbMax'
        Prob1 = combined_df[['predicted_class', 'ProbMax']]    
        
        return Prob1
      #///////////////////
      #     savefics    : True , False     ->  sauvegarde des du modele
      #     Train       : "None" , "Save"  , "Load"  =>   "Save" : on enregistre les données d'entrainement
      #                                              =>   "Load" : on charge les données d'entrainement                  
     def fit_modele(self,savefics=False,Train="None"):
        pass
     
        
 # ************************************************************************************************  

 
class ML_SVC(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        
        self.set_REPORT_ID("SVM1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("SVM AVEC TDIDF")
       
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_clf = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', SVC(C=10,class_weight=None,kernel='rbf',probability=True)),
            ])
        start_time = datetime.now()
        print("L'heure au début de l'entraînement était : ", start_time)    
         
        # Entraînez le modèle
        text_clf.fit(X_train, y_train)

        end_time = datetime.now()
        print("L'heure à la fin de l'entraînement était : ", end_time)
        if Train == 'Save' :
            ds.joblib_dump(text_clf,self.__nom_modele+'_dump')

        # Testez le modèle sur l'ensemble de test
        y_pred = text_clf.predict(X_test)
        ds.save_ndarray(y_pred,self.__nom_modele+'_pred') 
        y_prob = text_clf.predict_proba(X_test)
        ds.save_ndarray(y_prob,self.__nom_modele+'_prob') 
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_clf.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)

      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
        
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
        
        #if savefics :
        #    ds.save_ndarray(train_acc,self.__nom_modele+'_accuracy')
        #    ds.save_ndarray(val_acc,self.__nom_modele+'_val_accuracy')
        #    ds.save_ndarray(tloss,self.__nom_modele+'_loss')
        #    ds.save_ndarray(tvalloss,self.__nom_modele+'_val_loss')
        #    ds.save_ndarray(y_test_original2,self.__nom_modele+'_y_orig')
        #    ds.save_ndarray(test_pred_orinal2,self.__nom_modele+'_y_pred')
        #    ds.save_dataframe(self.__df_pred,self.__nom_modele+'_df_predict')
        
        duration = end_time - start_time
        print("La durée de l'entraînement était : ", duration)
        
     def predire_phrases(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER(),self.__nom_modele+'_dump')
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         y_pred = text_clf.predict(pd.Series(X_test))
         return(y_pred)
     
     def proba_phrases(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER(),self.__nom_modele+'_dump')
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         y_pred = text_clf.predict_proba(pd.Series(X_test))
         return(y_pred)    
     
     def load_modele(self,Train="None"):   
         print('load_modele')
         print(self.__nom_modele+'_dump')
         
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         
        
         X_test = ds.load_ndarray('X_test')
         y_test = ds.load_ndarray('y_test')
         
   
         start_time = datetime.now()
          
         if Train=="True":
            # Testez le modèle sur l'ensemble de test
            y_pred = text_clf.predict(X_test)
            y_prob = text_clf.predict_proba(X_test)
         else :    
            y_pred = ds.load_ndarray(self.__nom_modele+'_pred')
            y_prob= ds.load_ndarray(self.__nom_modele+'_prob')   
         #f1 = f1_score(y_test, y_pred, average='weighted')
         #print("F1 Score: ", f1)
         #accuracy = text_clf.score(X_test, y_test)
         #print("Accuracy: ", accuracy)
       
         
         self.set_y_orig(y_test)
         self.set_y_pred(y_pred)
         self.set_y_prob(y_prob)
         
         df_prob = self.Calculer_df_prob(y_pred,y_prob)
         self.set_df_prob(df_prob)
         
         ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
        

       
         df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
         self.set_df_pred(df_pred)
         
         df_cross = ds.get_df_crosstab(y_test, y_pred)
         self.set_df_cross(df_cross)
         
         return text_clf   
         

# ************************************************************************************************  
class ML_LinearSVCFromModel(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        
        self.set_REPORT_ID("SVM2")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("LinearSVC AVEC TDIDF")
       
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_lsvm = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', SelectFromModel(LinearSVC(penalty="l2",  C=1,
                                            tol=1e-5, max_iter=3000),max_features=3000)),
            ])
        start_time = datetime.now()
        print("L'heure au début de l'entraînement était : ", start_time)    
       
            
        # Entraînez le modèle
        text_lsvm.fit(X_train, y_train)

        end_time = datetime.now()
        print("L'heure à la fin de l'entraînement était : ", end_time)
        print("nom modele : ",self.__nom_modele)
        
         
        if Train == 'Save' :
            ds.joblib_dump(text_lsvm,self.__nom_modele+'_dump') 
        
        X_train_SVC = text_lsvm.fit_transform(X_train,y_train)
        X_test_SVC = text_lsvm.transform(X_test)

        X_train_SVC = X_train_SVC.todense()
        X_test_SVC = X_test_SVC.todense()
        
        ds.save_ndarray(X_train_SVC,self.__nom_modele+'_CONCAT2_X_train')
        ds.save_ndarray(X_test_SVC,self.__nom_modele+'_CONCAT2_X_test')
        ds.save_ndarray(y_train,self.__nom_modele+'_CONCAT2_y_train')
        ds.save_ndarray(y_test,self.__nom_modele+'_CONCAT2_y_test')
        
        

        # Testez le modèle sur l'ensemble de test
        #y_pred = text_lsvm.predict(X_test)
        #y_prob = text_lsvm.predict_proba(X_test)
        
        #f1 = f1_score(y_test, y_pred, average='weighted')
        #print("F1 Score: ", f1)
        #accuracy = text_lsvm.score(X_test, y_test)
        #print("Accuracy: ", accuracy)

        #self.set_y_orig(y_test)
        #self.set_y_pred(y_pred)

      
        #df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        #self.set_df_pred(df_pred)
        
        #df_prob = self.Calculer_df_prob(y_pred,y_prob)
        #self.set_df_prob(df_prob)
        
        duration = end_time - start_time
        print("La durée de l'entraînement était : ", duration)

     def preparer_concatenation(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER(),self.__nom_modele+'_dump')
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         X_test_SVC = text_clf.transform(pd.Series(X_test))
         X_test_SVC = X_test_SVC.todense()
         return(X_test_SVC) 
     
    
        
     def load_modele(self,Train="None"):   
         print('load_modele')
         print(self.__nom_modele+'_dump')
         
         text_lsvm = ds.joblib_load(self.__nom_modele+'_dump')
         print("L'heure au début de l'entraînement était : ", start_time)   
         X_test = ds.load_ndarray('X_test')
         y_test = ds.load_ndarray('y_test')
         
          
             
         print(X_test[:5])
         
         
         return text_lsvm     
# ************************************************************************************************  
class ML_LinearSVC(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        
        self.set_REPORT_ID("SVM2")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("LinearSVC AVEC TDIDF")
       
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_lsvm = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', LinearSVC(penalty="l2",  C=1,tol=1e-5, max_iter=3000)),
            ])
        start_time = datetime.now()
        print("L'heure au début de l'entraînement était : ", start_time)    
        
        
        # Entraînez le modèle
        text_lsvm.fit(X_train, y_train)

        end_time = datetime.now()
        print("L'heure à la fin de l'entraînement était : ", end_time)
        print("nom modele : ",self.__nom_modele)
        
        
           
        if Train == 'Save' :
            ds.joblib_dump(text_lsvm,self.__nom_modele+'_dump')
        
        

        # Testez le modèle sur l'ensemble de test
        y_pred = text_lsvm.predict(X_test)
      
        #y_prob = text_lsvm.predict_proba(X_test)
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_lsvm.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)

      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
        
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
        
        #df_prob = self.Calculer_df_prob(y_pred,y_prob)
        #self.set_df_prob(df_prob)
        
        duration = end_time - start_time
        print("La durée de l'entraînement était : ", duration)       
        
     def load_modele(self,Train="None"):   
        print('load_modele',self.__nom_modele+'_dump')
        text_lsvm = ds.joblib_load(self.__nom_modele+'_dump')
       
        X_test = ds.load_ndarray('X_test')
        y_test = ds.load_ndarray('y_test')
      
       
          
        print(X_test[:5])
      
      # Testez le modèle sur l'ensemble de test
        y_pred = text_lsvm.predict(X_test)
        #y_prob = text_lsvm.predict_proba(X_test)
      
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_lsvm.score(X_test, y_test)
        print("Accuracy: ", accuracy)
      
        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
       # self.set_y_prob(y_prob)
      
      
       # df_prob = self.Calculer_df_prob(y_pred,y_prob)
       # self.set_df_prob(df_prob)
      
       # ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')

    
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
      
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
      
        return text_lsvm                  

# ************************************************************************************************          
class ML_LogisticRegression(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("LR1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("LogisticRegression AVEC TDIDF")
        
     
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_lr = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', LogisticRegression(C=1,fit_intercept=True,solver='liblinear',penalty='l2',max_iter= 200)),
            ])
            
            
            
        # Entraînez le modèle
        text_lr.fit(X_train, y_train)

        # Testez le modèle sur l'ensemble de test
        y_pred = text_lr.predict(X_test)
        y_prob = text_lr.predict_proba(X_test)
        
        if Train == 'Save' :
            ds.joblib_dump(text_lr,self.__nom_modele+'_dump')
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_lr.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
       

      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
        
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
        
     def load_modele(self,Train="None"):   
         print('load_modele')
         text_lr = ds.joblib_load(self.__nom_modele+'_dump')
         X_test = ds.load_ndarray('X_test')
         y_test = ds.load_ndarray('y_test')
         
          
             
         print(X_test[:5])
         
         # Testez le modèle sur l'ensemble de test
         y_pred = text_lr.predict(X_test)
         y_prob = text_lr.predict_proba(X_test)

         f1 = f1_score(y_test.values, y_pred, average='weighted')
         print("F1 Score: ", f1)
         accuracy = text_lr.score(X_test, y_test.values)
         print("Accuracy: ", accuracy)
         
         self.set_y_orig(y_test)
         self.set_y_pred(y_pred)
         self.set_y_prob(y_prob)
         
         df_prob = self.Calculer_df_prob(y_pred,y_prob)
         self.set_df_prob(df_prob)
         
         ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
        

       
         df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
         self.set_df_pred(df_pred)
         
         df_cross = ds.get_df_crosstab(y_test, y_pred)
         self.set_df_cross(df_cross)
         
         return text_lr
     
     def predire_phrases(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER(),self.__nom_modele+'_dump')
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         y_pred = text_clf.predict(pd.Series(X_test))
         return(y_pred)   
     
     def proba_phrases(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER(),self.__nom_modele+'_dump')
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         y_pred = text_clf.predict_proba(pd.Series(X_test))
         return(y_pred)    
         
         
# ************************************************************************************************   
 
class ML_RandomForest(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("FOREST1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("RandomForest AVEC TDIDF")
        
     def fit_modele(self,savefics=False,Train="None"):
      
        #reg="[a-zA-Zé°]{2,}"
        #stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_forest = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', RandomForestClassifier(n_jobs=-1,random_state=321)),
            ])
        
        #   DecisionTreeClassifier()     
        #Meilleurs paramètres : {'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 5} 
            
        # Entraînez le modèle
        text_forest.fit(X_train, y_train)
        
        
        if Train == 'Save' :
            ds.joblib_dump(text_forest,self.__nom_modele+'_dump')

        # Testez le modèle sur l'ensemble de test
        y_pred = text_forest.predict(X_test)
        y_prob = text_forest.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_forest.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
        
      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
        
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
        
     def load_modele(self,Train="None"):   
         print('load_modele')
         text_forest = ds.joblib_load(self.__nom_modele+'_dump')
         X_test = ds.load_ndarray('X_test')
         y_test = ds.load_ndarray('y_test')
         
          
             
         print(X_test[:5])
         
         # Testez le modèle sur l'ensemble de test
         y_pred = text_forest.predict(X_test)
         y_prob = text_forest.predict_proba(X_test)
         
         f1 = f1_score(y_test, y_pred, average='weighted')
         print("F1 Score: ", f1)
         accuracy = text_forest.score(X_test, y_test)
         print("Accuracy: ", accuracy)
         
         self.set_y_orig(y_test)
         self.set_y_pred(y_pred)
         self.set_y_prob(y_prob)
         
         df_prob = self.Calculer_df_prob(y_pred,y_prob)
         self.set_df_prob(df_prob)
         
         ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
        

       
         df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
         self.set_df_pred(df_pred)
         
         df_cross = ds.get_df_crosstab(y_test, y_pred)
         self.set_df_cross(df_cross)
         
         return text_forest   
     
     def predire_phrases(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER(),self.__nom_modele+'_dump')
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         y_pred = text_clf.predict(pd.Series(X_test))
         return(y_pred)   
     
     def proba_phrases(self,designation,description): 
         X_test=self.traiter_phrases(designation,description)
         print(ds.get_RACINE_DOSSIER())
         text_clf = ds.joblib_load(self.__nom_modele+'_dump')
         y_pred = text_clf.predict_proba(pd.Series(X_test))
         return(y_pred)    
        
# ************************************************************************************************     
class ML_GradientBoosting(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("GRBOOST1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("GradientBoosting AVEC TDIDF")
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_gboost = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', GradientBoostingClassifier(criterion= 'squared_error', learning_rate= 0.1, \
                    loss= 'log_loss', max_depth = 18, max_features = 'sqrt')),
            ])
            
            
            
        # Entraînez le modèle
        text_gboost.fit(X_train, y_train)
        
        if Train == 'Save' :
            ds.joblib_dump(text_gboost,self.__nom_modele+'_dump')

        # Testez le modèle sur l'ensemble de test
        y_pred = text_gboost.predict(X_test)
        y_prob = text_gboost.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_gboost.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')

      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)  
        
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
        
     def load_modele(self,Train="None"):   
        print('load_modele',self.__nom_modele+'_dump')
        text_gboost = ds.joblib_load(self.__nom_modele+'_dump')
       
        X_test = ds.load_ndarray('X_test')
        y_test = ds.load_ndarray('y_test')
      
       
          
        print(X_test[:5])
      
      # Testez le modèle sur l'ensemble de test
        y_pred = text_gboost.predict(X_test)
        y_prob = text_gboost.predict_proba(X_test)
      
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_gboost.score(X_test, y_test)
        print("Accuracy: ", accuracy)
      
        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
      
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
      
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
     

    
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
      
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
      
        return text_gboost         
 # ************************************************************************************************     
class ML_XGBClassifier(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("xgboost1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("XGBClassifier AVEC TDIDF")
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        
        label_encoder = LabelEncoder()
        
        
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
       
        text_xgboost = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', xgb.XGBClassifier(learning_rate=0.1,n_estimators=500,max_depth=10)),
            ])
            
            
            
        # Entraînez le modèle
        text_xgboost.fit(X_train, y_train_encoded)
        
        if Train == 'Save' :
            ds.joblib_dump(text_xgboost,self.__nom_modele+'_dump')
            ds.save_ndarray(label_encoder,self.__nom_modele+'_label_encoder')

        # Testez le modèle sur l'ensemble de test
        y_pred_classes = text_xgboost.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_classes)
        y_prob = text_xgboost.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_xgboost.score(X_test, y_test_encoded)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')

      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)      

        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)   
     
     def load_modele(self,Train="None"):   
        print('load_modele',self.__nom_modele+'_dump')
        text_XGB = ds.joblib_load(self.__nom_modele+'_dump')
       
        X_test = ds.load_ndarray('X_test')
        y_test = ds.load_ndarray('y_test')

        label_encoder = LabelEncoder()
        label_encoder = ds.load_ndarray('XGBClassifier_label_encoder')
      
        y_test_encoded = label_encoder.transform(y_test)
          
        print(X_test[:5])
      
      # Testez le modèle sur l'ensemble de test
        y_pred_classes = text_XGB.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_classes)
        y_prob = text_XGB.predict_proba(X_test)
      
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_XGB.score(X_test, y_test_encoded)
        print("Accuracy: ", accuracy)
      
        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
      
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
      
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
     

    
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
      
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
      
        return text_XGB              
            
# ************************************************************************************************     
class ML_MultinomialNB(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("MULTINB1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("MultinomialNB AVEC TDIDF")
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_NB = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', MultinomialNB(alpha = 0.1, fit_prior = False)),
            ])
            
            
            
        # Entraînez le modèle
        text_NB.fit(X_train, y_train)
        
        if Train == 'Save' :
            ds.joblib_dump(text_NB,self.__nom_modele+'_dump')

        # Testez le modèle sur l'ensemble de test
        y_pred = text_NB.predict(X_test)
        y_prob = text_NB.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_NB.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        

        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred) 

        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)    
        
     def load_modele(self,Train="None"):   
        print('load_modele',self.__nom_modele+'_dump')
        text_MultiNb = ds.joblib_load(self.__nom_modele+'_dump')
       
        X_test = ds.load_ndarray('X_test')
        y_test = ds.load_ndarray('y_test')
      
       
          
        print(X_test[:5])
      
      # Testez le modèle sur l'ensemble de test
        y_pred = text_MultiNb.predict(X_test)
        y_prob = text_MultiNb.predict_proba(X_test)
      
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_MultiNb.score(X_test, y_test)
        print("Accuracy: ", accuracy)
      
        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
      
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
      
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
     

    
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
      
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
      
        return text_MultiNb            
# ************************************************************************************************     
class ML_DecisionTreeClassifier(DS_ML):     

     def __init__(self, nom_modele,process=True):
        super().__init__(nom_modele,process=process)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("DTCL1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("DecisionTreeClassifier AVEC TDIDF")
        
     def fit_modele(self,savefics=False,Train="None"):
      
        reg="[a-zA-Zé°]{2,}"
        stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        text_DTCL = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', DecisionTreeClassifier(class_weight=None,criterion='gini',max_depth=None,min_samples_split= 3)),
            ])
         #ecisionTreeClassifier(criterion ='entropy', max_depth=60, random_state=123)   
            
            
        # Entraînez le modèle
        text_DTCL.fit(X_train, y_train)
        
        if Train == 'Save' :
            ds.joblib_dump(text_DTCL,self.__nom_modele+'_dump')

        # Testez le modèle sur l'ensemble de test
        y_pred = text_DTCL.predict(X_test)
        y_prob = text_DTCL.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_DTCL.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')

      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)   

        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)    
        
     def load_modele(self,Train="None"):   
        print('load_modele',self.__nom_modele+'_dump')
        text_DecTree = ds.joblib_load(self.__nom_modele+'_dump')
       
        X_test = ds.load_ndarray('X_test')
        y_test = ds.load_ndarray('y_test')
      
       
          
        print(X_test[:5])
      
      # Testez le modèle sur l'ensemble de test
        y_pred = text_DecTree.predict(X_test)
        y_prob = text_DecTree.predict_proba(X_test)
      
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = text_DecTree.score(X_test, y_test)
        print("Accuracy: ", accuracy)
      
        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
      
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
      
        ds.save_dataframe(df_prob,self.__nom_modele+'_prob.csv')
     

    
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
      
        df_cross = ds.get_df_crosstab(y_test, y_pred)
        self.set_df_cross(df_cross)
      
        return text_DecTree             
# ************************************************************************************************  
class ML_Grid_RandomForest(DS_ML):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("GRIDFOREST1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("GRID SUR RandomForest")
        
     def fit_modele(self,savefics=False,Train="None"):
      
        #reg="[a-zA-Zé°]{2,}"
        #stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
        param_grid = {
            'clf__n_estimators': [180,200,220],
            'clf__max_features': [ 'sqrt'],
            'clf__max_depth': [50,60,70],
            'clf__criterion': [ 'entropy']
           #'tfidf__max_df': [ 0.3,0.4,0.5], # Ignore les termes qui apparaissent dans plus de max_df % des documents. Utilisé pour éliminer les termes trop fréquents qui sont moins informatifs.
           # 'tfidf__min_df': [18,30,50] #  Ignore les termes qui apparaissent dans moins de min_df documents. Utilisé pour éliminer les termes trop rares.
}



   
        text_forest = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', RandomForestClassifier(n_jobs=-1, random_state=321))
        ])
       
        CV_text_forest = GridSearchCV(estimator=text_forest, param_grid=param_grid, cv=5)
        
        start_time = datetime.now()
        print("L'heure au début de l'entraînement était : ", start_time)    
 
        CV_text_forest.fit(X_train, y_train)
        
        print("Les meilleurs paramètres sont :")
        print(CV_text_forest.best_params_)
        
        print("Le meilleur score est :")
        print(CV_text_forest.best_score_)
        
        end_time = datetime.now()
        print("L'heure à la fin de l'entraînement était : ", end_time)
         
        y_pred = CV_text_forest.predict(X_test)
        y_prob = CV_text_forest.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = CV_text_forest.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
        
        duration = end_time - start_time
        print("La durée de l'entraînement était : ", duration)
        
 # ************************************************************************************************  
class ML_Grid_MultinomialNB(DS_ML):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("GRIDFOREST1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("GRID SUR RandomForest")
        
     def fit_modele(self,savefics=False,Train="Load"):
      
        #reg="[a-zA-Zé°]{2,}"
        #stopwordFR = self.get_stopwordFR()
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
        param_grid_NB = {
        'clf__alpha': [1e-3, 1e-2,0.1, 1, 10],
        'clf__fit_prior': [True, False] }

    
        text_NB = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, max_df=0.8, min_df=2)),
        ('clf', MultinomialNB())
    ])
    
        CV_text_NB = GridSearchCV(estimator=text_NB, param_grid=param_grid_NB, cv=5)
        
        start_time = datetime.now()
        print("L'heure au début de l'entraînement était : ", start_time)    
 
        CV_text_NB.fit(X_train, y_train)
        
        print("Les meilleurs paramètres sont :")
        print(CV_text_NB.best_params_)
        
        print("Le meilleur score est :")
        print(CV_text_NB.best_score_)
        
        end_time = datetime.now()
        print("L'heure à la fin de l'entraînement était : ", end_time)
         
        y_pred = CV_text_NB.predict(X_test)
        y_prob = CV_text_NB.predict_proba(X_test)
        
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 Score: ", f1)
        accuracy = CV_text_NB.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        self.set_y_orig(y_test)
        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)
        
        df_prob = self.Calculer_df_prob(y_pred,y_prob)
        self.set_df_prob(df_prob)
        
      
        df_pred = ds.get_def_prediction(y_test, y_pred,self.get_cat())
        self.set_df_pred(df_pred)
        
        duration = end_time - start_time
        print("La durée de l'entraînement était : ", duration)       
        
# ************************************************************************************************          