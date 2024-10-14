La plateforme Challenge Data est gérée par l'équipe Data (ENS Paris), en partenariat avec le Collège  
de France et le Data Lab de l'Institut Louis Bachelier.  
Elle est soutenue par la Chaire CFM, l'Institut PRAIRIE et l'IDRIS du CNRS.

Chaque année, elle organise des challenges en Data Science à partir de données fournies par les services publics,  
les entreprises ou les laboratoires

https://challengedata.ens.fr/

Depuis 2016, challengedata.ens.fr organise des datas challenges sur le traitement des données par apprentissage  
supervisé. 

Le challenge traité ici est le "Rakuten Multi-modal Colour Extraction project" par Rakuten Institute of Technology, Paris

https://challengedata.ens.fr/challenges/35

Ce dépot est la présentation du projet fil rouge  que j'ai réalisé pour mon diplôme de data Scientist chez Datascientest.com.  
Cette présentation utilise le framework python Streamlit.  
**Streamlit** est un framework open source conçu pour simplifier le développement d’applications web interactives.  
Il est particulièrement utile pour intégrer des modèles de machine learning créés avec des bibliothèques comme TensorFlow, PyTorch, ou Scikit-learn, et les exposer via une interface simple.
Ce dépot est visualisable à cette [adresse](http://streamlit.aventuresdata.com/rakuten/)    
 

**Description du problème**    

L'objectif de ce projet était la classification multimodale de 85000 produits (à partir de descriptions et d'images) en 27 catégories.


Ce dépot GitHub fait la synthèse des 3 dépots précédents :

[Partie Machine Learning](https://github.com/ManDes71/Rakuten_Text_Classification_ML/blob/main/ReadMe.md) , 
[Partie réseaux de neurones récurrents](https://github.com/ManDes71/Rakuten_Text_Classification_TensorFlow/blob/main/ReadMe.md) pour la partie texte ,
[Partie réseaux de neuronnes convolutifs](https://github.com/ManDes71/Rakuten_Images_Classification_TensorFlow/blob/main/README.md) pour la partie classification par les images.  

 **Taches accomplies :**  
  
 *  Exploration des données
 *  traitement des valeurs manquantes
 *  visualisation à l’aide des librairies Python Matplotlib et Seaborn
 *  Utilisation de modèles de Machine Learning et de réseaux de neurones récurrents  pour la partie texte (prédiction avec une précision de 80%)
 *  Utilisation de réseaux de neurones convolutifs pour la partie image
 *  Présentation du projet à l'aide de Streamlit
 
 Utiliser **Streamlit** pour présenter un projet de Datascience necessite d'avior enregistré les modèles au préalable
 ainsi que les resultats des entrainements.  
 Streamlit est capable d'entrainer les modèles en direct (je le fais pour les sections de concaténations de modèles) mais  
géneralement la machine sur laquelle s'execute la présentation du projet ne dispose pas des ressources nécessaires  
en Cpu et en mémoire pour entrainer un réseau convolutif avec plus de 89000 images à traiter.  

<span style="color:blue">*Remarque* : github n'héberge pas les fichiers de plus de 50 M. Ceux-ci sont à télécharger en cliquant sur les liens ci-dessous  </span>

Voucs pouvez télécharger le fichier X_train_update.csv :  [ici](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/X_train_update.csv)
   
Pour le **Machine Learnig**, cette page a besoin pour fonctionner :  
 *  des fichiers paramètres de chaque modèle  :  *_dump.joblib dans le dossier 'fichiers' 
 *  des fichiers de test sur lesquels les modèles vont s'executer : 'X_test.pkl' et 'y_test.pkl' dans le dossier 'fichiers' 
 *  des classes prédites obtenues pour le modèle sur le jeu de test (=y_pred) :  *_pred.pkl dans le dossier 'fichiers' --> 
 *  des probabilités obtenues pour chaque classe par le modèle SVC (la prédiction par le calcul est trop longue à s'executer) :  Mon_Modele_SVC_prob.pkl dans le dossier 'input' 
 *  des prédictions obtenues pour chaque classe par le modèle SVC (la prédiction par le calcul est trop longue à s'executer) :  Mon_Modele_SVC_pred.pkl dans le dossier 'input'
 *  des fichiers pour faire la transformation inverse de l'encoder utilisé sur les classes lors de l'entrainement :  XGBClassifier_label_encoder.pkl dans le dossier 'fichiers' 
 *  il y en a 7 par modèle : 'SVC', 'LogisticRegression', 'RandomForestClassifier', 'GradientBoosting', 'XGBClassifier', 'DecisionTreeClassifier', 'MultinomialNB', 'LinearSVC'   
 
Pour les **Réseaux à convolution (CNN)**, cette page a besoin pour fonctionner :  

 *  des classes prédites réelles du jeu de test (=y_orig) : *_y_orig.pkl dans le dossier 'fichiers' 
 *  RandomForestClassifier_dump.joblib [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/RandomForestClassifier_dump.joblib)
 *  GradientBoosting_dump.joblib [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/GradientBoosting_dump.joblib)
 *  de l'historique de la precision et de perte lors de l'entrainement du modèle :  *_accuracy.pkl et *_loss.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte issues de l'application du modèle au jeu de validation :  *_val_accuracy.pkl et *_val_loss.pkl dans le dossier 'fichiers' 
 *  des classes prédites obtenues par chaque modèle sur le jeu de test (=y_pred) :  *_y_pred.pkl dans le dossier 'fichiers' 
 *  des dataframes representant pour chaque classe réelle les 3 plus importantes classes prédites   :  *_df_predict dans le dossier 'input'
 *  il y en a 6 par modèle : 'Modèle EfficientNetB1', 'Modèle InceptionV3', 'Modèle VGG16', 'Modèle ResNet50', 'Modèle VGG19', 'Modèle Xception'   
 *
 
 Pour les **Réseaux récurrents (RNN)**, cette page a besoin pour fonctionner :  

 *  des classes prédites réelles du jeu de test (=y_orig) : *_y_orig.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte lors de l'entrainement du modèle :  *_accuracy.pkl et *_loss.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte issues de l'application du modèle au jeu de validation :  *_val_accuracy.pkl et *_val_loss.pkl dans le dossier 'fichiers' 
 *  des classes prédites obtenues par chaque modèle sur le jeu de test (=y_pred) :  *_y_pred.pkl dans le dossier 'fichiers' 
 *  des dataframes representant pour chaque classe réelle les 3 plus importantes classes prédites   :  *_df_predict dans le dossier 'input'
 *  il y en a 4 par modèle : 'Modèle EMBEDDING SIMPLE', 'Modèle EMBEDDING + Lemmatisation', Modèle EMBEDDING + stemming', 'Modèle EMBEDDING + SPACY'
   
Pour le **Contexte du projet**, cette page a besoin pour fonctionner :  
 *  des fichiers csv du challenge **Rakuten** : 'X_train_update.csv' , 'Y_train_CVw08PX.csv' et 'X_test_update.csv' dans le dossier 'input'
 *  du fichier des nomenclatures pas classe : 'NOMENCLATURE.csv' dans le dossier 'input' 
 
 
 Pour l'**Exploration des données**, cette page a besoin pour fonctionner :  
 *  des fichiers csv du challenge **Rakuten** : 'X_train_update.csv' et 'Y_train_CVw08PX.csv' dans le dossier 'input'
 *  du fichier des nomenclatures pas classe : 'NOMENCLATURE.csv' dans le dossier 'input' 
 *  du fichier Stopwords : 'stopwords_FR_02.csv' dans le dossier 'input' 
 *  d'un fichier créé à partir de la bibliothèque **detectlang**  pour détecter la langue des libellés   :  'df_langue.csv' dans le dossier 'input' 
 *  d'un fichier csv pour créer les nuages de mots   :  'Top40.csv' dans le dossier 'input'
 *  d'un fichier json pour créer l'histogramme RGB   :  'DfhistoMean.json' dans le dossier 'input' [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/DfColorMean.json)
 *  d'un fichier json pour créer le diagramme 'Moyenne des images'   :  'DfColorMean.json.json' dans le dossier 'input'
 *  d'un fichier csv pour créer l'sistogramme des contours   :  'Dfcontour.csv' dans le dossier 'input' [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/Dfcontour.csv)
  
  Pour le **test d'une image**, cette page a besoin pour fonctionner :  

 *  des poids du modèle EfficientNetB1 à charger : EfficientNetB1_weight.h5 dans le dossier 'input'
 *  des fichiers pour faire la transformation inverse de l'encoder utilisé sur les classes lors de l'entrainement :  'EfficientNetB1_label_encoder.pkl' et 'EfficientNetB1_y_train' dans le dossier 'fichiers' 
 *  du fichier des nomenclatures pas classe : 'NOMENCLATURE.csv' dans le dossier 'input' 
 *  des poids du modèle SVC à charger :  'Mon_Modele_SVC_dump.joblib' dans le dossier 'fichiers' 
 
  Pour le **Modèle concaténation 3 modèles**, cette page a besoin pour fonctionner :  

 *  des classes prédites réelles du jeu de test (=y_orig) : EfficientNetB1_y_orig.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte lors de l'entrainement du modèle :  EfficientNetB1_accuracy.pkl et EfficientNetB1_loss.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte issues de l'application du modèle au jeu de validation :  EfficientNetB1_val_accuracy.pkl et EfficientNetB1_val_loss.pkl dans le dossier 'fichiers' 
 *  des classes prédites obtenues par chaque modèle sur le jeu de test (=y_pred) :  EfficientNetB1_y_pred.pkl dans le dossier 'fichiers' 
 *  des situations du reseaux à l'avant derniére couche : 
 *  *  'EfficientNetB1_CONCAT2_X_train' [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/EfficientNetB1_CONCAT2_X_train.pkl)
 *  *  'EfficientNetB1_CONCAT2_X_test'  [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/EfficientNetB1_CONCAT2_X_test.pkl)
 *  *  'EfficientNetB1_CONCAT2_y_train'
 *  *  'EfficientNetB1_CONCAT2_y_test'
 *  des classes prédites réelles du jeu de test (=y_orig) : EMBEDDING_y_orig.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte lors de l'entrainement du modèle :  EMBEDDING_accuracy.pkl et EMBEDDING_loss.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte issues de l'application du modèle au jeu de validation :  EMBEDDING_val_accuracy.pkl et EMBEDDING_val_loss.pkl dans le dossier 'fichiers' 
 *  des classes prédites obtenues par chaque modèle sur le jeu de test (=y_pred) :  EMBEDDING_y_pred.pkl dans le dossier 'fichiers' 
 *  des situations du reseaux à l'avant-derniére couche : 
 *  *  'EMBEDDING_CONCAT2_X_train'  [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/EMBEDDING_CONCAT2_X_train.pkl)
 *  *  'EMBEDDING_CONCAT2_X_test'
 *  *  'EMBEDDING_CONCAT2_y_train'
 *  *  'EMBEDDING_CONCAT2_y_test'
 *  des fichiers de test sur lesquels le modèles "LinearSVC" va s'executer : 'X_test.pkl' et 'y_test.pkl' dans le dossier 'fichiers'  
 *  des situations du modèles "LinearSVC"  à l'avant-derniére couche : 
 *  *  'LinearSVC_CONCAT2_X_train' [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/LinearSVC_CONCAT2_X_train.pkl)
 *  *  'LinearSVC_CONCAT2_X_test' [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/LinearSVC_CONCAT2_X_test.pkl)
 *  *  'LinearSVC_CONCAT2_y_train'
 *  *  'LinearSVC_CONCAT2_y_test'
 
  Pour le **Modèle concaténation LM er RNN**, cette page a besoin pour fonctionner :  

 *  des classes prédites réelles du jeu de test (=y_orig) : EMBEDDING GRU_y_orig.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte lors de l'entrainement du modèle :  EMBEDDING GRU_accuracy.pkl et EMBEDDING GRU_loss.pkl dans le dossier 'fichiers' 
 *  de l'historique de la precision et de perte issues de l'application du modèle au jeu de validation :  EMBEDDING GRU_val_accuracy.pkl et EMBEDDING GRU_val_loss.pkl dans le dossier 'fichiers' 
 *  des classes prédites obtenues par chaque modèle sur le jeu de test (=y_pred) :  EMBEDDING GRU_y_pred.pkl dans le dossier 'fichiers' 
 *  des situations du reseaux à l'avant-derniére couche : 
 *  *  'EMBEDDING GRU_CONCAT2_X_train' [à télécharger](https://streamlit-rakuten.s3.eu-west-3.amazonaws.com/EMBEDDING+GRU_CONCAT2_X_train.pkl)
 *  *  'EMBEDDING GRU_CONCAT2_X_test' 
 *  *  'EMBEDDING GRU_CONCAT2_y_train'
 *  *  'EMBEDDING GRU_CONCAT2_y_test'
 *  des fichiers de test sur lesquels le modèles "LinearSVC" va s'executer : 'X_test.pkl' et 'y_test.pkl' dans le dossier 'fichiers'  
 *  des situations du modèles "LinearSVC"  à l'avant-derniére couche : 
 *  *  'LinearSVC_CONCAT2_X_train' 
 *  *  'LinearSVC_CONCAT2_X_test' 
 *  *  'LinearSVC_CONCAT2_y_train'
 *  *  'LinearSVC_CONCAT2_y_test'
 
 Pour le **Modèle probabiliste**, cette page a besoin pour fonctionner :  
 
 *  du fichier paramètres du modèle SVC  :  Mon_Modele_SVC_dump.joblib dans le dossier 'fichiers' 
 *  des probabilités obtenues pour chaque classe pour le modèle (modele.predict_proba()) :  Mon_Modele_SVC_prob.csv dans le dossier 'input' 
 *  des fichiers de test sur lesquels le modèle va s'executer : 'X_test.pkl' et 'y_test.pkl' dans le dossier 'fichiers'  
 *  du fichier paramètres du modèle LogisticRegression  :  LogisticRegression_dump.joblib dans le dossier 'fichiers' 
 *  des probabilités obtenues pour chaque classe pour le modèle (modele.predict_proba()) :  LogisticRegression_prob.csv dans le dossier 'input'  
 *  du fichier paramètres du modèle RandomForestClassifier  :  RandomForestClassifier_dump.joblib dans le dossier 'fichiers' 
 *  des probabilités obtenues pour chaque classe pour le modèle (modele.predict_proba()) :  Mon_Modele_SVC_prob.pkl dans le dossier 'finput' 