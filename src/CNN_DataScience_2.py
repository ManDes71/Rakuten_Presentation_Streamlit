import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#image

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB1
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras import callbacks
from keras.optimizers import Adam


import pickle



import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from  src import Bibli_DataScience_3 as ds



class DS_CNN(ds.DS_Model):
    
     def __init__(self, nom_modele):
     
        super().__init__(nom_modele)
            
        
        self.__nom_modele = nom_modele
        
        self.__model = Sequential()
        self.__base_model = Sequential()
        self.__df_pred = pd.DataFrame()
        self.__y_orig =[]
        self.__y_pred = []
        self.__X = np.array([])
        self.__y = np.array([])
        self.__label_encoder = LabelEncoder()
        self.__report_ID = "CNN30"
        self.__report_MODELE = nom_modele
        self.__report_LIBBELLE = "INCEPTON 5000-2000 SIZE400 DEFREEZE  DR30-CC1024-NORM-CC1024-DR40"
      
     def get_df_pred(self):
        return self.__df_pred
        
     def get_y_orig(self) :
        return self.__y_orig
        
     def get_y_pred(self) :
        return self.__y_pred
     def get_df_cross(self):
        return self.__df_cross

     def set_df_cross(self,cross):
        self.__df_cross = cross   

     def get_labelencoder(self):
        return self.__label_encoder

     def set_labelencoder(self,label):
        self.__label_encoder =label          
        
     def get_model(self) :
        return self.__model
        
     def get_REPORT_ID(self) :
        return self.__report_ID
     def set_REPORT_ID(self,id):
        self.__report_ID = id           
     def get_REPORT_MODELE(self) :
        return self.__report_MODELE
     def set_REPORT_MODELE(self,modele):
        self.__report_MODELE = modele          
     def get_REPORT_LIBELLE(self) :
        return self.__report_LIBBELLE  
     def set_REPORT_LIBELLE(self,libelle):
        self.__report_LIBBELLE = libelle      
      
#     def charger_X_Y(self):
#        df=self. get_DF() 
#        self.__X=np.array(df.filepath)
#        self.__y=np.array(df.prdtypecode) 
        
     def __get_strategie_max(self,Y,max_value):
        new_class_counts = pd.Series(Y).value_counts()
        sampling_strategy=new_class_counts.to_dict()
        for key in sampling_strategy:
          if sampling_strategy[key] > max_value:
            sampling_strategy[key] = max_value
        
        return sampling_strategy
    
     def __get_strategie_min(self,Y,min_value):
        new_class_counts = pd.Series(Y).value_counts()
        sampling_strategy=new_class_counts.to_dict()
        for key in sampling_strategy:
          if sampling_strategy[key] < min_value :
            sampling_strategy[key] = min_value
        
        return sampling_strategy   

     #     Train       : "None" , "Save"  , "Load" ,"Weight"        =>   "Save" : on enregistre les données d'entrainement
     #                                                              =>   "Load" : on charge les données d'entrainement      
     #        
      
     def Train_Test_Split_(self,train_size=0.8, random_state=1234,  RandUnderSampl = True,  RandomOverSampl = True,fic="None"):    
        
        print("Train_Test_Split_1")
        if fic == "Load" :
            print("Récupération de jeu d'entrainement")
            X_train = ds.load_ndarray(self.__nom_modele+'_X_train')
            X_test = ds.load_ndarray(self.__nom_modele+'_X_test')
            y_train_avant = ds.load_ndarray(self.__nom_modele+'_y_train')
            y_test_avant = ds.load_ndarray(self.__nom_modele+'_y_test')
            label_encoder = ds.load_ndarray(self.__nom_modele+'_label_encoder')
            
            return X_train, X_test, y_train_avant, y_test_avant
        
        print("Train_Test_Split_ ")  
        filep =  self.get_DF()['filepath'][:1].values      
        print(filep[:5])        
            
        X_train_avant, X_test_avant, y_train_avant, y_test_avant = super().Train_Test_Split_(train_size, random_state)
      
        print("Train_Test_Split_2",y_train_avant[:5])
      
        X_train = np.array(X_train_avant['filepath'])
        print(X_train.shape)
        X_test = np.array(X_test_avant['filepath'])
        y_train = np.array(y_train_avant)
        print(X_test.shape)
        y_test = np.array(y_test_avant )
        X_train=X_train.reshape(-1,1)
        y_train=y_train.reshape(-1,1)
        X_test=X_test.reshape(-1,1)
        y_test=y_test.reshape(-1,1)
        #print(X_train[:5])
       
        if RandUnderSampl :
             print("Ramdom under sampling : MAX ",self._NB_PAR_LABEL_MAX)
             print("y : ",y_train_avant.shape) 
             sampling_strategy = self.__get_strategie_max(y_train_avant,self._NB_PAR_LABEL_MAX)
             print("Répartition :")
             print(sampling_strategy)
             
             rUs = RandomUnderSampler(sampling_strategy=sampling_strategy)
             X_ru, y_ru = rUs.fit_resample(X_train,y_train  )
        else:
            X_ru = X_train.copy()
            y_ru = y_train.copy()
            print("Train_Test_Split_3",y_ru[:5])
            
        print(len(X_ru))
        print(len(X_ru))
        
        if RandomOverSampl :
            print("Ramdom over sampling : MIN ",self._NB_PAR_LABEL_MIN) 
            sampling_strategy = self.__get_strategie_min(y_ru,self._NB_PAR_LABEL_MIN)
            print("Répartition :")
            print(sampling_strategy)
            rOs = RandomOverSampler(sampling_strategy=sampling_strategy)
            X_train_path, y_train = rOs.fit_resample(X_ru, y_ru)
            X_train_path, y_train = shuffle(X_train_path, y_train, random_state=42)
        else:
            X_train_path = X_ru.copy()
            y_train = y_ru.copy()
            print("Train_Test_Split_4",y_train[:5])

        y_test=y_test.ravel()
        X_train_path=X_train_path.reshape(-1,)
        X_test_path=X_test
        X_test_path=X_test_path.reshape(-1,)
        #print(X_train_path[:5])
        
        if fic == "Save" :
            print("Sauvegarde de jeu d'entrainement")
            ds.save_ndarray(X_train_path,self.__nom_modele+'_X_train')
            ds.save_ndarray(X_test_path,self.__nom_modele+'_X_test')
            ds.save_ndarray(y_train,self.__nom_modele+'_y_train')
            ds.save_ndarray(y_test,self.__nom_modele+'_y_test')
            
        return X_train_path,X_test_path, y_train, y_test  
        
     def preprossessing_Y(self,y_train,y_test):
          label_encoder = LabelEncoder()
          y_classes_converted = label_encoder.fit_transform(y_train)
          y_train_Network = to_categorical(y_classes_converted)
          y_classes_converted = label_encoder.transform(y_test)
          y_test_Network = to_categorical(y_classes_converted)
          print(y_train_Network.shape)
          print(y_test_Network.shape)
          return y_train_Network,y_test_Network,label_encoder
        
    
     def get_image(self,index_im):
        folder_path = ds.get_RACINE_IMAGES()
        # Chemin de l'image
        filename = self.get_DF().nom_image[index_im]
        filepath = os.path.join(folder_path, filename)
        print(filepath) 
        # Lecture du fichier
        im = tf.io.read_file(filepath)

        # On décode le fichier
        im = tf.image.decode_jpeg(im, channels=3)

        return(im)
    
     @tf.function
     def augment_image(self,image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_crop(image,[400, 400, 3])
        return image

     @tf.function
     def load_and_augment_image(self,filepath, IMGSIZE):
        resize=(IMGSIZE,IMGSIZE)
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = self.augment_image(image)
        return tf.image.resize(image, resize)

     @tf.function
     def load_image(self,filepath, IMGSIZE):
        print(filepath) 
        resize=(IMGSIZE,IMGSIZE) 
        im = tf.io.read_file(filepath)
        im = tf.image.decode_jpeg(im, channels=3)
        return tf.image.resize(im, resize)
        
     def generate_dataset(self,X_train_path,X_test_path,y_train_Network,y_test_Network):

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train_path, y_train_Network ))

        dataset_train = dataset_train.map(lambda x, y : [self.load_and_augment_image(x,self._IMGSIZE),y], num_parallel_calls=-1).batch(self._BATCH_SIZE )

        dataset_test = tf.data.Dataset.from_tensor_slices((X_test_path, y_test_Network))

        dataset_test = dataset_test.map(lambda x, y : [self.load_image(x,self._IMGSIZE),y], num_parallel_calls=-1).batch(self._BATCH_SIZE )
        
        return dataset_train,dataset_test
        
     def test_generate_images(self,X_train_path):
     
        random_samples = X_train_path[:5]

        # Créer une figure pour afficher les images
        fig, axs = plt.subplots(5, 2, figsize=(10, 20))

        for i, nom in enumerate(random_samples):
            print(i, nom)
            # Charger l'image originale
            original_image = self.load_image(nom,self._IMGSIZE)/255.0

            # Redimensionner l'image pour l'affichage
            #resized_image = tf.image.resize(original_image, (IMGSIZE, IMGSIZE))

            # Charger l'image à l'aide de la fonction load_image
            augmented_image = self.load_and_augment_image(nom,self._IMGSIZE)/255.0

            # Afficher l'image originale et l'image augmentée
            axs[i, 0].imshow(original_image)
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Original Image')

            axs[i, 1].imshow(augmented_image)
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Augmented Image')

        plt.show()
        
     def freeze_base_modele(self,FREEZE_LAYERS):
        i=0
        #print(dir(self.__base_model))
        for layer in self.__base_model.layers[:FREEZE_LAYERS]:
            i+=1
            layer.trainable = False
        for layer in self.__base_model.layers[FREEZE_LAYERS:]:
            i+=1
            layer.trainable = True
        self._FREEZE_LAYERS = FREEZE_LAYERS
        
        print("il y a ",FREEZE_LAYERS,"couches freezées sur ",i)
        print(i)
        
        
     def create_modele(self):
        pass
        
     def get_summary(self):
        return  self.__model.summary()
    
     #     Train       : "None" , "Save"  , "Load" ,"Weight"        =>   "Save" : on enregistre les données d'entrainement
     #                                                              =>   "Load" : on charge les données d'entrainement      
     #                                                              =>   "Weight" : on charge les poids     
     @tf.autograph.experimental.do_not_convert
     def fit_modele(self,epochs,savefics=False,freeze = 0,newfit=True, RandUnderSampl = True,  RandomOverSampl = True,Train="Save"):
        if newfit :
            print("newfit")
            X_train_path,X_test_path,y_train,y_test = self.Train_Test_Split_(fic=Train, RandUnderSampl = RandUnderSampl,  RandomOverSampl = RandomOverSampl)
            
            
            #print( y_train)
            #print(y_test)            
            y_train,y_test,label_encoder = self.preprossessing_Y(y_train,y_test)
            print("**************")
            #print( y_train)
            #print(y_test)  
            self.set_labelencoder(label_encoder)
            dataset_train,dataset_test = self.generate_dataset(X_train_path,X_test_path,y_train,y_test)
            #self.test_generate_images(X_train_path)
            model = self.create_modele(freeze)  
            if Train == "Load"  or Train == "Weight" :
                ds.load_model(model,self.__nom_modele+'_weight')
            self.__model = model  
            
        else:
            print("load modele")
            model = self.__model  
            ds.load_model(model,self.__nom_modele+'_weight')   
            
            
        lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                            patience=4,
                            factor=0.5,
                            verbose=1,
                            mode='min')
                            
        if Train == 'Save' :
            #print("Sauvegarde de jeu pour la concatenation")
            ds.save_ndarray(X_train_path,self.__nom_modele+'_CONCAT_X_train')
            ds.save_ndarray(X_test_path,self.__nom_modele+'_CONCAT_X_test')
            ds.save_ndarray(y_train,self.__nom_modele+'_CONCAT_y_train')
            ds.save_ndarray(y_test,self.__nom_modele+'_CONCAT_y_test')   
            #ds.save_dataset(dataset_train,self.__nom_modele)
            #ds.save_dataset(dataset_test,self.__nom_modele)
            
        
        training_history = model.fit(dataset_train, epochs=epochs, validation_data = dataset_test, callbacks=[lr_plateau])
        
        if Train == "Save" :
            print("Sauvegarde des poids du modèle")
            print(self.__nom_modele+'_weight')
            ds.save_model(model,self.__nom_modele+'_weight')
        
        print("Sauvegarde de l'historique")
        train_acc = training_history.history['accuracy']
        val_acc = training_history.history['val_accuracy']
        tloss = training_history.history['loss']
        tvalloss=training_history.history['val_loss']
        print("Prédiction")
        predictions = model.predict(dataset_test)
        
        #feature_model_cnn = Model(inputs=model.input, outputs=model.layers[-2].output)
        #x_train_cnn = feature_model_cnn.predict(dataset_train)
        #x_test_cnn = feature_model_cnn.predict(dataset_test)
        
        #x_train_list, y_train_list = [], []
        #for x, y in train_dataset_cnn:
        #    x_train_list.append(x)
        #    y_train_list.append(y)
            
        #x_train_cnn = np.concatenate(x_train_list, axis=0)
        #y_train_cnn = np.concatenate(y_train_list, axis=0)
        
        #x_test_list, y_test_list = [], []
        #for x, y in test_dataset_cnn:
        #    x_test_list.append(x)
        #    y_test_list.append(y)
            
        #x_test_cnn = np.concatenate(x_test_list, axis=0)
        #y_test_cnn = np.concatenate(y_test_list, axis=0)

        
        
       
            
            
        
        y_test_original = np.argmax(y_test, axis=1)
        y_test_original2=label_encoder.inverse_transform(y_test_original)
        y_pred = np.argmax(predictions, axis=1)
        test_pred_orinal2=label_encoder.inverse_transform(y_pred)
        
        
       #if Train == 'Save' or Train == "Weight" :
       #    #print("Sauvegarde de jeu pour la concatenation")
       #    ds.save_ndarray(x_train_cnn,self.__nom_modele+'_CONCAT2_X_train')
       #    ds.save_ndarray(x_test_cnn,self.__nom_modele+'_CONCAT2_X_test')
       #    ds.save_ndarray(y_train,self.__nom_modele+'_CONCAT2_y_train')
       #    ds.save_ndarray(y_test,self.__nom_modele+'_CONCAT2_y_test')
        
        
        
        top5_df = pd.DataFrame({'prdtypecode': y_test_original2,'predict': test_pred_orinal2})

        df_cross=pd.crosstab(top5_df['prdtypecode'], top5_df['predict'],normalize='index')
        self.set_df_cross(df_cross)
        
        self.__df_pred = pd.DataFrame()
        for c in self.get_cat():
            s = df_cross.loc[c].sort_values(ascending=False)[:5]
            df_temp = pd.DataFrame([{'Categorie':c,'predict':s.index[0],'pourc':s.values[0],'predict2':s.index[1],'pourc2':s.values[1],'predict3':s.index[2],'pourc3':s.values[2]}])
            self.__df_pred = pd.concat([self.__df_pred, df_temp], ignore_index=True)
        
        self.__y_orig = y_test_original2
        self.__y_pred = test_pred_orinal2
        
        if Train == "Save" :
            print("Sauvegarde de jeu du label encoder")
            ds.save_ndarray(label_encoder,self.__nom_modele+'_label_encoder')
            #print(self.__nom_modele+'_weight')
            ds.save_model(model,self.__nom_modele+'_weight')
        
        if savefics :
            ds.save_ndarray(train_acc,self.__nom_modele+'_accuracy')
            ds.save_ndarray(val_acc,self.__nom_modele+'_val_accuracy')
            ds.save_ndarray(tloss,self.__nom_modele+'_loss')
            ds.save_ndarray(tvalloss,self.__nom_modele+'_val_loss')
            ds.save_ndarray(y_test_original2,self.__nom_modele+'_y_orig')
            ds.save_ndarray(test_pred_orinal2,self.__nom_modele+'_y_pred')
            ds.save_dataframe(self.__df_pred,self.__nom_modele+'_df_predict')
        
        return train_acc,val_acc,tloss,tvalloss
        
       
class DS_INCEPTION(DS_CNN):
#https://www.kaggle.com/code/modernmariam/flower-cnn-project     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.__base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(self._IMGSIZE,self._IMGSIZE,3))
        
        self.set_REPORT_ID("CNN30")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("INCEPTON 5000-2000 SIZE400 DEFREEZE  DR30-CC1024-NORM-CC1024-DR40")
        
        
        self.set_BATCH_SIZE(16)

     def create_modele(self,freeze=0):
        model = Sequential()
        if freeze > 0 :
            i=0
            for layer in self.__base_model.layers[:freeze]:
                i+=1
                layer.trainable = False
            for layer in self.__base_model.layers[freeze:]:
                i+=1
                layer.trainable = True
            self._FREEZE_LAYERS = freeze
            
            print("il y a ",freeze,"couches freezées sur ",i)
        model.add(self.__base_model) # Ajout du modèle INCEPTION : 311 couches , FREEZE_LAYERS = 289
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.__model= model

        return model  
        
class DS_EfficientNetB1(DS_CNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.__base_model = EfficientNetB1(weights='imagenet', include_top=False,input_shape=(self._IMGSIZE,self._IMGSIZE,3))
        
      
        self.set_REPORT_ID("CNN31")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EfficientNetB1 5000-2000 SIZE400 DEFREEZE  DR40-CC1024-CC1024-DR40")
        self.set_BATCH_SIZE(16)
        

     def create_modele(self,freeze=0):
        model = Sequential()
        if freeze > 0 :
            i=0
            for layer in self.__base_model.layers[:freeze]:
                i+=1
                layer.trainable = False
            for layer in self.__base_model.layers[freeze:]:
                i+=1
                layer.trainable = True
            self._FREEZE_LAYERS = freeze
            
            print("il y a ",freeze,"couches freezées sur ",i)
        model.add(self.__base_model) # Ajout du modèle EfficientNetB1 : 340 couches , FREEZE_LAYERS = 333
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.4))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.__model= model

        return model  
        
class DS_VGG16(DS_CNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.__base_model = VGG16(weights='imagenet', include_top=False,input_shape=(self._IMGSIZE,self._IMGSIZE,3))
        #print(dir(self.__base_model))
        self.set_REPORT_ID("CNN32")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("VGG16 5000-2000 SIZE400 DEFREEZE  DR40-CC1024-CC1024-DR40")
       
        self.set_BATCH_SIZE(16)

     def create_modele(self,freeze=0):
        model = Sequential()
        if freeze > 0 :
            i=0
            for layer in self.__base_model.layers[:freeze]:
                i+=1
                layer.trainable = False
            for layer in self.__base_model.layers[freeze:]:
                i+=1
                layer.trainable = True
            self._FREEZE_LAYERS = freeze
            
        print("il y a ",freeze,"couches freezées sur ",i)
        model.add(self.__base_model) # Ajout du modèle VGG16 : 19 couches ,  FREEZE_LAYERS = 15
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.4))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.__model= model

        return model         
class DS_RESNET50(DS_CNN):     

     def __init__(self, nom_modele=0):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        #ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)
        self.__base_model = ResNet50(weights='imagenet', pooling='avg',include_top=False,input_shape=(self._IMGSIZE,self._IMGSIZE,3))
        #print(dir(self.__base_model))
        self.set_REPORT_ID("CNN120")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("RESNET50 Adam 4000-2000 SIZE400 DeFreeze  DR40-CC1024-CC1024-DR40")
       
        self.set_BATCH_SIZE(16)

     def create_modele(self,freeze):
        model = Sequential()
        if freeze > 0 :
            i=0
            for layer in self.__base_model.layers[:freeze]:
                i+=1
                layer.trainable = False
            for layer in self.__base_model.layers[freeze:]:
                i+=1
                layer.trainable = True
            self._FREEZE_LAYERS = freeze
            
            print("il y a ",freeze," couches freezées sur ",i)
        else : 
            print("il y a ",len(self.__base_model.layers)," couches sur ce modèle.")
        model.add(self.__base_model) # Ajout du modèle ResNet50 : 19 couches ,  FREEZE_LAYERS = 15
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1024, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.__model= model

        return model                  
class DS_VGG19(DS_CNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        #ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)
        self.__base_model = VGG19(weights='imagenet', include_top=False,input_shape=(self._IMGSIZE,self._IMGSIZE,3))
        #print(dir(self.__base_model))
        self.set_REPORT_ID("CNN35")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("DS_VGG19 5000-2000 SIZE400 DEFREEZE  DR40-CC1024-CC1024-DR40")
       
        self.set_BATCH_SIZE(32)

     def create_modele(self,freeze=0):
        model = Sequential()
        if freeze > 0 :
            i=0
            for layer in self.__base_model.layers[:freeze]:
                i+=1
                layer.trainable = False
            for layer in self.__base_model.layers[freeze:]:
                i+=1
                layer.trainable = True
            self._FREEZE_LAYERS = freeze
            
            print("il y a ",freeze," couches freezées sur ",i)
        else : 
            print("il y a ",len(self.__base_model.layers)," couches sur ce modèle.")
        model.add(self.__base_model) # Ajout du modèle ResNet50 : 19 couches ,  FREEZE_LAYERS = 15
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.__model= model

        return model   
class DS_Xception(DS_CNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        #ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)
        self.__base_model = Xception(weights='imagenet', include_top=False,input_shape=(self._IMGSIZE,self._IMGSIZE,3))
        #print(dir(self.__base_model))
        self.set_REPORT_ID("CNN35")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("DS_Xception 5000-2000 SIZE400 DEFREEZE  DR40-CC1024-CC1024-DR40")
       
        self.set_BATCH_SIZE(8)
        print(self.__base_model.summary())

     def create_modele(self,freeze=0):
        model = Sequential()
        if freeze > 0 :
            i=0
            for layer in self.__base_model.layers[:freeze]:
                i+=1
                layer.trainable = False
            for layer in self.__base_model.layers[freeze:]:
                i+=1
                layer.trainable = True
            self._FREEZE_LAYERS = freeze
            
            print("il y a ",freeze," couches freezées sur ",i)
        else : 
            print("il y a ",len(self.__base_model.layers)," couches sur ce modèle.")
        model.add(self.__base_model) # Ajout du modèle ResNet50 : 19 couches ,  FREEZE_LAYERS = 15
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.__model= model

        return model                                 