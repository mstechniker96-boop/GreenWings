# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Décompressez le dossier pour accéder aux fichiers
from zipfile import ZipFile
# !unzip drive/MyDrive/plantvillagedataset.zip

# Importer les bibliothèques nécessaires
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
import os 
import pandas as pd
import plotly.graph_objs as go
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Spécifier les répertoires train et testtrain_di
r ="/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
test_dir="/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"


# Utilisation d'ImageDataGenerator pour redimensionner et augmenter les images du répertoire
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

# Spécifier la taille de l'image et la taille du lot
img_width,img_height =224,224
input_shape=(img_width,img_height,3)
batch_size =32

# Créer un objet pour analyser les données
train_generator =train_datagen.flow_from_directory(train_dir,target_size=(img_width,img_height),batch_size=batch_size)
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,target_size=(img_width,img_height),batch_size=batch_size)

# Importer les bibliothèques nécessaires
from keras.applications.mobilenet import MobileNet
from keras.models import Model
import keras
from keras import optimizers

# Charger le modèle, ajouter des couches et spécifier les compilateurs
model_finetuned = Sequential()
model_finetuned.add(MobileNet(weights='imagenet'))
model_finetuned.add(BatchNormalization())
model_finetuned.add(Dense(128, activation="relu"))
model_finetuned.add(Dense(38, activation="softmax"))
for layer in model_finetuned.layers[0].layers:
  if layer.__class__.__name__=="BatchNormalization":
    layer.trainable=True
  else:
    layer.trainable=False
model_finetuned.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])

# Afficher les calques, la forme et les paramètres du modèlemodel_finetuned.summary()

# Entraîner le modèle avec les données de l'objet générateur d'images
from keras.callbacks import ReduceLROnPlateau
validation_generator = train_datagen.flow_from_directory(
                       test_dir, # same directory as training data
                       target_size=(img_height, img_width),
                       batch_size=batch_size)

# Spécifier les paramètres d'entraînementhistory
_1 = model_finetuned.fit(train_generator,                                    
                                  steps_per_epoch=None, 
                                  epochs=8,validation_data=validation_generator,validation_steps=None
                                  ,verbose=1,callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)],use_multiprocessing=False,
               shuffle=True)

# importer les bibliothèques nécessaires
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Enregistrer le modèle entraînémodel_finetuned.save('plantdiseasemobilenet8epoch.h5')

# Compiler le modèlemodel_finetuned.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

img_width=224
img_height=224

# Fonction de traitement d'image
def prepare(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

# Spécifier les étiquettes de classeclasses=list(train_generator.class_indices.keys())

# Faire une prédiction et afficher le résultatresul
t = model_finetuned.predict([prepare("/content/drive/MyDrive/images (3).jpg")])
disease=image.load_img('/content/drive/MyDrive/images (3).jpg')
plt.imshow(disease)
print(result)

# Afficher la classe prédite
import numpy as np
classresult=np.argmax(result,axis=1)
print(classes[classresult[0]])

import tensorflow as tf

# Charger le modèle TensorFlowkeras_mode
l = tf.keras.models.load_model("plantdiseasemobilenet8epoch.h5")

# Convertir le modèle à l'aide du convertisseur TFLiteconverte
r = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Ecrire le modèlemode
l = converter.convert()
file = open( 'outputmobilenetof8epoch.tflite' , 'wb' ) 
file.write(model)

import numpy as np
import tensorflow as tf

# Chargez le modèle TFLite et allouez des tenseurs.interpreter = tf.lite.Interpreter(model_path="outputmobilenetof8epoch.tflite")
interpreter.allocate_tensors()

# Obtenez les tenseurs d'entrée et de sortie.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print("")
print(output_details)

# Préparer l'image de test
def prepare(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)
    
# Spécifier le répertoire d'imagesinput_dat
a = [prepare("/content/drive/MyDrive/images (5).jpg")]

# Définir l'interprèteinput_shap
e = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_data[0])
interpreter.invoke()

# La fonction `get_tensor()` renvoie une copie des données tensorielles.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# Afficher la prédiction
classresult=np.argmax(output_data,axis=1)
print(classes[classresult[0]])

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

# résumer l'historique pour plus d'exactitude
print(history_1.history.keys())
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history 
for loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



