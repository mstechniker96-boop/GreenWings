# Monter Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Importer les bibliothèques nécessaires
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
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

# Spécifier les étiquettes de répertoire et de classeos.chdir("/content/drive/My Drive/")
root_dir = ('/content/drive/MyDrive/Dataset_new/')
classes = os.listdir(root_dir)
classes

# La fonction Splitfolder divise les données en dossiers test, train et val dans votre répertoire spécifié
import splitfolders
splitfolders.ratio("/content/drive/MyDrive/Dataset_new/", output="/content/drive/MyDrive/Output/",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

# Attribuez les répertoires pour l'ensemble de données d'entraînement, de test et de validation
train_dir = '/content/drive/MyDrive/Output/train/'
test_dir = '/content/drive/MyDrive/Output/test/'
val_dir = '/content/drive/MyDrive/Output/val/'

# Créer un générateur de données pour la formation et la validation
from keras.preprocessing.image import ImageDataGenerator

# Spécifier la forme de l'image et la taille du lot
img_width,img_height =224,224
input_shape=(img_width,img_height,3)
batch_size =32

# Redimensionner et augmenter les images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Créer un générateur de données pour la formation et la validation
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(img_width,img_height),batch_size=batch_size,shuffle=True)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(img_width,img_height),batch_size=batch_size)
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(img_width,img_height),batch_size=batch_size)

# Charger le modèle MobileNet et ajouter des couches
from keras.applications.mobilenet import MobileNet
from keras.models import Model
import keras
from keras import optimizers
model_finetuned = Sequential()

model_finetuned.add(MobileNet(weights='imagenet'))
model_finetuned.add(BatchNormalization())
model_finetuned.add(Dense(128, activation="relu"))
model_finetuned.add(Dense(10, activation="softmax"))
for layer in model_finetuned.layers[0].layers:
  if layer.__class__.__name__=="BatchNormalization":
    layer.trainable=True
  else:
    layer.trainable=False

# Ajouter des compilateurs
model_finetuned.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])

model_finetuned.summary()

# Entraîner et ajuster le modèle
from PIL import Image
from keras.callbacks import ReduceLROnPlateau

history_1 = model_finetuned.fit(train_generator,                                    
                                  steps_per_epoch=None, 
                                  epochs=8,validation_data=val_generator,validation_steps=None
                                  ,verbose=1,callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)],use_multiprocessing=False,
               shuffle=True)

# Enregistrer le modèle
from keras.models import load_model
model_finetuned.save('plantdiseasemobilenet8epoch.h5')

classes=list(train_generator.class_indices.keys())
import numpy as np
import matplotlib.pyplot as plt

# Données de test de pré-traitement identiques aux données de train.img_widt
h=224
img_height=224
model_finetuned.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# à partir de l'image d'importation keras.preprocessing
import tensorflow
from keras.preprocessing import image

# Fonction de traitement d'image
def prepare(img_path):
    img = image.load_img(img_path, target_size=(224,224))
 
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)


# Faire une prédiction avec une image de test
import keras.utils as image
test_img = ('/content/drive/MyDrive/Output/test/Tomato___Leaf_Mold/image (251).PNG')
result = model_finetuned.predict([prepare(test_img)])
disease=image.load_img(test_img)
plt.imshow(disease)
print(result)

import numpy as np
classresult=np.argmax(result,axis=1)
print(classes[classresult[0]])

# Convertir le modèle TF en TFLite
import tensorflow as tf
keras_model = tf.keras.models.load_model("plantdiseasemobilenet8epoch.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

model = converter.convert()
file = open('/content/drive/MyDrive/outputmobilenetof8epoch.tflite' , 'wb' ) 
file.write(model)

# Chargez le modèle TFLite et allouez des tenseurs.
import numpy as np
import tensorflow as tf

# Chargez le modèle TFLite et allouez des tenseurs.
interpreter = tf.lite.Interpreter(model_path="outputmobilenetof8epoch.tflite")
interpreter.allocate_tensors()

# Obtenez les tenseurs d'entrée et de sortie.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print("")
print(output_details)

# Préparer le chemin de l'image
def prepare(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)
    
# Spécifiez le répertoire de l'image de test
input_data = [prepare("/content/drive/MyDrive/Output/test/Tomato___Septoria_leaf_spot/image (1077).PNG")]

# Allouer des tenseurs
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_data[0])
interpreter.invoke()

# La fonction `get_tensor()` renvoie une copie des données tensorielles.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

classresult=np.argmax(output_data,axis=1)
print(classes[classresult[0]])

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

print(history_1.history.keys())

plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# résumer l'historique des pertes
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



