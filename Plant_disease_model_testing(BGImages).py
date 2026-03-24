# Montez le lecteur Google
from google.colab import drive
drive.mount('/content/drive')

# Spécifier le répertoire de l'ensemble de données de testtest_di
r = '/content/drive/MyDrive/Output_bg/test'

# Utiliser le générateur d'images pour prétraiter les images
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
img_width,img_height =224,224
input_shape=(img_width,img_height,3)
batch_size =32

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(img_width,img_height),batch_size=batch_size)

# Obtenez un lot d'images et d'étiquettes à partir du générateur d'ensembles de données de validation
import numpy as np

test_image_batch, test_label_batch = next(iter(test_generator))
true_label_ids = np.argmax(test_label_batch, axis=-1)

print("Test batch shape:", test_image_batch.shape)

# Générer des étiquettes d'ensemble de données
dataset_labels = sorted(test_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)

import tensorflow as tf

# Spécifier le chemin du modèlemode
l1 = "/content/drive/MyDrive/model1-outputmobilenetof8epoch.tflite"

# Chargez le modèle TFLite (avec arrière-plan) et allouez des tenseurs.
interpreter_1 = tf.lite.Interpreter(model_path=model1)

# Obtenez les tenseurs d'entrée et de sortie.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()

print("== Input details (Bg) ==")
print("name:", input_details_1[0]['name'])
print("shape:", input_details_1[0]['shape'])
print("type:", input_details_1[0]['dtype'])

print("\n== Output details (Model1) ==")
print("name:", output_details_1[0]['name'])
print("shape:", output_details_1[0]['shape'])
print("type:", output_details_1[0]['dtype'])

# Redimensionner le modèle 1 (avec bg)
interpreter_1.resize_tensor_input(input_details_1[0]['index'], (32, 224, 224, 3))
interpreter_1.resize_tensor_input(output_details_1[0]['index'], (32, 10))
interpreter_1.allocate_tensors()

input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()

print("== Input details (model 1) ==")
print("name:", input_details_1[0]['name'])
print("shape:", input_details_1[0]['shape'])
print("type:", input_details_1[0]['dtype'])

print("\n== Output details (model 1) ==")
print("name:", output_details_1[0]['name'])
print("shape:", output_details_1[0]['shape'])
print("type:", output_details_1[0]['dtype'])

# Obtenez les prédictions du modèle 1
interpreter_1.set_tensor(input_details_1[0]['index'], test_image_batch)
interpreter_1.invoke()

model1_predictions = interpreter_1.get_tensor(output_details_1[0]['index'])
print("Prediction results shape:", model1_predictions.shape)

# Convertissez les résultats de prédiction du modèle 1 en dataframe Pandas, pour une meilleure visualisation
import pandas as pd
model1_pred_dataframe = pd.DataFrame(model1_predictions)
model1_pred_dataframe.columns = dataset_labels

print("Model 1 prediction results for the first five elements")
model1_pred_dataframe.head()

# Imprimer des prédictions par lots d'images et d'étiquettes pour le modèle 2
import matplotlib.pyplot as plt
model1_predicted_ids = np.argmax(model1_predictions, axis=-1)
model1_predicted_labels = dataset_labels[model1_predicted_ids]
model1_label_id = np.argmax(test_label_batch, axis=-1)

plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(10,3,n+1)
  plt.imshow(test_image_batch[n])
  color = "green" if model1_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(model1_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Predictions for Model 1 (Green: Correct, Red: Incorrect)\n Trained on images with backgrounds")

import matplotlib.pyplot as plt

# Générer des prédictionsnum_image
s=len(test_image_batch)
model1_predicted_ids = np.argmax(model1_predictions, axis=-1)
model1_predicted_labels = dataset_labels[model1_predicted_ids]
model1_label_id = np.argmax(test_label_batch, axis=-1)

# initialiser un décompte pour garder une trace du nombre de prédictions correctescorrect_prediction
s = 0

# parcourir chaque prédiction et mettre à jour le décompte si la prédiction est correcte
for n in range(num_images):
    if model1_predicted_ids[n] == true_label_ids[n]:
        correct_predictions += 1

# calculer le pourcentage de prédictions correctespercent_correc
t = correct_predictions / num_images * 100

# afficher le pourcentage de prédictions correctes
print("Model 1: {:.2f}% correct predictions".format(percent_correct))


import matplotlib.pyplot as plt

num_images = len(test_image_batch)

model1_predicted_ids = np.argmax(model1_predictions, axis=-1)
model1_predicted_labels = dataset_labels[model1_predicted_ids]
model1_label_id = np.argmax(test_label_batch, axis=-1)

# initialiser un décompte pour garder une trace du nombre de prédictions correctescorrect_prediction
s = 0

# parcourir chaque prédiction et mettre à jour le décompte si la prédiction est correcte
for n in range(num_images):
    if model1_predicted_ids[n] == true_label_ids[n]:
        correct_predictions += 1

# calculer le pourcentage de prédictions correctespercent_correc
t = correct_predictions / num_images * 100

# afficher le pourcentage de prédictions correctes dans un graphique à barres
plt.bar(['Model 1'], percent_correct)
plt.ylim(0, 100)
plt.ylabel('Percentage Correct')
plt.show()


# Spécifier le chemin du modèle pour le modèle sans arrière-plan de l'imagemode
l2 = "/content/drive/MyDrive/model2-outputmobilenetof8epoch.tflite"

# Chargez le modèle TFLite (pas d'arrière-plan) et allouez des tenseurs.
interpreter_2 = tf.lite.Interpreter(model_path=model2)

# Obtenez les tenseurs d'entrée et de sortie.
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

# Redimensionner les tenseurs d'entrée et de sortie pour gérer un lot de 32 imagesinterpreter_2.resize_tensor_input(input_details_2[0]['index'], (32, 224, 224, 3))
interpreter_2.resize_tensor_input(output_details_2[0]['index'], (32, 10))
interpreter_2.allocate_tensors()

# Obtenez les tenseurs d'entrée et de sortie.
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

print("== Input details (model2) ==")
print("name:", input_details_2[0]['name'])
print("shape:", input_details_2[0]['shape'])
print("type:", input_details_2[0]['dtype'])

print("\n== Output details (model2) ==")
print("name:", output_details_2[0]['name'])
print("shape:", output_details_2[0]['shape'])
print("type:", output_details_2[0]['dtype'])

# Exécuter l'inférenceinterpreter_2.set_tensor(input_details_2[0]['index'], test_image_batch)

interpreter_2.invoke()

model2_predictions = interpreter_2.get_tensor(output_details_2[0]['index'])
print("\nPrediction results shape:", model2_predictions.shape)

# Imprimer des prédictions par lots d'images et d'étiquettes pour le modèle 2
import matplotlib.pyplot as plt

model2_predicted_ids = np.argmax(model2_predictions, axis=-1)
model2_predicted_labels = dataset_labels[model2_predicted_ids]
model2_label_id = np.argmax(test_label_batch, axis=-1)

plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(10,3,n+1)
  plt.imshow(test_image_batch[n])
  color = "green" if model2_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(model2_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Predictions for Model 2 (Green: Correct, Red: Incorrect) \n Trained on images without backgrounds")

import matplotlib.pyplot as plt

# Générer des prédictions du modèle 2num_image
s=len(test_image_batch)
model2_predicted_ids = np.argmax(model2_predictions, axis=-1)
model2_predicted_labels = dataset_labels[model2_predicted_ids]
model2_label_id = np.argmax(test_label_batch, axis=-1)

# initialiser un décompte pour garder une trace du nombre de prédictions correctescorrect_prediction
s = 0

# parcourir chaque prédiction et mettre à jour le décompte si la prédiction est correcte
for n in range(num_images):
    if model2_predicted_ids[n] == true_label_ids[n]:
        correct_predictions += 1

# calculer le pourcentage de prédictions correctespercent_correc
t = correct_predictions / num_images * 100

# afficher le pourcentage de prédictions correctes
print("Model 2: {:.2f}% correct predictions".format(percent_correct))

import matplotlib.pyplot as plt

# générer des prédictions du modèle 2num_image
s=len(test_image_batch)
model2_predicted_ids = np.argmax(model2_predictions, axis=-1)
model2_predicted_labels = dataset_labels[model2_predicted_ids]
model2_label_id = np.argmax(test_label_batch, axis=-1)

# initialiser un décompte pour garder une trace du nombre de prédictions correctescorrect_prediction
s = 0

# parcourir chaque prédiction et mettre à jour le décompte si la prédiction est correcte
for n in range(num_images):
    if model2_predicted_ids[n] == true_label_ids[n]:
        correct_predictions += 1

# calculer le pourcentage de prédictions correctespercent_correc
t = correct_predictions / num_images * 100

# afficher le pourcentage de prédictions correctes dans un graphique à barres
plt.bar(['Model 2'], percent_correct)
plt.ylim(0, 100)
plt.ylabel('Percentage Correct')
plt.show()


import matplotlib.pyplot as plt

num_images = len(test_image_batch)

model1_predicted_ids = np.argmax(model1_predictions, axis=-1)
model1_predicted_labels = dataset_labels[model1_predicted_ids]
model1_label_id = np.argmax(test_label_batch, axis=-1)

# initialiser un décompte pour garder une trace du nombre de prédictions correctes pour le modèle 1correct_predictions
_1 = 0

# parcourir chaque prédiction et mettre à jour le décompte si la prédiction est correcte
for n in range(num_images):
    if model1_predicted_ids[n] == true_label_ids[n]:
        correct_predictions_1 += 1

# calculer le pourcentage de prédictions correctes pour le modèle 1percent_correct
_1 = correct_predictions_1 / num_images * 100

# Répétez le même processus pour le modèle 2model2_predicted_ids = np.argmax(model2_predictions, axis=-1)
model2_predicted_labels = dataset_labels[model2_predicted_ids]
model2_label_id = np.argmax(test_label_batch, axis=-1)

correct_predictions_2 = 0
for n in range(num_images):
    if model2_predicted_ids[n] == true_label_ids[n]:
        correct_predictions_2 += 1

percent_correct_2 = correct_predictions_2 / num_images * 100

# afficher le pourcentage de prédictions correctes dans un graphique à barres
plt.bar(['Model 1', 'Model 2'], [percent_correct_1, percent_correct_2])
plt.ylim(0, 100)
plt.ylabel('Percentage Correct')
plt.show()


# Convertissez les résultats de prédiction du modèle 2 en dataframe Pandas, pour une meilleure visualisation
model2_pred_dataframe = pd.DataFrame(model2_predictions)
model2_pred_dataframe.columns = dataset_labels

print("Model 2 prediction results for the first elements")
model2_pred_dataframe.head()

# Concaténer les résultats des deux modèles
all_models_dataframe = pd.concat([model1_pred_dataframe, 
                                  model2_pred_dataframe], 
                                 keys=['Model 1', 'Model 2'],
                                 axis='columns')
all_models_dataframe.head()

# Échangez les colonnes pour avoir une comparaison côte à côte
all_models_dataframe = all_models_dataframe.swaplevel(axis='columns')[model1_pred_dataframe.columns]
all_models_dataframe.head()

# Mettre en évidence les prédictions du modèle 2 qui sont différentes de celles du modèle 1
def highlight_diff(data, color='grey'):
    attr = 'background-color: {}'.format(color)
    other = data.xs('Model 1', axis='columns', level=-1)
    return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''),
                        index=data.index, columns=data.columns)

all_models_dataframe.style.apply(highlight_diff, axis=None)

# Concaténation de argmax et de la valeur max pour chaque ligne
def max_values_only(data):
  argmax_col = np.argmax(data, axis=1).reshape(-1, 1)
  max_col = np.max(data, axis=1).reshape(-1, 1)
  return np.concatenate([argmax_col, max_col], axis=1)

# Créez des tableaux de prédiction simplifiésmodel1_pred_simplifie
d = max_values_only(model1_predictions)
model2_pred_simplified = max_values_only(model2_predictions)

# Construire des DataFrames et présenter un exemplecolumns_name
s = ["Label_id", "Confidence"]
model1_simple_dataframe = pd.DataFrame(model1_pred_simplified)
model1_simple_dataframe.columns = columns_names

model2_simple_dataframe = pd.DataFrame(model2_pred_simplified)
model2_simple_dataframe.columns = columns_names

model1_simple_dataframe.head()
model2_simple_dataframe.head()

# Concaténer les résultats de tous les modèlesall_models_simple_datafram
e = pd.concat([model1_simple_dataframe,
                                         model2_simple_dataframe], 
                                        keys=['Model 1', 'Model 2'],
                                        axis='columns')

# Échangez les colonnes pour une comparaison côte à côteall_models_simple_dataframe = all_models_simple_dataframe.swaplevel(axis='columns')[model1_simple_dataframe.columns]

# Mettre en évidence les différencesall_models_simple_dataframe.style.apply(highlight_diff, axis=None)

