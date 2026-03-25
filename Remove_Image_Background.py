# Monter le lecteur là où se trouvent les dossiers d'images
# from google.colab import drive
# drive.mount('/content/drive')

# Importez les bibliothèques nécessaires
from rembg import remove
from PIL import Image
import io
import glob

# Spécifiquement le répertoire ciblefile
s = glob.glob('/content/drive/MyDrive/Dataset_new/Tomato___Target_Spot/*.PNG')
len(files)

# Cette boucle sur chaque image supprime l'arrière-plan et enregistre dans un nouveau répertoire
for file in files:
  input_path = file
  output_path = input_path.replace("Dataset", "Dataset_new")
  output_path = output_path.replace("JPG", "PNG")
  
  with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)

print('Completed Successfully!!!')

