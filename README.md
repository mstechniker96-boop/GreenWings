# INTRODUCTION

Bienvenue dans le dépôt de mon projet de master en Data Science.

## Titre du projet : L'Edge Computing au Service de l'Agriculture de Précision - Identification Automatisée des Maladies des Plantes

L'objectif principal de ce projet est de développer l'agriculture de précision. Il s'agit d'un projet d'edge computing qui utilise des techniques d'apprentissage profond déployées directement sur des appareils mobiles et edge pour le diagnostic local des cultures.

Traditionnellement, l'identification des maladies des plantes reposait sur une inspection manuelle par des experts, ce qui est un processus long, coûteux et souvent limité en termes de scalabilité.

Cependant, avec la disponibilité généralisée des appareils mobiles et les progrès des capacités de traitement des données, il existe une opportunité passionnante de développer une solution pratique et accessible. Imaginez que les agriculteurs puissent identifier les maladies des cultures simplement en capturant une photo avec leur téléphone mobile.

Dans ce dépôt, vous trouverez divers fichiers liés au projet, y compris la documentation, le code et les ensembles de données. Ces ressources vous guideront sur la façon de les utiliser efficacement.

Merci de votre intérêt et n'hésitez pas à explorer le contenu de ce dépôt !

## APERÇU :

Ce projet présente un système alimenté par l'IA pour la détection automatisée des maladies des feuilles de tomate, optimisé pour un déploiement sur les appareils mobiles et edge. 

En utilisant un réseau neuronal convolutif MobileNet et TensorFlow Lite, le modèle a été entraîné sur plus de 16 000 images, avec et sans suppression de l'arrière-plan, afin d'évaluer l'impact du prétraitement sur les performances de classification. 

Les résultats ont montré que la suppression des arrière-plans d'images améliorait la précision globale et la robustesse du modèle. 

Le système a atteint une précision de plus de 90 % et a démontré une efficacité adaptée aux environnements agricoles en temps réel et à faibles ressources. Ce travail contribue au diagnostic précoce des maladies des cultures et constitue la fondation d'AgroVerse, une plateforme de santé des plantes évolutive.

## PLAN DU CONTENU

1. Document de Thèse de Master en Data Science : PDF du projet complet avec revue de la littérature, méthodologie, discussions et conclusion.
2. Model1_disease_detection_using_mobilenet_tl.ipynb : L'algorithme de retraitement, d'entraînement et de validation du Modèle 1
3. Model2_plant_disease_detection_using_mobilenet_tl.ipynb : Algorithme pour l'entraînement et la validation du Modèle 2
4. Modelx_plant_disease_detection_using_Mobilenet : Algorithme pour l'entraînement et la validation du Modèle X
5. Plant_disease_model_testing(BGImages).ipynb : Algorithme pour tester les Modèles 1 et 2 (images de test de feuilles de tomate avec arrière-plan)
6. Plant_disease_model_testing(No_BGImages).ipynb : Algorithme pour tester les Modèles 1 et 2 (images de test de feuilles de tomate sans arrière-plan)
7. README.md - vous y êtes déjà :)
8. Remove_Image_Background.ipynb : Algorithme simple pour supprimer automatiquement les arrière-plans d'une collection de feuilles d'images.
9. Viva presentation.pdf - Une présentation sous forme de diaporama en pdf, affichant les résultats et la conclusion de ce travail.

Signé,
kevin mbuse
