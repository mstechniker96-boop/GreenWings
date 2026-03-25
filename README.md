# GREENWINGS : Drone Intelligent pour l'Agriculture de Précision

Bienvenue dans le dépôt officiel du projet **GreenWings**, un drone eVTOL autonome conçu pour révolutionner l'agriculture de précision grâce à l'Intelligence Artificielle et à l'Edge Computing.

## Objectif du Projet
L'objectif principal est de fabriquer et de déployer un drone capable de diagnostiquer en temps réel les maladies des plantes directement sur le terrain. En utilisant des techniques d'apprentissage profond (Deep Learning) embarquées, **GreenWings** permet une intervention rapide, ciblée et efficace, réduisant ainsi les coûts et l'utilisation de pesticides.

### Caractéristiques Principales :
*   **Identification Automatisée** : Détection des maladies via caméra embarquée et modèles MobileNet.
*   **Edge Computing** : Traitement des données localement sur Raspberry Pi 5 / Coral TPU pour une latence minimale.
*   **Optimisation Énergétique** : Algorithmes IA (Random Forest) pour maximiser l'autonomie de vol.
*   **Monitoring en Temps Réel** : Interface de contrôle (Dashboard) pour suivre la télémétrie et les détections.

## APERÇU TECHNIQUE :
Ce système utilise un réseau neuronal convolutif **MobileNet V2** optimisé via **TensorFlow Lite**. Le modèle a été entraîné sur des milliers d'images pour garantir une précision de plus de 90%, même dans des conditions de ressources limitées.

## ### PRÉPARATION DES DONNÉES
Pour que l'IA puisse apprendre, vous devez placer vos images dans les dossiers suivants :
*   **Entraînement** : Placez vos dossiers de classes (ex: `Apple___Black_rot`, `Tomato___Healthy`) dans le dossier `dataset/`.
*   **Sortie** : Le dossier `Output_bg/` sera utilisé automatiquement pour stocker les images sans arrière-plan.

STRUCTURE DU PROJET
Le projet a été rendu entièrement standalone (sans dépendance aux notebooks Jupyter) pour une portabilité maximale :

1.  **run_project.bat** : Le lanceur principal interactif pour tout gérer.
2.  **install_env.bat** : Script d'installation automatique de l'environnement virtuel.
3.  **monitoring/** : Dossier contenant le Dashboard de contrôle Web (HTML/CSS/JS).
4.  **requirements.txt** : Liste exhaustive des dépendances Python.
5.  **Scripts de Modélisation (Python)** :
    *   `Model1_disease_detection_using_mobilenet_tl.py`
    *   `Model2_plant_disease_detection_using_mobilenet_tl.py`
    *   `Modelx_plant_disease_detection_using_Mobilenet.py`
    *   `Plant_disease_model_testing(BGImages).py`
    *   `Remove_Image_Background.py`
6.  **Documentation** : Thèse de Master et présentations PDF incluses.

## Comment démarrer ?
1. Exécutez `install_env.bat` pour configurer l'environnement.
2. Lancez `run_project.bat` pour accéder au menu de contrôle et au Dashboard de monitoring.

---
Signé,
**kevin mbuse**
*Nuru Labs*

