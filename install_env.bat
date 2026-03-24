@echo off
echo ===================================================
echo Configuration de l'environnement virtuel pour le projet
echo ===================================================

echo.
echo [1/4] Creation de l'environnement virtuel (venv)...
python -m venv venv

echo.
echo [2/4] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

echo.
echo [3/4] Mise a jour de pip...
python -m pip install --upgrade pip

echo.
echo [4/4] Installation des librairies requises (IA, vision, donnees)...
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn split-folders rembg jupyter ipykernel deep-translator

echo.
echo Installation terminee !
echo Pour activer l'environnement par la suite, utilisez la commande :
echo venv\Scripts\activate
echo.
pause
