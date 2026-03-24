@echo off
echo ===================================================
echo Bienvenue dans GreenWings (Version Nomade)
echo ===================================================

if not exist venv\Scripts\activate.bat (
    echo [!] L'environnement virtuel n'est pas encore configure.
    echo Veuillez d'abord executer install_env.bat !
    pause
    exit /b
)

echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

:menu
echo.
echo Que souhaitez-vous executer ?
echo 1. Entrainer Model1 (MobileNet TL)
echo 2. Entrainer Model2 (MobileNet TL)
echo 3. Entrainer Modelx (MobileNet TL)
echo 4. Tester le modele (Avec Background)
echo 5. Tester le modele (Sans Background)
echo 6. Supprimer le background des images
echo 7. Traduire le texte (translate_notebooks.py)
echo 8. Ouvrir le Dashboard de Monitoring (Interface Web)
echo 9. Quitter
echo.
set /p choix="Votre choix (1-9): "

if "%choix%"=="1" python Model1_disease_detection_using_mobilenet_tl.py
if "%choix%"=="2" python Model2_plant_disease_detection_using_mobilenet_tl.py
if "%choix%"=="3" python Modelx_plant_disease_detection_using_Mobilenet.py
if "%choix%"=="4" python "Plant_disease_model_testing(BGImages).py"
if "%choix%"=="5" python "Plant_disease_model_testing(No_BGImages).py"
if "%choix%"=="6" python Remove_Image_Background.py
if "%choix%"=="7" python translate_notebooks.py
if "%choix%"=="8" start monitoring\index.html
if "%choix%"=="9" exit /b

echo.
pause
goto menu
