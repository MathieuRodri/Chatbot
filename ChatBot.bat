@echo off

echo Vérification de l'existence de l'environnement virtuel...
if not exist "venv" (
    echo Création de l'environnement virtuel...
    python -m venv venv
) else (
    echo L'environnement virtuel existe déjà.
)

echo Activation de l'environnement virtuel...
call venv\Scripts\activate

echo Installation des dépendances...
pip install -r requirements.txt

echo Lancement de main.py...
streamlit run main.py

echo Fin du script.
pause
