from pathlib import Path
import os

def launch_mlflow_server():
    # Chemin absolu vers le dossier "mlruns" à la racine du projet
    mlruns_path = Path("./mlruns").resolve()

    # Vérification que le dossier "mlruns" existe, sinon le créer
    if not mlruns_path.exists():
        print(f"Le dossier {mlruns_path} n'existe pas. Création du dossier...")
        mlruns_path.mkdir(parents=True)

    # Lancer le serveur MLflow avec le bon chemin
    os.system(f"mlflow ui --backend-store-uri {mlruns_path.as_uri()} --host 127.0.0.1 --port 5000")

if __name__ == "__main__":
    launch_mlflow_server()
