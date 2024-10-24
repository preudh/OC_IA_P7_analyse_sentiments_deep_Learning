FROM python:3.11-slim
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Commande pour lancer l'application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]

