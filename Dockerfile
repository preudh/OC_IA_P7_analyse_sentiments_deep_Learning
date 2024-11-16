FROM python:3.11-slim
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ajouter un répertoire temporaire (si nécessaire) et définir les permissions
RUN mkdir -p /temp && chmod -R 777 /temp

# Copier le reste du projet
COPY . .

# Exposer le port par défaut
EXPOSE 80

# Commande pour lancer l'application (utilise la variable d'environnement PORT)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-80}"]



