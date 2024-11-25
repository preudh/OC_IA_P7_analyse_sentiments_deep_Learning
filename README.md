# Réalisez une Analyse de Sentiments avec le Deep Learning et le MLOps

Bienvenue dans ce projet d’analyse de sentiments basé sur le Deep Learning et les pratiques MLOps. Ce dépôt GitHub présente toutes les étapes nécessaires pour développer, déployer, et maintenir des modèles de prédiction de sentiments à partir de données textuelles.

## Objectifs du Projet

Ce projet a pour but de :

- Développer des modèles prédictifs : Construire des modèles de sentiment en utilisant des approches variées, de la régression logistique aux réseaux de neurones profonds (FastText, GloVe, BERT).
- Implémenter des pratiques MLOps : Suivre les expérimentations avec MLFlow, déployer les modèles via des pipelines CI/CD, et surveiller leur performance en production.
- Automatiser le déploiement continu : Utiliser Docker et GitHub Actions pour déployer le modèle sur Azure.
- Créer un système de monitoring : Configurer Azure Application Insights pour surveiller les performances et détecter les anomalies en production.

## Contexte

Vous êtes ingénieur IA chez MIC (Marketing Intelligence Consulting), une entreprise spécialisée en marketing digital. Votre mission consiste à développer un système permettant de prédire les sentiments exprimés dans des tweets concernant Air Paradis, une compagnie aérienne souhaitant mieux gérer les risques de "bad buzz" sur les réseaux sociaux.

## Compétences Développées

1. Modélisation IA :
   - Construction de modèles simples (régression logistique) et avancés (embeddings, BERT).
   - Évaluation et optimisation des modèles.
2. MLOps :
   - Suivi des expérimentations avec MLFlow.
   - Automatisation des pipelines CI/CD.
   - Surveillance en production avec Azure Application Insights.
3. Déploiement Cloud :
   - Création et déploiement d'une API FastAPI sur Azure Web App.
   - Intégration des tests automatisés et monitoring continu.

## Architecture du Projet

Voici la structure des fichiers et répertoires :

- `.github/workflows/` : Contient le pipeline CI/CD (deploy.yml).
- `api/` : Code source de l'API de prédiction (FastAPI).
- `data/` : Données pour l’entraînement et le test des modèles.
- `models/` : Modèles entraînés (FastText, GloVe, BERT).
- `notebooks/` : Notebooks pour l'exploration des données et la création des modèles.
- `mlruns/` : Dossier MLFlow pour le suivi des expérimentations.
- `docker-compose.yml` : Configuration Docker pour la conteneurisation.

## Prérequis

Pour exécuter le projet, installez les dépendances listées dans le fichier requirements.txt :

pip install -r requirements.txt

Principales dépendances :
- TensorFlow
- FastAPI
- Uvicorn
- Azure SDK
- MLFlow

## Fonctionnalités

### Prédiction des Sentiments

Le projet implémente une API capable de prédire le sentiment d’un tweet (positif ou négatif). Une démo de l’API est disponible publiquement :  
https://p7-myapp-deploy-fchea8befabmcxhk.westeurope-01.azurewebsites.net/docs#/default/feedback_feedback_post

### Approches de Modélisation

Trois approches ont été testées :
1. Modèle simple : Régression logistique.
2. Modèle avancé : Utilisation d’embeddings (FastText, GloVe).
3. Modèle pré-entraîné : BERT.


### Déploiement et Monitoring

- **CI/CD** : GitHub Actions pour déployer automatiquement les modèles sur Azure.
- **Monitoring** : Suivi des performances des modèles avec Azure Application Insights.  
  Cela inclut la configuration de règles pour déclencher une alerte (par email) en cas de trop nombreux tweets mal prédits, par exemple 3 erreurs en l’espace de 5 minutes. Ces alertes permettent une réaction rapide pour ajuster ou retrainer le modèle si nécessaire.


## Données pour l'Entraînement

Pour réaliser l'entraînement des modèles, voici les liens pour télécharger les ressources nécessaires :

1. Base de données des tweets pour l'entraînement :  
   https://www.kaggle.com/datasets/kazanova/sentiment140

2. Embeddings pré-entraînés FastText (crawl-300d-2M-subword.bin) :  
   https://fasttext.cc/docs/en/crawl-vectors.html

3. Embeddings GloVe (glove.twitter.27B.100d.txt) :  
   https://nlp.stanford.edu/projects/glove/

## Exemple de Requêtes

Toutes les interactions avec l'API doivent se faire en anglais. Voici deux exemples illustrant l'utilisation des endpoints `/predict` et `/feedback`.

1. Test de l'API avec un texte simple en anglais, via l'endpoint `/predict` :

{text: "This is a great day!"}

2. Enregistrement du feedback utilisateur, via l'endpoint `/feedback` :

{
text: "This is a great day!",
prediction: "positive",
validation: "true"
}

Lorsque le feedback est posté avec succès, le message suivant est renvoyé par l'API :

{message: "Feedback received, thank you!"}

## Installation et Lancement

Clonez le dépôt :

git clone https://github.com/preudh/OC_IA_P7_analyse_sentiments_deep_Learning.git

Lancez l’API en local :

uvicorn api.main:app --reload

## Utilisation avec Docker

Ce projet utilise **Docker Compose** pour gérer les environnements de développement, d'intégration continue et de production. Trois fichiers Docker Compose sont fournis pour répondre à des besoins spécifiques :

- **docker-compose.dev.yml** :  
  Utilisé pour l’environnement de développement local. Il permet de tester les modifications du code en temps réel grâce à des volumes montés.

- **docker-compose.ci.yml** :  
  Conçu pour les pipelines CI/CD. Ce fichier est optimisé pour exécuter des tests unitaires et des étapes d’intégration continue dans un environnement conteneurisé.

- **docker-compose.prod.yml** :  
  Utilisé pour le déploiement en production. Il configure les services de manière optimisée pour la performance et la sécurité.

### Installer Docker Desktop

Pour exécuter ces configurations Docker Compose en local, vous devez installer **Docker Desktop**, qui fournit une interface graphique pour gérer les conteneurs et les images Docker. Vous pouvez télécharger Docker Desktop ici :  
https://www.docker.com/products/docker-desktop

### Documentation de MLflow 
https://mlflow.org/docs/latest/index.html

### Documentation de FastAPI
https://fastapi.tiangolo.com/

### Documentation de Azure Application Insights
https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview

### Documentation de GitHub Actions
https://docs.github.com/en/actions

### Azure App Service et Déploiement Continu via GitHub
https://learn.microsoft.com/fr-fr/azure/app-service/deploy-continuous-deployment?tabs=github%2Cgithubactions

### Documentation de Docker
https://docs.docker.com/
