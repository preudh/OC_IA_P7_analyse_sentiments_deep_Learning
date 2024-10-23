# from fastapi import FastAPI, HTTPException
# import tensorflow as tf
# from pydantic import BaseModel
# import numpy as np
# import uvicorn
# import webbrowser
#
# # Charger le modèle
# model = tf.keras.models.load_model('models/best_model_fasttext.keras')  # Adapter le chemin si nécessaire
#
# # Définir le nombre de mots et la longueur maximale des séquences
# num_words = 10000
# max_sequence_length = 100
#
# # Initialiser la couche TextVectorization
# tv_layer = tf.keras.layers.TextVectorization(
#     max_tokens=num_words,
#     output_mode='int',
#     output_sequence_length=max_sequence_length
# )
#
# # Adapter la couche tv_layer en utilisant un exemple ou un ensemble de données textuelles préalablement utilisées
# # Utilisez les données textuelles que vous avez utilisées lors de l'entraînement du modèle
# # Exemple: tv_layer.adapt(data['clean_text_embeddings'])
# # Remplacez par un jeu de données approprié si nécessaire
#
# # Initialiser l'API
# app = FastAPI()
#
# # Définir la classe d'entrée pour les prédictions
# class TextInput(BaseModel):
#     text: str
#
# # Définir l'endpoint pour les prédictions
# @app.post("/predict")
# async def predict(input: TextInput):
#     try:
#         # Vectoriser le texte en utilisant tv_layer
#         sequences = tv_layer([input.text])
#         # Faire la prédiction
#         prediction = model.predict(sequences)
#         return {"prediction": "positive" if prediction[0][0] > 0.5 else "negative"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# # Lancer l'API en local si ce script est exécuté directement
# if __name__ == "__main__":
#     # Ouvrir la documentation directement dans le navigateur
#     webbrowser.open("http://127.0.0.1:8000/docs")
#     uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import uvicorn
import webbrowser
import json

# Charger le modèle
model = tf.keras.models.load_model('models/best_model_fasttext.keras')  # Adapter le chemin si nécessaire

# Charger la configuration de TextVectorization
with open("models/tv_layer_config.json", "r") as file:
    tv_layer_config = json.load(file)

# Créer la couche TextVectorization avec la configuration chargée
tv_layer = tf.keras.layers.TextVectorization.from_config(tv_layer_config)

# Charger le vocabulaire
with open("models/tv_layer_vocabulary.txt", "r", encoding="utf-8") as vocab_file:
    vocabulary = [line.strip() for line in vocab_file]

# Adapter le vocabulaire à la couche TextVectorization
tv_layer.set_vocabulary(vocabulary)

# Initialiser l'API
app = FastAPI()

# Définir la classe d'entrée pour les prédictions
class TextInput(BaseModel):
    text: str

# Définir l'endpoint pour les prédictions
@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Vectoriser le texte en utilisant tv_layer
        sequences = tv_layer([input.text])
        # Faire la prédiction
        prediction = model.predict(sequences)
        return {"prediction": "positive" if prediction[0][0] > 0.5 else "negative"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Lancer l'API en local si ce script est exécuté directement
if __name__ == "__main__":
    # Ouvrir la documentation directement dans le navigateur
    webbrowser.open("http://127.0.0.1:8000/docs")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
