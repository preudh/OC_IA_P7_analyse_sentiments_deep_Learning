import os
import json
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

# Chemin vers le modèle décompressé
model_path = os.path.join("/app", "models", "best_model_fasttext.keras")

# Vérification que le modèle existe
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Ensure the model is saved in the correct format.")

# Chargement du modèle
try:
    model = tf.keras.models.load_model(model_path)
except ValueError as e:
    raise ValueError("Error loading model: ensure it is in a compatible format (.keras or .h5).") from e

# Charger la configuration de TextVectorization
config_path = os.path.join("/app", "models", "tv_layer_config.json")
with open(config_path, "r") as file:
    tv_layer_config = json.load(file)

# Créer la couche TextVectorization en utilisant la configuration chargée
tv_layer = tf.keras.layers.TextVectorization.from_config(tv_layer_config)

# Charger le vocabulaire pour la couche TextVectorization
vocab_path = os.path.join("/app", "models", "tv_layer_vocabulary.txt")
with open(vocab_path, "r", encoding="utf-8") as vocab_file:
    vocabulary = [line.strip() for line in vocab_file]

# Définir le vocabulaire dans la couche TextVectorization
tv_layer.set_vocabulary(vocabulary)

# Initialiser l'application FastAPI
app = FastAPI()

# Rediriger la racine vers /docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

# Définir les classes d'entrée pour les prédictions et les retours
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

# Point de terminaison pour les prédictions
@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Vectoriser le texte d'entrée
        sequences = tv_layer([input.text])
        # Prédire le sentiment
        prediction = model.predict(sequences)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        return {"prediction": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Point de terminaison pour le feedback
@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        # Traiter le retour utilisateur
        if not input.validation:
            # Enregistrer ou traiter le feedback négatif si nécessaire
            print(f"Negative feedback received: {input.text}, Prediction: {input.prediction}")
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exécuter l'API
if __name__ == "__main__":
    # Déterminer le port à utiliser
    port = int(os.getenv("PORT", 8000))  # Azure fournit le port via la variable d'environnement PORT
    # Lancer l'application
    uvicorn.run("main:app", host="0.0.0.0", port=port)




