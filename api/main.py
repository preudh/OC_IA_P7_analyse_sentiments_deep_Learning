from fastapi import FastAPI, HTTPException
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import uvicorn
import webbrowser
import json
from starlette.responses import RedirectResponse

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

# Rediriger la racine vers /docs
@app.get("/", include_in_schema=False)  # include_in_schema=False pour cacher cette route de la documentation
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

# Définir les classes d'entrée pour les prédictions et la validation
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

# Définir l'endpoint pour les prédictions
@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Vectoriser le texte en utilisant tv_layer
        sequences = tv_layer([input.text])
        # Faire la prédiction
        prediction = model.predict(sequences)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        return {"prediction": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Définir l'endpoint pour recevoir la validation de l'utilisateur
@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        # Traiter le retour de l'utilisateur
        if not input.validation:
            # Par exemple, envoyer une trace à Application Insights ou loguer localement
            print(f"Feedback négatif reçu : {input.text}, Prédiction : {input.prediction}")
        return {"message": "Feedback reçu, merci !"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancer l'API en local si ce script est exécuté directement
if __name__ == "__main__":
    # Ouvrir la documentation directement dans le navigateur
    webbrowser.open("http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

