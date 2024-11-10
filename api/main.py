from fastapi import FastAPI, HTTPException
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import uvicorn
import webbrowser
import json
from starlette.responses import RedirectResponse
import os

# For GitHub Actions, we need to set the model path to the correct location
# Determine if running in GitHub Actions
is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"

# Define paths for the model in both zipped and regular formats
zipped_model_path = os.path.join("/app", "models", "best_model_fasttext.keras.zip")
regular_model_path = os.path.join("/app", "models", "best_model_fasttext.keras")

# Load the model based on the available format
if os.path.exists(zipped_model_path):
    print("Loading model from zipped format...")
    model = tf.keras.models.load_model(zipped_model_path)
elif os.path.exists(regular_model_path):
    print("Loading model from regular format...")
    model = tf.keras.models.load_model(regular_model_path)
else:
    raise FileNotFoundError("Model file not found. Ensure the model is saved in the correct format.")

# Load TextVectorization configuration
config_path = os.path.join("/app", "models", "tv_layer_config.json")
with open(config_path, "r") as file:
    tv_layer_config = json.load(file)

# Create the TextVectorization layer using the loaded configuration
tv_layer = tf.keras.layers.TextVectorization.from_config(tv_layer_config)

# Load vocabulary for TextVectorization layer
vocab_path = os.path.join("/app", "models", "tv_layer_vocabulary.txt")
with open(vocab_path, "r", encoding="utf-8") as vocab_file:
    vocabulary = [line.strip() for line in vocab_file]

# Set vocabulary in the TextVectorization layer
tv_layer.set_vocabulary(vocabulary)

# Initialize FastAPI application
app = FastAPI()

# Redirect root to /docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

# Define input classes for predictions and feedback
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

# Prediction endpoint
@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Vectorize the input text
        sequences = tv_layer([input.text])
        # Predict sentiment
        prediction = model.predict(sequences)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        return {"prediction": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoint
@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        # Process user feedback
        if not input.validation:
            # Log or handle negative feedback as needed
            print(f"Negative feedback received: {input.text}, Prediction: {input.prediction}")
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API locally if the script is executed directly
if __name__ == "__main__":
    # Open documentation in the browser
    webbrowser.open("http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


