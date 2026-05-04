from pathlib import Path
import joblib

MODEL_PATH = Path("models/content_classifier.joblib")

model = joblib.load(MODEL_PATH)

def predict_category(text: str) -> str:
    prediction = model.predict([text])[0]
    return prediction