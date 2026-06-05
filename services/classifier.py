from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path("models/content_classifier.joblib")

model = joblib.load(MODEL_PATH)

def predict_category(text: str) -> str:
    print("===== 분류 입력 =====")
    print(text[:500])
    print("===================")

    pred = model.predict([text])[0]

    print("예측:", pred)

    return pred

def predict_category_with_score(text: str):

    pred = model.predict([text])[0]

    scores = model.decision_function([text])
    scores = np.array(scores).reshape(-1)

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    confidence = float(probs.max()) * 100

    return {
        "category": pred,
        "confidence": round(confidence, 1)
    }