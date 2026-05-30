from pathlib import Path
import joblib

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

    try:
        confidence = float(scores.max())
    except:
        confidence = 0.0

    return {
        "category": pred,
        "confidence": round(confidence, 3)
    }
