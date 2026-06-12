from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import re

MODEL_PATH = Path("models/content_classifier.joblib")

model = joblib.load(MODEL_PATH)

embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
CATEGORY_DESCRIPTIONS = {
    "뉴스": "정치 경제 사회 국제 이슈 사건 사고 최신 뉴스 보도",
    "운동": "홈트 헬스 다이어트 전신 운동 유산소 근력 스트레칭 요가 필라테스",
    "요리": "요리 레시피 재료 조리 과정 집밥 도시락 밀프렙 음식 만들기",
    "여행": "여행 브이로그 관광지 숙소 맛집 해외 국내 여행 코스",
    "자기계발": "영어 공부 토익 자격증 취업 코딩 공부 대학 공부 학습 교육 강의",
    "콘텐츠": "일상 브이로그 생산적인 하루 미라클모닝 루틴 챌린지 리뷰 예능 게임 먹방",
    "기타": "특정 카테고리로 분류하기 어려운 일반 영상"
}

category_embeddings = embedding_model.encode(
    list(CATEGORY_DESCRIPTIONS.values())
)

category_names = list(CATEGORY_DESCRIPTIONS.keys())


def classify_with_sentence_transformer(text: str) -> str:
    text_embedding = embedding_model.encode(text)

    similarities = []

    for category, category_embedding in zip(category_names, category_embeddings):
        similarity = np.dot(text_embedding, category_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(category_embedding)
        )
        similarities.append(float(similarity))

    best_index = int(np.argmax(similarities))
    return category_names[best_index]

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s가-힣a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_category(text: str) -> str:
    text = clean_text(text)

    print("===== 분류 입력 =====")
    print(text[:500])
    print("===================")

    pred = model.predict([text])[0]
    pred = apply_keyword_override(text, pred)

    print("예측:", pred)

    return pred


def predict_category_with_score(text: str):
    text = clean_text(text)

    pred = model.predict([text])[0]
    pred = apply_keyword_override(text, pred)

    scores = model.decision_function([text])
    scores = np.array(scores).reshape(-1)

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    confidence = float(probs.max()) * 100
    
    if confidence < 40:
        pred = classify_with_sentence_transformer(text)

    return {
        "category": pred,
        "confidence": round(confidence, 1)
    }

def apply_keyword_override(text: str, pred: str) -> str:
    exercise_keywords = [
        "운동", "다이어트", "전신", "홈트", "헬스", "근력",
        "유산소", "스트레칭", "스쿼트", "복근", "하체", "상체",
        "칼로리", "체지방", "필라테스", "요가"
    ]

    if any(keyword in text for keyword in exercise_keywords):
        return "운동"

    return pred

def apply_keyword_override(text: str, pred: str) -> str:
    cooking_action_keywords = [
        "요리", "레시피", "만들기", "조리", "밀프렙",
        "도시락", "집밥", "식단", "초간단", "반찬"
    ]

    food_keywords = [
        "분짜", "소시지빵", "덮밥", "국밥", "수육",
        "김밥", "파스타", "라면", "볶음밥", "샐러드",
        "빵", "밥", "한끼"
    ]

    content_keywords = [
        "먹방", "리뷰", "브이로그", "vlog", "챌린지",
        "맛집", "먹어봤", "먹어보", "리액션"
    ]

    cooking_action_score = sum(1 for k in cooking_action_keywords if k in text)
    food_score = sum(1 for k in food_keywords if k in text)
    content_score = sum(1 for k in content_keywords if k in text)

    # 먹방/리뷰/브이로그 성격이면 콘텐츠 우선
    if content_score >= 1 and cooking_action_score == 0:
        return "콘텐츠"

    # 요리 행동 단어가 있고 음식 키워드도 있으면 요리
    if cooking_action_score >= 1 and food_score >= 1:
        return "요리"

    # 밀프렙/도시락/식단은 요리 의도가 강함
    if any(k in text for k in ["밀프렙", "도시락", "집밥", "식단"]):
        return "요리"

    return pred
