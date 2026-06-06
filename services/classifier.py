from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import re

MODEL_PATH = Path("models/content_classifier.joblib")

model = joblib.load(MODEL_PATH)

embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")


CATEGORY_DESCRIPTIONS = {
    "뉴스": (
        "정치 경제 사회 국제 사건 사고 정책 이슈 보도 속보 뉴스 리포트 "
        "언론사에서 다루는 시사 정보"
    ),

    "운동": (
        "운동 헬스 홈트 요가 필라테스 러닝 스트레칭 근력운동 유산소 "
        "스포츠 배구 축구 농구 야구 테니스 골프 수영 자세 기본기 훈련 기술 연습 경기력 향상 "
        "workout exercise fitness training sports volleyball soccer basketball running stretch"
    ),

    "요리": (
        "요리 레시피 음식 만들기 재료 손질 조리 과정 집밥 도시락 반찬 베이킹 식사 준비 "
        "cooking cook recipe food meal ingredients kitchen bake baking dish meal prep lunch dinner"
    ),

    "여행": (
        "여행 관광 해외여행 국내여행 여행 브이로그 여행 코스 숙소 호텔 공항 맛집 일정 관광지 "
        "travel trip vlog hotel airport resort local tour itinerary"
    ),

    "자기계발": (
        "공부 공부법 영어 토익 자격증 취업 코딩 학습 생산성 시간관리 독서 자기관리 성장 커리어 "
        "study learning productivity career certificate coding interview"
    ),

    "콘텐츠": (
        "영화 드라마 예능 게임 웹툰 리뷰 해석 줄거리 결말 분석 캐릭터 콘텐츠 감상 반응 "
        "movie drama game review reaction story ending"
    ),

    "기타": (
        "특정 주제로 분류하기 어려운 일반 영상 일상 잡담 생활 정보 여러 주제가 섞인 영상"
    )
}


category_names = list(CATEGORY_DESCRIPTIONS.keys())

category_embeddings = {
    category: embedding_model.encode(description)
    for category, description in CATEGORY_DESCRIPTIONS.items()
}


MIXED_TOPIC_GROUPS = {
    "운동": [
        "운동", "헬스", "홈트", "요가", "필라테스", "러닝",
        "스포츠", "배구", "축구", "농구", "야구", "테니스",
        "골프", "수영", "근력", "유산소", "스트레칭",
        "스쿼트", "복근", "하체", "상체", "리시브", "서브",
        "스파이크", "블로킹", "훈련", "기본기",
        "workout", "exercise", "fitness", "training", "sports",
        "volleyball", "soccer", "basketball", "running", "stretch"
    ],

    "요리": [
        "요리", "레시피", "음식", "재료", "조리", "집밥",
        "도시락", "반찬", "베이킹", "먹방", "만들기",
        "식사", "한끼", "볶음밥", "파스타", "라면", "김밥",
        "밀프랩", "포케", "덮밥", "장조림", "라따뚜이",
        "cook", "cooking", "recipe", "food", "meal",
        "ingredients", "kitchen", "bake", "baking", "dish",
        "meal prep", "lunch", "dinner"
    ],

    "자기계발": [
        "공부", "공부법", "영어", "토익", "자격증", "취업",
        "코딩", "학습", "강의", "생산성", "시간관리",
        "독서", "커리어", "면접", "계획",
        "study", "learning", "productivity", "career",
        "certificate", "coding", "interview"
    ],

    "여행": [
        "여행", "관광", "숙소", "호텔", "공항", "일정",
        "해외여행", "국내여행", "도쿄", "오사카", "일본",
        "제주", "부산", "시드니", "방콕", "싱가포르",
        "롬복", "인도네시아", "발리", "동남아", "호캉스",
        "travel", "trip", "vlog", "hotel", "airport",
        "resort", "tour", "itinerary"
    ]
}


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"[^\w\s가-힣a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_mixed_topic(text: str):
    """
    여러 주제가 섞인 영상인지 감지한다.
    한 주제당 키워드 2개 이상 잡힐 때만 해당 주제로 인정한다.
    """
    hit_topics = []

    for topic, words in MIXED_TOPIC_GROUPS.items():
        count = sum(1 for word in words if word in text)

        if count >= 2:
            hit_topics.append(topic)

    is_mixed = len(hit_topics) >= 2

    return is_mixed, hit_topics


def classify_with_sentence_transformer(text: str, candidate_categories=None) -> str:
    """
    의미 기반 분류.
    전체 카테고리에서 마음대로 고르는 용도가 아니라,
    LinearSVC가 뽑은 후보 안에서 재판정하는 용도.
    """
    text_embedding = embedding_model.encode(text)

    if candidate_categories is None:
        candidate_categories = category_names

    similarities = []

    for category in candidate_categories:
        category_embedding = category_embeddings[category]

        similarity = np.dot(text_embedding, category_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(category_embedding)
        )

        similarities.append(float(similarity))

    best_index = int(np.argmax(similarities))

    return candidate_categories[best_index]


def predict_category(text: str) -> str:
    result = predict_category_with_score(text)
    return result["category"]


def predict_category_with_score(text: str):
    text = clean_text(text)

    print("===== 분류 입력 =====")
    print(text[:800])
    print("===================")

    if len(text.strip()) < 10:
        return {
            "category": "기타",
            "confidence": 0.0,
            "raw_pred": "기타",
            "top1": "기타",
            "top2": None,
            "top1_score": 0.0,
            "top2_score": 0.0,
            "margin": 0.0,
            "method": "empty_text",
            "mixed_topics": []
        }

    raw_pred = model.predict([text])[0]

    scores = model.decision_function([text])
    scores = np.array(scores).reshape(-1)

    classes = model.classes_

    top_indices = np.argsort(scores)[::-1]

    top1_index = top_indices[0]
    top2_index = top_indices[1]

    top1_score = float(scores[top1_index])
    top2_score = float(scores[top2_index])

    top1_class = str(classes[top1_index])
    top2_class = str(classes[top2_index])

    margin = top1_score - top2_score

    # LinearSVC는 확률이 아니므로 top1과 top2 점수 차이로 confidence 계산
    confidence = 1 / (1 + np.exp(-margin)) * 100

    # 단일/복합 주제 감지
    is_mixed, mixed_topics = detect_mixed_topic(text)

    # 1. raw_pred가 기타이거나 top1_score가 낮은데,
    #    하나의 주제만 강하게 감지되면 그 주제로 보정
    #    예: 밀프랩/레시피/재료만 많이 나오면 요리
    if len(mixed_topics) == 1 and (str(raw_pred) == "기타" or top1_score < 0):
        pred = mixed_topics[0]
        method = "single_topic_detected"

    # 2. top1_score가 음수이고 단일 주제도 감지되지 않으면 기타
    elif top1_score < 0:
        pred = "기타"
        method = "low_absolute_score"

    # 3. LinearSVC가 어느 정도 확신하면 그대로 사용
    #    기준은 사용자가 원한 대로 50
    elif confidence >= 50:
        pred = raw_pred
        method = "linear_svc"

    # 4. 모델도 애매한데 여러 주제가 섞였으면 기타
    elif is_mixed:
        pred = "기타"
        method = "mixed_topic"

    # 5. 마지막으로 애매한 경우에만 semantic_top3 사용
    else:
        candidate_categories = [
            str(classes[top_indices[0]]),
            str(classes[top_indices[1]]),
            str(classes[top_indices[2]])
        ]

        # 기타는 low_absolute_score / mixed_topic에서 따로 처리하므로
        # semantic 후보에서는 가능하면 제거
        non_etc_candidates = [
            category for category in candidate_categories
            if category != "기타"
        ]

        if len(non_etc_candidates) >= 2:
            candidate_categories = non_etc_candidates

        semantic_pred = classify_with_sentence_transformer(
            text,
            candidate_categories=candidate_categories
        )

        pred = semantic_pred
        method = "semantic_top3"

    print("raw_pred:", raw_pred)
    print("top1:", top1_class, round(top1_score, 3))
    print("top2:", top2_class, round(top2_score, 3))
    print("margin:", round(margin, 3))
    print("confidence:", round(confidence, 1))
    print("mixed_topics:", mixed_topics)
    print("method:", method)
    print("final_pred:", pred)

    return {
        "category": str(pred),
        "confidence": round(confidence, 1),
        "raw_pred": str(raw_pred),
        "top1": top1_class,
        "top2": top2_class,
        "top1_score": round(top1_score, 3),
        "top2_score": round(top2_score, 3),
        "margin": round(margin, 3),
        "method": method,
        "mixed_topics": mixed_topics
    }