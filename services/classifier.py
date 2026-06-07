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
        "cooking recipe ingredients kitchen baking meal prep"
    ),

    "여행": (
        "여행 관광 해외여행 국내여행 여행 코스 숙소 호텔 공항 맛집 일정 관광지 "
        "travel trip hotel airport resort local tour itinerary"
    ),

    "자기계발": (
        "공부 공부법 영어 토익 자격증 취업 코딩 학습 생산성 시간관리 독서 자기관리 성장 커리어 "
        "study learning productivity career certificate coding interview"
    ),

    "콘텐츠": (
        "영화 드라마 예능 게임 웹툰 애니메이션 리뷰 해석 줄거리 결말 분석 캐릭터 콘텐츠 감상 반응 "
        "movie drama entertainment game webtoon animation review reaction story ending"
    ),

    "기타": (
        "특정 주제로 분류하기 어려운 일반 영상 일상 브이로그 먹방 술방 가족 친구 방문 잡담 생활 기록 "
        "여러 주제가 섞인 영상 daily vlog mukbang personal daily life"
    )
}


category_names = list(CATEGORY_DESCRIPTIONS.keys())

category_embeddings = {
    category: embedding_model.encode(description)
    for category, description in CATEGORY_DESCRIPTIONS.items()
}


TITLE_KEYWORDS = {
    "뉴스": [
        "뉴스", "속보", "단독", "긴급", "기자회견", "현장영상",
        "브리핑", "사건", "사고", "채널a", "jtbc", "sbs", "mbc",
        "kbs", "ytn", "연합뉴스"
    ],

    "운동": [
        "운동", "헬스", "홈트", "요가", "필라테스", "러닝",
        "스쿼트", "복근", "하체", "상체", "스트레칭", "근력",
        "배구", "축구", "농구", "야구", "테니스", "골프", "수영",
        "workout", "exercise", "fitness", "training", "sports"
    ],

    "요리": [
        "요리", "레시피", "recipe", "만들기", "재료", "조리",
        "집밥", "도시락", "반찬", "베이킹", "식사", "아침식사",
        "밀프랩", "포케", "덮밥", "파스타", "라면", "김밥",
        "cooking", "ingredients", "kitchen", "baking", "meal prep"
    ],

    "여행": [
        "여행", "관광", "해외여행", "국내여행", "여행코스",
        "숙소", "호텔", "공항", "맛집", "일정", "관광지", "호캉스",
        "travel", "trip", "hotel", "airport", "resort", "tour", "itinerary"
    ],

    "자기계발": [
        "공부", "공부법", "영어", "토익", "자격증", "취업",
        "코딩", "학습", "강의", "생산성", "시간관리",
        "독서", "커리어", "면접", "계획", "투자", "주식",
        "study", "learning", "productivity", "career", "coding", "interview"
    ],

    "콘텐츠": [
        "영화", "드라마", "웹툰", "애니", "애니메이션",
        "게임", "예능", "리뷰", "해석", "줄거리", "결말",
        "결말포함", "결말 포함", "작품", "캐릭터", "강력 추천",
        "movie", "drama", "game", "webtoon", "animation", "review", "ending"
    ],

    "기타": [
        "브이로그", "먹브이로그", "먹방", "술방", "일상",
        "vlog", "daily vlog", "mukbang"
    ]
}


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"[^\w\s가-힣a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def match_title_keywords(title: str):
    """
    제목에서 카테고리별 키워드를 찾는다.
    강제 분류가 아니라 score boost에 사용할 힌트만 만든다.
    """
    title = clean_text(title)
    title_lower = title.lower()

    matched = {}

    for category, words in TITLE_KEYWORDS.items():
        matched_words = []

        for word in words:
            word = word.strip()
            word_lower = word.lower()

            # 영어 키워드: 단어 경계 기준
            if re.fullmatch(r"[a-zA-Z\s]+", word):
                pattern = r"\b" + re.escape(word_lower) + r"\b"

                if re.search(pattern, title_lower):
                    matched_words.append(word)

            # 한국어/혼합 키워드: 포함 여부
            else:
                if word_lower in title_lower:
                    matched_words.append(word)

        if matched_words:
            matched[category] = matched_words
            print(f"[title keyword] {category}:", matched_words)

    return matched


def build_title_boosts(title: str, classes):
    """
    제목 키워드 기반으로 카테고리 점수 보정값을 만든다.
    강제 분류가 아니라 LinearSVC decision score에 더한다.
    """
    matched = match_title_keywords(title)

    boosts = np.zeros(len(classes), dtype=float)

    for i, category in enumerate(classes):
        category = str(category)

        if category not in matched:
            continue

        count = len(matched[category])

        # 기본 boost
        boost = 0.18 * count

        # 제목에 같은 카테고리 신호가 여러 개면 조금 더 강하게 반영
        if count >= 2:
            boost += 0.12

        # 기타는 너무 쉽게 올라가면 위험하므로 약하게
        if category == "기타":
            boost = 0.14 * count
            if count >= 2:
                boost += 0.08

        # 뉴스/콘텐츠는 제목 신호가 비교적 명확한 편이라 살짝 더 줌
        if category in ["뉴스", "콘텐츠"] and count >= 2:
            boost += 0.08

        boosts[i] = boost

    return boosts, matched


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


def predict_category_with_score(text: str, title: str = ""):
    text = clean_text(text)
    title_for_keyword = clean_text(title)

    print("===== 분류 입력 =====")
    print(text[:800])
    print("===================")

    print("===== 제목 키워드 입력 =====")
    print(title_for_keyword)
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
            "title_keywords": {},
            "semantic_pred": None
        }

    raw_pred = model.predict([text])[0]

    raw_scores = model.decision_function([text])
    raw_scores = np.array(raw_scores).reshape(-1)

    classes = model.classes_

    title_boosts, title_keywords = build_title_boosts(title_for_keyword, classes)

    adjusted_scores = raw_scores + title_boosts

    raw_top_indices = np.argsort(raw_scores)[::-1]
    adjusted_top_indices = np.argsort(adjusted_scores)[::-1]

    raw_top1_index = raw_top_indices[0]
    raw_top2_index = raw_top_indices[1]

    adjusted_top1_index = adjusted_top_indices[0]
    adjusted_top2_index = adjusted_top_indices[1]

    raw_top1_class = str(classes[raw_top1_index])
    raw_top2_class = str(classes[raw_top2_index])

    adjusted_top1_class = str(classes[adjusted_top1_index])
    adjusted_top2_class = str(classes[adjusted_top2_index])

    raw_top1_score = float(raw_scores[raw_top1_index])
    raw_top2_score = float(raw_scores[raw_top2_index])

    adjusted_top1_score = float(adjusted_scores[adjusted_top1_index])
    adjusted_top2_score = float(adjusted_scores[adjusted_top2_index])

    raw_margin = raw_top1_score - raw_top2_score
    adjusted_margin = adjusted_top1_score - adjusted_top2_score

    confidence = 1 / (1 + np.exp(-adjusted_margin)) * 100

    # 기준값
    strong_margin_threshold = 0.45
    normal_margin_threshold = 0.25

    semantic_pred = None
    semantic_text = title_for_keyword if len(title_for_keyword) > 0 else text[:1000]

    # 1. 보정 후 margin이 충분하면 보정된 top1 사용
    if adjusted_margin >= normal_margin_threshold:
        pred = adjusted_top1_class
        method = "linear_svc_title_boost"

    # 2. 원래 모델이 아주 강하게 확신하면 원래 모델 결과 유지
    elif raw_margin >= strong_margin_threshold:
        pred = raw_top1_class
        method = "linear_svc_raw_strong"

    # 3. 여기부터는 애매한 경우라서 제목 semantic 사용
    else:
        candidate_categories = [
            str(classes[adjusted_top_indices[0]]),
            str(classes[adjusted_top_indices[1]]),
            str(classes[adjusted_top_indices[2]])
        ]

        semantic_pred = classify_with_sentence_transformer(
            semantic_text,
            candidate_categories=candidate_categories
        )

        print("semantic_text:", semantic_text[:300])
        print("semantic_pred:", semantic_pred)

        # 3-1. 제목 키워드가 있고 semantic 결과가 그 키워드 카테고리와 일치하면 사용
        if len(title_keywords) > 0 and semantic_pred in title_keywords:
            pred = semantic_pred
            method = "title_keyword_semantic_agree"

        # 3-2. 제목 키워드가 있고 semantic이 보정 top1과 같으면 사용
        elif len(title_keywords) > 0 and semantic_pred == adjusted_top1_class:
            pred = adjusted_top1_class
            method = "title_boost_semantic_agree"

        # 3-3. 제목 키워드는 없지만 semantic이 모델 top1과 같고 margin이 너무 낮지는 않으면 사용
        elif len(title_keywords) == 0 and semantic_pred == adjusted_top1_class and adjusted_margin >= 0.15:
            pred = adjusted_top1_class
            method = "model_semantic_agree"

        # 3-4. 그래도 margin이 낮고 서로 확신이 없으면 기타
        elif adjusted_margin < 0.25:
            pred = "기타"
            method = "low_margin_semantic_disagree"

        # 3-5. 마지막 fallback
        else:
            pred = adjusted_top1_class
            method = "adjusted_fallback"

    print("raw_pred:", raw_pred)
    print("raw_top1:", raw_top1_class, round(raw_top1_score, 3))
    print("raw_top2:", raw_top2_class, round(raw_top2_score, 3))
    print("raw_margin:", round(raw_margin, 3))

    print("title_boosts:")
    for i, category in enumerate(classes):
        if title_boosts[i] != 0:
            print(str(category), "+", round(float(title_boosts[i]), 3))

    print("adjusted_top1:", adjusted_top1_class, round(adjusted_top1_score, 3))
    print("adjusted_top2:", adjusted_top2_class, round(adjusted_top2_score, 3))
    print("adjusted_margin:", round(adjusted_margin, 3))
    print("confidence:", round(confidence, 1))
    print("title_keywords:", title_keywords)
    print("semantic_pred:", semantic_pred)
    print("method:", method)
    print("final_pred:", pred)

    return {
        "category": str(pred),
        "confidence": round(confidence, 1),
        "raw_pred": str(raw_pred),
        "top1": adjusted_top1_class,
        "top2": adjusted_top2_class,
        "top1_score": round(adjusted_top1_score, 3),
        "top2_score": round(adjusted_top2_score, 3),
        "margin": round(adjusted_margin, 3),
        "method": method,
        "title_keywords": title_keywords,
        "semantic_pred": semantic_pred,
        "raw_top1": raw_top1_class,
        "raw_top2": raw_top2_class,
        "raw_margin": round(raw_margin, 3)
    }