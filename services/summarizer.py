import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CATEGORY_FORMATS = {
    "자기계발": ["핵심 메시지", "실천 포인트", "기대 효과"],
    "운동": ["운동 종류", "방법", "효과", "주의사항"],
    "요리": ["음식명", "핵심 재료", "조리 순서", "팁"],
    "여행": ["장소", "추천 이유", "준비물/비용", "주의사항"],
    "뉴스": ["핵심 사건", "배경", "영향", "한줄 요약"],
    "콘텐츠": ["작품명", "주요 줄거리", "핵심 인물/포인트", "감상 포인트"],
    "기타": ["핵심 내용", "중요한 정보", "한줄 요약"],
}


def build_prompt(text: str, category: str):
    items = CATEGORY_FORMATS.get(category, CATEGORY_FORMATS["기타"])
    item_text = "\n".join([f"- {item}" for item in items])

    return f"""
너는 유튜브 자막 요약 AI야.

아래 자막은 자동 생성 자막이라 반복, 추임새, 오타, 영어 혼합이 있을 수 있어.
불필요한 말은 제거하고 핵심 내용만 정리해줘.

카테고리: {category}

반드시 아래 항목으로 나눠서 요약해줘:
{item_text}

작성 규칙:
- 한국어로 작성
- 자막에 없는 내용은 지어내지 않기
- 각 항목은 1~3문장으로 작성
- 운동과 요리는 방법/순서 항목을 더 자세히 작성
- 내용이 명확하지 않은 항목은 "자막에서 명확히 확인되지 않습니다."라고 작성
- 마크다운 형식으로 보기 좋게 작성

자막:
{text[:20000]}
"""


def summarize_text(text: str, category: str = "기타"):
    if len(text.strip()) < 100:
        return text

    prompt = build_prompt(text, category)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text