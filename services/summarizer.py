import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY")

print("[Gemini key prefix]", key[:8])
print("[Gemini key length]", len(key))

client = genai.Client(api_key=key)

CATEGORY_FORMATS = {
    "자기계발": "핵심 메시지와 실천 포인트 중심",
    "운동": "운동 순서와 동작 핵심 중심",
    "요리": "조리 순서와 핵심 재료 중심",
    "여행": "장소 특징과 추천 이유 중심",
    "뉴스": "핵심 사건과 영향 중심",
    "콘텐츠": "주요 내용과 핵심 포인트 중심",
    "기타": "핵심 내용 중심",
}


def get_format_rule(category: str) -> str:
    if category == "운동":
        return """
출력 형식:
## 핵심 요약
1. 핵심 동작 - 짧은 설명
2. 핵심 동작 - 짧은 설명
3. 핵심 동작 - 짧은 설명

규칙:
- 운동 순서가 있으면 반드시 1, 2, 3 번호 형식으로 작성
- 각 줄은 "번호. 키워드 - 짧은 설명" 형식
- 긴 문장 금지
- 동작 설명은 핵심만 짧게
"""
    elif category == "요리":
        return """
출력 형식:
## 핵심 요약
1. 핵심 과정 - 짧은 설명
2. 핵심 과정 - 짧은 설명
3. 핵심 과정 - 짧은 설명

규칙:
- 조리 순서가 있으면 반드시 1, 2, 3 번호 형식으로 작성
- 각 줄은 "번호. 키워드 - 짧은 설명" 형식
- 재료 준비, 조리, 마무리 순서가 드러나게 정리
- 긴 문장 금지
"""
    else:
        return """
출력 형식:
## 핵심 요약
- 핵심 키워드: 짧은 설명
- 핵심 키워드: 짧은 설명
- 핵심 키워드: 짧은 설명

규칙:
- bullet 형식으로 작성
- 키워드 중심으로 정리
- 각 bullet은 한 줄로 짧게 작성
- 긴 문장 금지
"""


def build_prompt(text: str, category: str):
    guide = CATEGORY_FORMATS.get(category, CATEGORY_FORMATS["기타"])
    format_rule = get_format_rule(category)

    return f"""
너는 유튜브 자막 요약 AI야.

아래 자막을 카테고리에 맞게 짧고 핵심만 요약해줘.

카테고리: {category}
요약 방향: {guide}

공통 작성 규칙:
- 반드시 한국어로만 작성
- 반드시 "## 핵심 요약" 제목 사용
- 자막에 없는 내용은 지어내지 않기
- 자막 원문을 그대로 번역하지 말고 핵심만 요약
- 설명형 긴 문장보다 키워드 중심으로 작성
- 불필요한 설명, 인사말, 안내문 금지
- 출력은 최종 요약만 작성

{format_rule}

자막:
{text[:12000]}
"""


def summarize_text(text: str, category: str = "기타"):
    if not isinstance(text, str) or len(text.strip()) < 100:
        return text

    prompt = build_prompt(text, category)

    import time
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2)



def improve_summary_with_gemini(original_text: str, qwen_summary: str, category: str):
    guide = CATEGORY_FORMATS.get(category, CATEGORY_FORMATS["기타"])
    format_rule = get_format_rule(category)

    prompt = f"""
다음은 Qwen LoRA 모델이 생성한 유튜브 자막 요약이야.
Qwen 요약을 바탕으로 부족한 표현만 보완해서 최종 요약을 만들어줘.

카테고리: {category}
요약 방향: {guide}

Qwen 요약:
{qwen_summary}

원본 자막 일부:
{original_text[:6000]}

공통 작성 규칙:
- 반드시 한국어로만 작성
- 반드시 "## 핵심 요약" 제목 사용
- 자막에 없는 내용 추가 금지
- Qwen 요약이 이상하면 원본 자막 기준으로 자연스럽게 보완
- 자막 원문을 그대로 번역하지 말고 핵심만 요약
- "~합니다", "~해요", "~입니다", "~할 수 있습니다" 같은 서술형 문장 사용 금지
- 키워드만 나열하지 말고, 사용자가 내용을 이해할 수 있도록 짧은 설명을 함께 작성
- 너무 짧게 줄이지 말고, 핵심 내용을 5~8개 항목으로 정리
- 각 항목은 한 문장으로 자연스럽게 작성
- 불필요한 설명, 인사말, 안내문 금지
- 출력은 최종 요약만 작성

{format_rule}
"""

    import time
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2)