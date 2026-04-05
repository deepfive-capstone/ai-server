def predict_category(text: str) -> str:
    text = text.lower()

    if "python" in text or "fastapi" in text or "코드" in text:
        return "개발"
    elif "취업" in text or "면접" in text or "자소서" in text:
        return "취업"
    elif "공부" in text or "강의" in text or "학습" in text:
        return "공부"
    else:
        return "기타"
