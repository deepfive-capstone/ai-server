def summarize_text(text: str) -> str:
    return text[:100] + "..." if len(text) > 100 else text
