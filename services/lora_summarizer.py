import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "qwen_lora_output"

tokenizer = None
model = None


def load_lora_model():
    global tokenizer, model

    if model is not None:
        return tokenizer, model

    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError("qwen_lora_output 폴더가 없습니다.")

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    return tokenizer, model


def clean_lora_output(result: str):
    if "### 정답:" in result:
        result = result.split("### 정답:")[-1].strip()

    stop_words = ["Human:", "Assistant:", "### 지시문:", "### 입력:", "You are", "I want"]
    for word in stop_words:
        if word in result:
            result = result.split(word)[0].strip()

    lines = result.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 영어가 너무 많은 줄 제거
        alpha_count = sum(ch.isalpha() and ch.isascii() for ch in line)
        korean_count = sum('가' <= ch <= '힣' for ch in line)

        if alpha_count > korean_count * 2 and korean_count < 5:
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # 너무 길면 앞 6줄만 사용
    result_lines = result.splitlines()
    result = "\n".join(result_lines[:6]).strip()

    return result


def summarize_with_lora(text: str, category: str = "기타") -> str:
    try:
        tokenizer, model = load_lora_model()

        prompt = f"""### 지시문:
너는 유튜브 자막 요약 AI다.

아래 자막을 보고 핵심만 아주 짧게 요약해라.

반드시 지켜야 할 규칙:
- 출력은 반드시 한국어만 사용한다.
- 영어 문장, 영어 단어 나열, 원문 복붙을 절대 하지 않는다.
- 자막을 그대로 번역하지 말고 핵심 내용만 요약한다.
- 운동 동작명도 가능하면 한국어로 자연스럽게 설명한다.
- bullet 4~6개로 작성한다.
- 각 bullet은 한 문장으로 짧게 쓴다.
- Human:, Assistant:, ### 지시문:, ### 입력: 같은 문구를 출력하지 않는다.
- 불필요한 설명, 사과, 안내 문구를 쓰지 않는다.

### 입력:
카테고리: {category}

자막:
{text[:1000]}

### 정답:
"""

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_lora_output(result)

    except Exception as e:
        return f"[Qwen LoRA 요약 실패] {str(e)}"