import os
import time
import hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "qwen_lora_output"

tokenizer = None
model = None
summary_cache = {}

def load_lora_model():
    global tokenizer, model

    if model is not None:
        return tokenizer, model
    
    start = time.time()
    print("[Qwen LoRA] 모델 로딩 시작")

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
    
    print(f"[Qwen LoRA] 모델 로딩 완료: {time.time() - start:.2f}초")
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
        global summary_cache

        # 같은 입력은 바로 반환
        cache_key = hashlib.md5(
            f"{category}:{text[:900]}".encode("utf-8")
        ).hexdigest()

        if cache_key in summary_cache:
            print("[Qwen LoRA] summary cache hit")
            return summary_cache[cache_key]

        tokenizer, model = load_lora_model()

        short_text = " ".join(text.split())[:700]

        prompt = f"""### 지시문:
너는 유튜브 자막 요약 AI다.
아래 자막을 보고 핵심 키워드와 내용을 아주 짧게 요약해라.

반드시 지켜야 할 규칙:
- 출력은 반드시 한국어만 사용한다.
- 영어 문장, 영어 단어 나열, 원문 복붙을 절대 하지 않는다.
- 자막을 그대로 번역하지 말고 핵심 내용만 요약한다.
- 운동 동작명도 가능하면 한국어로 자연스럽게 설명한다.
- bullet 4~5개로 작성한다.
- 각 bullet은 짧게 작성한다.
- Human:, Assistant:, ### 지시문:, ### 입력: 같은 문구를 출력하지 않는다.
- 불필요한 설명, 사과, 안내 문구를 쓰지 않는다.
- 원문을 그대로 복사하지 않는다.

### 입력:
카테고리: {category}

자막:
{short_text}

### 정답:
"""

        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=450,
        )

        start = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=38,
                max_time=70,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_len:]

        result = tokenizer.decode(generated, skip_special_tokens=True).strip()
        result = clean_lora_output(result)
        
        if not result.strip():
            result = "- 영상의 핵심 내용을 간단히 요약했습니다."

        summary_cache[cache_key] = result

        print(f"[Qwen LoRA] generate 완료: {time.time() - start:.2f}초")
        return result
    
    except Exception as e:
        return f"[Qwen LoRA 요약 실패] {str(e)}"