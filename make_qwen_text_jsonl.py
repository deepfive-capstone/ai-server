import json

INPUT_PATH = "train/qwen_summary_train.jsonl"
OUTPUT_PATH = "train/qwen_summary_train_text.jsonl"

count = 0

with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)

        text = f"""### 지시문:
{data["instruction"]}

### 입력:
{data["input"]}

### 정답:
{data["output"]}"""

        fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        count += 1

print(f"변환 완료: {count}개")