import pandas as pd
import json

INPUT_PATH = "train/summary_dataset_claude.csv"
OUTPUT_PATH = "train/qwen_summary_train.jsonl"

df = pd.read_csv(INPUT_PATH)

count = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        transcript = str(row.get("transcript", "")).strip()
        summary = str(row.get("summary", "")).strip()

        if transcript == "" or summary == "" or transcript == "nan" or summary == "nan":
            continue

        item = {
            "instruction": "다음 유튜브 자막을 카테고리에 맞게 핵심 요약해줘.",
            "input": f"""카테고리: {row.get('category', '기타')}
제목: {row.get('title', '')}
채널명: {row.get('channel', '')}

자막:
{transcript}""",
            "output": summary
        }

        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1

print(f"변환 완료: {count}개 저장")
print(f"저장 위치: {OUTPUT_PATH}")