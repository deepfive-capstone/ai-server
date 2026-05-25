import pandas as pd
import time
from services.summarizer import summarize_text

df = pd.read_csv("train/dataset_cleaned.csv")

if "summary" not in df.columns:
    df["summary"] = ""

for i, row in df.iterrows():
    #if pd.notna(row["summary"]) and str(row["summary"]).strip():
     #   continue

    try:
        summary = summarize_text(row["transcript"], row["category"])
        df.at[i, "summary"] = summary
        print(f"{i} 완료")
    except Exception as e:
        print(f"{i} 에러:", e)

    df.to_csv("train/summary_dataset_claude.csv", index=False, encoding="utf-8-sig")
    time.sleep(2)