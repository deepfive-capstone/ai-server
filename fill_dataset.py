import pandas as pd
import requests
import time

INPUT_FILE = "youtube_category_dataset_filled.xlsx"
OUTPUT_FILE = "youtube_category_dataset_filled.xlsx"
API_URL = "http://127.0.0.1:8000/transcript"

df = pd.read_excel(INPUT_FILE)

for idx, row in df.iterrows():
    # 1~18번은 이미 채웠으니까 건너뛰기
    if idx < 18:
        continue

    url = row.get("url")

    if pd.isna(url) or not str(url).strip():
        continue

    existing_transcript = row.get("transcript")
    if pd.notna(existing_transcript) and str(existing_transcript).strip():
        continue

    print(f"{idx + 1}번 처리 중: {url}")

    try:
        response = requests.get(API_URL, params={"url": url}, timeout=120)
        data = response.json()

        if "error" in data:
            print("ERROR:", data["error"])
            df.loc[idx, "notes"] = f"ERROR: {data['error']}"
            df.to_excel(OUTPUT_FILE, index=False)
            time.sleep(30)
            continue

        transcript = data.get("transcript", "")

        print("title:", data.get("title"))
        print("transcript length:", len(transcript))

        df.loc[idx, "video_id"] = str(data.get("video_id", ""))
        df.loc[idx, "title"] = str(data.get("title", ""))
        df.loc[idx, "channel"] = str(data.get("channel", ""))
        df.loc[idx, "thumbnail"] = str(data.get("thumbnail", ""))
        df.loc[idx, "transcript"] = transcript
        df.loc[idx, "notes"] = ""

        df.to_excel(OUTPUT_FILE, index=False)
        time.sleep(30)

    except Exception as e:
        print("EXCEPTION:", str(e))
        df.loc[idx, "notes"] = f"ERROR: {str(e)}"
        df.to_excel(OUTPUT_FILE, index=False)
        time.sleep(30)

print(f"완료: {OUTPUT_FILE}")