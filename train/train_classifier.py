import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = Path('dataset_cleaned.csv')
MODEL_PATH = Path('content_classifier.joblib')

# CSV 로드: utf-8-sig는 엑셀/윈도우에서도 한글 깨짐이 적음
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

# 학습에 필요한 컬럼만 사용
# title + transcript를 같이 쓰면 짧은 transcript에서도 분류 힌트가 늘어남
df['text'] = (
    df['title'].fillna('').astype(str)
    + '\n'
    + df['transcript'].fillna('').astype(str)
)

# 혹시 빈 텍스트가 있으면 제거
df = df[df['text'].str.strip().str.len() > 0].copy()

X = df['text']
y = df['category']

# 현재 파일의 split이 전부 train이라서, 모델 검증용으로 임시 train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )),
    ('clf', LinearSVC(random_state=42))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print(f'모델 저장 완료: {MODEL_PATH}')

# 사용 예시
sample = '하체 운동 루틴과 스쿼트 자세, 근력 향상 방법을 설명하는 영상입니다.'
print('예측 예시:', model.predict([sample])[0])

