# ai-server
ai-server/
│
├── main.py                     # FastAPI 서버 시작점
├── requirements.txt            # 필요한 라이브러리 목록
├── .env                        # 비밀키 저장용 (깃허브엔 올리지 않음)
│
├── api/                        # API 라우터 모음
│   ├── summary.py              # 요약 요청 API
│   ├── chat.py                 # 챗봇 요청 API
│   └── classify.py             # 분류 요청 API
│
├── services/                   # 실제 기능 로직
│   ├── crawler.py              # 링크 본문/자막 추출
│   ├── summarizer.py           # 요약 기능
│   ├── classifier.py           # 카테고리 분류 기능
│   └── recommender.py          # 추천 기능(나중용)
│
├── train/                      # 모델 학습 코드
│   └── train_classifier.py     # 분류 모델 학습 코드
│
├── data/                       # 데이터 폴더
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 데이터
│
└── models/                     # 학습된 모델 저장 폴더
    └── category_classifier/    # 분류 모델 저장 위치
