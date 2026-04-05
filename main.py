from fastapi import FastAPI
from api.classify import router as classify_router

app = FastAPI()

app.include_router(classify_router)

@app.get("/")
def root():
    return {"message": "AI server is running"}
