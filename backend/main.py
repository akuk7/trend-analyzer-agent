from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import run_ai_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Trend Analyzer Backend is running."}

@app.post("/analyze")
def analyze():
    result = run_ai_analysis()
    return {"result": result} 