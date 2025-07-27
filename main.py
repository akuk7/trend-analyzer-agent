from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pipeline import run_ai_analysis

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
async def analyze(request: Request):
    data = await request.json()
    topic = data.get('topic', 'AI & Machine Learning')
    result = await run_ai_analysis(topic)
    return {"result": result} 