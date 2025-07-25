# AI Trend Analyzer Backend

This folder contains the FastAPI backend server for the AI Trend Analyzer project.

## How to Run

1. Install dependencies (from project root):
   ```sh
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```sh
   uvicorn backend.main:app --reload
   ```

3. The API will be available at `http://127.0.0.1:8000/`.

## Endpoints
- `GET /` — Health check
- `POST /analyze` — (To be implemented) Run the AI analysis pipeline and return results

## Next Steps
- Connect the `/analyze` endpoint to the pipeline in `agent.py`
- Add parameters and error handling as needed 