# SmartGrid AI Backend

This is the FastAPI backend for the SmartGrid AI application, providing AI-powered data enrichment capabilities using OpenAI's GPT-3.5 Turbo.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Server

Start the development server:
```bash
uvicorn app.main:app --reload
```

The server will run on `http://localhost:8000`

## API Endpoints

### POST /api/enrich

Enriches spreadsheet data using OpenAI's GPT-3.5 Turbo.

Request body:
```json
{
    "rows": [
        {"column1": "value1", "column2": "value2"},
        {"column1": "value3", "column2": "value4"}
    ],
    "prompt": "Your enrichment prompt here"
}
```

Response:
```json
{
    "enriched_data": [
        {"column1": "value1", "column2": "value2", "ai_result": "enriched value"},
        {"column1": "value3", "column2": "value4", "ai_result": "enriched value"}
    ],
    "error": null
}
```

## Features

- Uses OpenAI's GPT-3.5 Turbo for AI enrichment
- Automatic retry logic for failed API calls
- Error handling and validation
- CORS enabled for frontend integration 