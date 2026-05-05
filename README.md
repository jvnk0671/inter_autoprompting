# inter_autoprompting

Web client + FastAPI backend for interactive autoprompting.

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Without `OPENROUTER_API_KEY`, the backend still starts and uses a simple fallback compressor.

## Frontend

```bash
cd client
npm install
npm run dev
```

The frontend sends requests to `http://localhost:8000/optimize`.

## Methods

- `example` — local fallback, no external API.
- `coolprompt` — uses CoolPrompt/OpenRouter when dependencies and key are available.
- `promptomatix` — uses custom/official Promptomatix when dependencies and key are available.
