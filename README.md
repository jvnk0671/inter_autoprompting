```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

```bash
cd client
npm install
npm run dev
```

The frontend sends requests to `http://localhost:8000/optimize`.

