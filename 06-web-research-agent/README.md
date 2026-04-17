# 06 Web Research Agent

Production-style AI agent for Part 6. This project is isolated in its own folder and does not modify the existing lab files.

## What It Includes

- FastAPI REST API
- Conversation history stored in Redis
- API key authentication
- Redis-backed rate limiting: `10 req/min`
- Redis-backed monthly cost guard: `$10/month`
- Health and readiness endpoints
- Graceful shutdown hooks
- Structured JSON logging
- Docker multi-stage build
- Docker Compose stack with `agent + redis + nginx`
- Two agent tools:
  - `search_web` via Serper through LangChain
  - `fetch_webpage` via Crawl4AI
- LLM: `gpt-5-mini`

## Project Structure

```text
06-web-research-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── auth.py
│   ├── config.py
│   ├── cost_guard.py
│   ├── logging_utils.py
│   ├── main.py
│   ├── models.py
│   ├── rate_limiter.py
│   ├── redis_client.py
│   ├── session_store.py
│   └── tools.py
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── railway.toml
├── render.yaml
└── requirements.txt
```

## Environment

Copy `.env.example` to `.env.local`, then fill in:

- `OPENAI_API_KEY`
- `SERPER_API_KEY`
- `AGENT_API_KEY`

## Run Locally Without Docker

```powershell
cd 06-web-research-agent
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m uvicorn app.main:app --reload
```

Start Redis separately, for example:

```powershell
docker run --rm -p 6379:6379 redis:7-alpine
```

## Run Full Stack With Docker

```powershell
cd 06-web-research-agent
docker compose up --build --scale agent=3
```

Nginx is exposed on `http://localhost:8080`.

## API Flow

### 1. Health

```powershell
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

### 2. Ask the agent

```powershell
curl -X POST http://localhost:8080/ask ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: your-secret-key" ^
  -d "{\"question\":\"Find the latest official FastAPI release notes and summarize the key changes.\"}"
```

### 3. Continue the same conversation

Reuse the returned `session_id`:

```json
{
  "question": "Now compare that with the previous release.",
  "session_id": "your-session-id"
}
```

### 4. Get conversation history

```powershell
curl -H "X-API-Key: your-secret-key" http://localhost:8080/sessions/your-session-id
```

## Notes

- `/ready` checks application state and Redis connectivity.
- The agent refuses `/ask` if `OPENAI_API_KEY` or `SERPER_API_KEY` is missing.
- Cost guard uses configurable approximate per-token and per-tool costs from environment variables.
- Crawl4AI requires Playwright Chromium to be installed. The Docker image installs it automatically.

## Deployment

- `railway.toml` is included for Railway.
- `render.yaml` is included for Render with a Redis service.

For Railway, create a Redis service and set:

- `OPENAI_API_KEY`
- `SERPER_API_KEY`
- `AGENT_API_KEY`
- `REDIS_URL`
