Meno-Core

Meno-Core is a small OpenAI-compatible backend that adds a Retrieval-Augmented Generation (RAG) layer, abbreviation handling and link suggestions on top of a chat completion API.

Project structure
- `src/meno_core/api/main.py` – FastAPI application exposing an OpenAI-style `POST /v1/chat/completions` endpoint (non-streaming and streaming)
- `src/meno_core/core/rag_engine.py` – RAG pipeline, LLM calls, abbreviation and anaphora resolution
- `src/meno_core/core/link_searcher.py` – selects and formats relevant links from `resources/validated_urls.json`
- `src/meno_core/core/link_correcter.py` – normalizes and fixes selected URLs
- `src/meno_core/config/settings.py` – configuration via environment variables
- `src/meno_core/infrastructure/logdb` – optional PostgreSQL logging of conversations
- `resources/` – abbreviations and validated URLs
- `scripts/run_backend.sh` – helper script to run the backend

Main features
- OpenAI Chat Completions–compatible HTTP API (request and response schema)
- Retrieval-Augmented Generation over a local knowledge base
- Automatic handling of abbreviations and anaphora in user questions
- Optional enrichment of answers with a short list of “interesting links”
- Optional logging of conversations to PostgreSQL

Requirements
- Python 3.12 or newer
- Access to an OpenAI-compatible LLM endpoint (or a self-hosted equivalent)
- Local embedding and reranker models (paths configured via environment variables)
- Optional: running PostgreSQL instance for logging

Installation
- Clone the repository
- Create and activate a virtual environment
- Install dependencies (using pip or uv)

Example with uv:
- Install uv (if needed): `pip install uv`
- From the project root: `uv sync`

Configuration
- Copy `example.env` to `.env`
- Adjust values to your environment, in particular:
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL_NAME`
  - `LOCAL_EMBEDDER_PATH`, `LOCAL_RERANKER_PATH`
  - `WORKING_DIR`
  - `ABBREVIATIONS_FILE` (for example `resources/full_abbreviations_updated.json`)
  - `URLS_PATH` (for example `resources/validated_urls.json`)
- Optionally tune RAG and link settings via variables defined in `src/meno_core/config/settings.py`

Running the backend
- Using uvicorn directly:
  - From the project root: `uv run uvicorn meno_core.api.main:app --host 127.0.0.1 --port 8888`
- Or using the helper script:
  - From the project root: `bash scripts/run_backend.sh`

Basic usage
- Base URL: `http://127.0.0.1:8888`
- Endpoint: `POST /v1/chat/completions`
- Request body (simplified):
  - `model`: string
  - `messages`: list of objects with `role` (`system`, `user`, `assistant`) and `content` (string)
  - optional: `stream` (boolean) to enable server-sent events
- Response shape follows the OpenAI Chat Completions API (choices with `message.content`)

Logging
- If configured, conversations are stored in PostgreSQL using SQLAlchemy models in `src/meno_core/infrastructure/logdb`
- Connection parameters can be adjusted in the logging configuration and environment variables

License
- Add a license statement here if you plan to distribute this project

