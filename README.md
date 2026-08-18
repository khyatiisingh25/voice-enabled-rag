# Voice-Enabled RAG

Voice-Enabled RAG system built by Team Zero Signal for Hacker House Goa 2026.

## Overview

The system combines Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), FastAPI, and Gemini to answer questions using information retrieved from project documents.

## Architecture

Voice Input
↓
Speech-to-Text (STT)
↓
Clean Transcript
↓
POST /query
↓
RAG Retrieval
↓
Relevance Filtering
↓
Gemini
↓
answer + sources + grounded

## RAG Pipeline

Documents are loaded from `data/documents/`, split into chunks, converted into embeddings using Sentence Transformers, and indexed using FAISS.

Retrieval uses:

- `top_k = 3`
- `score_threshold = 0.20`

Low-relevance results are rejected.

If no relevant document is found, the system returns `grounded: false`.

## API

### Health Check

```text
GET /health
```

### Query

```text
POST /query
```

Request body:

```json
{
  "query": "What is RAG?"
}
```

## Setup

Create and activate a virtual environment.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the Gemini API key

#### macOS / Linux

```bash
export GEMINI_API_KEY="your-api-key"
```

#### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

### Start the API

```bash
python -m uvicorn app.main:app --reload
```

### Run tests

```bash
python -m pytest -q
```

## End-to-End Flow

STT → /query → RAG → Gemini → answer/sources/grounded
