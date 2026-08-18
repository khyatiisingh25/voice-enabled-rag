# RAG Design

## Overview

The project uses a Retrieval-Augmented Generation (RAG) pipeline to answer user questions using information retrieved from the available documents.

The pipeline combines:

- Document loading
- Text chunking
- Sentence Transformer embeddings
- FAISS similarity search
- Relevance filtering
- Gemini-based answer generation

## Pipeline Flow

```text
Documents
   ↓
Document Loader
   ↓
Text Chunking
   ↓
Sentence Transformer Embeddings
   ↓
FAISS Vector Index

User Query
   ↓
Query Embedding
   ↓
FAISS Similarity Search
   ↓
Relevance Filtering
   ↓
Relevant Documents?
   ├── NO → grounded=False
   │        sources=[]
   │
   └── YES
        ↓
   Retrieved Context
        ↓
      Gemini
        ↓
answer + sources + grounded=True
