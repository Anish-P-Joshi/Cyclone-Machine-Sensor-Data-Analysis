# Task2 — RAG Prototype (open-source)

## Overview
Minimal RAG prototype:
- Ingests documents from `docs/` (PDF and TXT)
- Chunks text, computes embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Indexes embeddings in FAISS
- On query: retrieves top-k chunks, passes to `google/flan-t5-small` generator with a citation-aware prompt
- Outputs answer + SOURCES

## Requirements
- Python 3.9+
- ~2-4 GB free disk and a bit of memory (depends on docs)
- Recommended: virtualenv

## Install
```bash
python -m venv venv
source venv/bin/activate      # mac/linux
venv\Scripts\activate.bat     # windows
pip install -U pip
pip install -r requirements.txt
