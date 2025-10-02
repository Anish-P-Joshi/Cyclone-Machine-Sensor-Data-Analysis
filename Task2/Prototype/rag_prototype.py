RAG prototype (open-source stack)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Index: FAISS (faiss-cpu)
- Generator: google/flan-t5-small (transformers)
- Ingests PDF/TXT files from docs/, chunks, indexes, answers queries with citations.

Usage:
  python rag_prototype.py --mode index --docs_dir docs/ --index_path ./task2_index/
  python rag_prototype.py --mode query --index_path ./task2_index/ --query "What is X?"
"""

import os
import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import pdfplumber
import nltk
nltk.download('punkt', quiet=True)

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------
# CONFIG / HYPERPARAMS
# -----------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"
CHUNK_SIZE = 350     # approximate tokens (character-based heuristic)
CHUNK_OVERLAP = 80
EMBED_DIM = 384      # all-MiniLM-L6-v2 embedding dim
TOP_K = 6
RERANK_BY_LEXICAL = True
SIM_THRESHOLD = 0.4  # similarity threshold to decide "no relevant content"

# -----------------------
# UTIL: load docs
# -----------------------
def extract_pdf_text(path: str) -> str:
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_text = page_text.replace("\n", " ").strip()
                text.append(page_text)
    except Exception as e:
        print("pdfplumber error:", e)
    return "\n".join(text)

def load_docs_from_dir(docs_dir: str) -> List[Dict]:
    docs = []
    for p in sorted(Path(docs_dir).glob("*")):
        if p.suffix.lower() in [".pdf"]:
            txt = extract_pdf_text(str(p))
        elif p.suffix.lower() in [".txt", ".md"]:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        docs.append({"id": p.name, "text": txt})
    return docs

# -----------------------
# CHUNKING
# -----------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Tuple[str,int,int]]:
    # naive char-based chunker with sentence boundaries
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    cur = ""
    cur_len = 0
    idx = 0
    for s in sentences:
        if cur_len + len(s) <= chunk_size or cur_len == 0:
            cur += " " + s
            cur_len += len(s)
        else:
            # finalize
            start = max(0, idx - len(cur))  # approximate char position (not exact)
            chunks.append((cur.strip(), start, idx))
            # start new chunk with overlap sentences
            # simple overlap strategy: carry last N chars approx
            overlap_text = cur[-overlap:] if overlap < len(cur) else cur
            cur = overlap_text + " " + s
            cur_len = len(cur)
        idx += len(s)
    if cur.strip():
        chunks.append((cur.strip(), 0, idx))
    return chunks

# -----------------------
# EMBEDDINGS + INDEX
# -----------------------
def build_embeddings(texts: List[str], embed_model_name=EMBED_MODEL):
    model = SentenceTransformer(embed_model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs, model

def create_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product with normalized vectors -> cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# -----------------------
# MAIN: index building
# -----------------------
def build_and_save_index(docs_dir: str, index_path: str):
    docs = load_docs_from_dir(docs_dir)
    all_chunks = []
    metadata = []
    for d in docs:
        txt = d['text']
        if not txt.strip():
            continue
        chunks = chunk_text(txt)
        for i, (c, start, end) in enumerate(chunks):
            meta = {"source": d['id'], "chunk_id": i, "start": start, "end": end}
            all_chunks.append(c)
            metadata.append(meta)
    print(f"[index] total chunks: {len(all_chunks)}")
    embs, embedder = build_embeddings(all_chunks)
    # normalize embeddings for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embs)
    index = create_faiss_index(embs)
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "faiss.index"))
    # save metadata and chunks
    with open(os.path.join(index_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    with open(os.path.join(index_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    print("[index] saved to", index_path)
    return

# -----------------------
# Retrieval + Rerank
# -----------------------
def load_index(index_path: str):
    index = faiss.read_index(os.path.join(index_path, "faiss.index"))
    with open(os.path.join(index_path, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(index_path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, chunks, metadata, embedder

def lexical_score(query: str, text: str) -> float:
    # crude lexical overlap: fraction of query tokens present in text
    qtokens = set([t.lower() for t in nltk.word_tokenize(query) if t.isalnum()])
    ttokens = set([t.lower() for t in nltk.word_tokenize(text) if t.isalnum()])
    if not qtokens: return 0.0
    return len(qtokens & ttokens) / len(qtokens)

def retrieve(index, chunks, metadata, embedder, query: str, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    results = []
    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(chunks): continue
        results.append({"chunk": chunks[idx], "meta": metadata[idx], "sim": float(sim)})
    # optional rerank
    if RERANK_BY_LEXICAL:
        for r in results:
            r['lex'] = lexical_score(query, r['chunk'])
        # weighted rerank: give priority to lexical overlap for fidelity
        results = sorted(results, key=lambda r: (0.7*r['sim'] + 0.3*r['lex']), reverse=True)
    return results

# -----------------------
# GENERATION / PROMPTING
# -----------------------
def build_prompt(question: str, retrieved: List[Dict], max_snippets=4) -> str:
    # include top snippets and metadata
    prompt = "You are an assistant that answers questions using only the provided document snippets. "
    prompt += "If the answer cannot be found, say 'No reliable answer found in the documents.'\n\n"
    prompt += f"QUESTION: {question}\n\n"
    prompt += "SOURCES:\n"
    for i, r in enumerate(retrieved[:max_snippets]):
        src = r['meta']
        prompt += f"[{i+1}] SOURCE: {src['source']} | chunk_id: {src['chunk_id']}\n{r['chunk']}\n\n"
    prompt += "INSTRUCTIONS: Answer concisely. At the end, include a SOURCES line listing the numbered sources used (e.g. SOURCES: [1,3]).\nAnswer:\n"
    return prompt

def generate_answer(prompt: str, gen_model_name=GEN_MODEL, max_length=200):
    tok = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    outputs = model.generate(input_ids, max_length=max_length, num_beams=3, early_stopping=True)
    ans = tok.decode(outputs[0], skip_special_tokens=True)
    return ans

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "query"], required=True)
    parser.add_argument("--docs_dir", type=str, default="docs/")
    parser.add_argument("--index_path", type=str, default="./task2_index/")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "index":
        build_and_save_index(args.docs_dir, args.index_path)
    else:
        assert args.query is not None, "Provide --query for query mode"
        index, chunks, metadata, embedder = load_index(args.index_path)
        retrieved = retrieve(index, chunks, metadata, embedder, args.query, top_k=TOP_K)
        if not retrieved:
            print("No retrieval results.")
            return
        if retrieved[0]['sim'] < SIM_THRESHOLD:
            print("No reliable content found in documents (low similarity). Returning top snippets instead.")
        prompt = build_prompt(args.query, retrieved, max_snippets=4)
        ans = generate_answer(prompt)
        print("=== ANSWER ===")
        print(ans)
        print("\n=== TOP SOURCES ===")
        for i, r in enumerate(retrieved[:4]):
            print(f"[{i+1}] {r['meta']['source']} (sim={r['sim']:.3f}, lexic_overlap={r.get('lex', None)})")

if __name__ == "__main__":
    main()

