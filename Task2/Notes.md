# RAG + LLM System Design — Notes & Tradeoffs

## Goal
Provide a robust Retrieval-Augmented Generation (RAG) system to answer natural-language queries over 50+ technical PDFs (manuals, SOPs, troubleshooting). Requirements: open-source tools only, cited answers, scalable, and resistant to hallucination.

## High-level architecture
1. Document ingestion
   - Convert PDFs → text (pdfplumber / pypdf). Preprocess (normalize whitespace, remove headers/footers heuristics).
   - Keep simple metadata: filename, page number, chunk_id.

2. Chunking
   - Fixed-token chunking (~200-400 tokens) with overlap (50-100 tokens).
   - Store: chunk_text, metadata {source, page, chunk_id, char_offsets}.

3. Embeddings & Indexing
   - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (small, fast, free).
   - Vector index: FAISS (faiss-cpu) for local dense retrieval.
   - Option: Chroma for convenience / persistence if preferred.

4. Retrieval layer
   - First pass: dense nearest-neighbor (FAISS) top-k (k=8).
   - Reranker: simple lexical score (BM25 or recall of query terms) or cross-encoder (if compute allows) to reorder top-k.
   - Return top 3–5 chunks with metadata for citation.

5. LLM answer generation
   - Free model: `google/flan-t5-small` or a slightly larger HF model if local resources allow.
   - Prompt template includes: user question + retrieved chunks (as quoted snippets and `SOURCE: filename:page:chunk_id`) + instruction to answer and cite sources.
   - Enforce a strict format in prompt (Answer: ... \n SOURCES: [list]).

6. Guardrails & failure modes
   - No relevant content: if retrieval similarity < threshold or returned chunks low lexical overlap → return graceful fallback: “No reliable answer found in the documents. Here are related sections…”
   - Hallucinations: require answer to include explicit SOURCE lines; if model produces content without sources or invents facts beyond snippets, mark untrusted and require operator review.
   - Sensitive queries: block via a simple keyword filter or a small classifier (if required).
   - Monitor: precision@k, faithfulness rate (percentage answers with verifiable source overlap), average retrieval similarity.

7. Scalability
   - 10x docs: reindexing is linear; use sharded FAISS or vector DB (Weaviate/Chroma) with GPU nodes for embeddings.
   - 100+ concurrent users: deploy retrieval as stateless microservice behind autoscaling; generate responses on dedicated inference nodes or using faster small LLMs.
   - Cost: use CPU-only embeddings and small LLMs; use serverless for ingestion and a small cluster for vector search.

## Prototype choices (tradeoffs)
- Embedding model: All-MiniLM is small/fast but less nuanced than larger SBERTs; good for initial retrieval.
- Generator: flan-t5-small is cheap and deterministic-ish; will be less factual than larger models. Enforce citation to mitigate hallucination.
- Vector DB: FAISS (local) keeps everything open-source and simple.

## Evaluation
- Retrieval: precision@k, recall@k on a small labeled Q&A set.
- End-to-end: human or simulated judge to measure faithfulness (does the answer content appear in cited sources?).
- Simple automated faithfulness metric: percent of non-stopwords in answer that appear in source snippets (crude but useful).

## Recommendations for production
1. Add a cross-encoder reranker for top-k to improve precision.
2. Use Llama 2 (7B) or a hosted stronger generator if local compute available — but ensure cost/ops tradeoffs.
3. Logging & monitoring: query patterns, retrieval similarity histograms, hallucination flags, latency percentiles.
4. UI: show top-3 cited snippets with context and highlight text where answer was taken from.
