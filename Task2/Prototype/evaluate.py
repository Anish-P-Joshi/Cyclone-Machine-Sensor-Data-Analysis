"""
evaluate.py
Simple evaluation runner for RAG prototype.
Generates evaluation.csv with basic retrieval + faithfulness metrics.
"""

import csv
import nltk
from rag_prototype import load_index, retrieve, build_prompt, generate_answer

nltk.download('punkt', quiet=True)

QUERIES = [
    "What does a sudden draft drop indicate?",
    "What is the safe outlet gas temperature range?",
    "How often does shutdown occur?",
    "Steps for checking inlet valve blockage",
    "Why does material temperature spike suddenly?"
]

def lexical_overlap(text1, text2):
    """ crude lexical overlap ratio """
    t1 = set([t.lower() for t in nltk.word_tokenize(text1) if t.isalnum()])
    t2 = set([t.lower() for t in nltk.word_tokenize(text2) if t.isalnum()])
    if not t1: return 0.0
    return len(t1 & t2) / len(t1)

def run_eval(index_path="./task2_index/", out_file="../evaluation.csv"):
    index, chunks, metadata, embedder = load_index(index_path)
    rows = []
    for q in QUERIES:
        retrieved = retrieve(index, chunks, metadata, embedder, q, top_k=5)
        if not retrieved:
            rows.append([q, "-", 0.0, 0.0, "No", "No relevant content"])
            continue

        # crude metrics
        retrieved_chunks = [f"{r['meta']['source']}[{r['meta']['chunk_id']}]" for r in retrieved]
        retrieved_texts = " ".join([r['chunk'] for r in retrieved])
        prompt = build_prompt(q, retrieved)
        ans = generate_answer(prompt)

        # lexical overlap of answer with retrieved text
        overlap = lexical_overlap(ans, retrieved_texts)

        # dummy heuristic for precision/recall
        precision = 1.0 if overlap > 0.3 else 0.5 if overlap > 0.1 else 0.0
        recall = 1.0 if len(retrieved) >= 3 else 0.5 if len(retrieved) > 0 else 0.0
        faithful = "Yes" if overlap > 0.3 else "Partial" if overlap > 0.1 else "No"

        rows.append([q, "; ".join(retrieved_chunks), precision, recall, faithful, ans[:80] + "..."])

    # save CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "retrieved_chunks", "precision@5", "recall@5", "faithfulness", "notes"])
        writer.writerows(rows)

    print(f"[eval] saved results to {out_file}")

if __name__ == "__main__":
    run_eval()
