# Cyclone Machine Sensor Data Analysis & RAG System Design

## Overview  
This repository contains my submission for a two-part data science take-home assignment:  

- **Task 1:** Machine Data Analysis (Time-series)  
- **Task 2:** RAG + LLM System Design  

Together, they demonstrate end-to-end capabilities across data engineering, anomaly detection, forecasting, NLP system design, and scalable AI architecture.

---

## Task 1: Machine Data Analysis (Time-series)

### Objective  
Analyze 3 years of cyclone machine sensor data (~370k records at 5-minute intervals) to detect shutdowns, segment operational states, find anomalies, and forecast inlet gas temperature.  

### Approach  
1. **Data Preparation**  
   - Handled missing values, timestamp gaps, and outliers  
   - Enforced strict 5-minute indexing  
   - Computed summary statistics and correlation matrix  

2. **Shutdown Detection**  
   - Programmatically identified idle periods  
   - Computed downtime statistics (frequency and duration)  
   - Year-long visualization with shutdowns highlighted  

3. **Operational State Segmentation (Clustering)**  
   - KMeans clustering on multivariate sensor features  
   - Derived interpretable states (Normal, Startup, High Load, Degraded)  
   - Generated cluster-wise statistics and persistence patterns  

4. **Contextual Anomaly Detection**  
   - State-aware Isolation Forest and rolling MAD thresholds  
   - Consolidated anomalous events (start, end, implicated variables)  
   - Root-cause hypotheses based on variable correlations  

5. **Short-Horizon Forecasting**  
   - Forecasted `Cyclone_Inlet_Gas_Temp` for 1 hour (12 steps)  
   - Compared Persistence, ARIMA, and Prophet models  
   - Evaluated with RMSE and MAE  

6. **Insights and Recommendations**  
   - Shutdowns often preceded by anomalies in degraded states  
   - Higher anomaly density in certain operational regimes  
   - Forecasting effective in stable states, weaker in non-stationary regimes  
   - Suggested: monitor transitions into “risky states” for early alerts  

### Deliverables (Task1/)  
- `task1_analysis.ipynb` — Jupyter Notebook with analysis pipeline  
- `state_summary.csv`, `anomalous_periods.csv`, `forecasts.csv`  
- `plots/` — visualization outputs  
- `README.md` — task-specific instructions  

---

## Task 2: RAG + LLM System Design

### Objective  
Design and prototype a Retrieval-Augmented Generation (RAG) system to query 50+ technical PDFs (manuals, SOPs, troubleshooting guides) and return reliable, cited answers.  

### Approach  
1. **System Architecture**  
   - Document ingestion and preprocessing (PDF/TXT → clean text)  
   - Chunking (200–400 tokens with overlap)  
   - Embeddings with `all-MiniLM-L6-v2` (SentenceTransformers)  
   - Indexing via FAISS (vector DB)  
   - Retrieval (dense vector plus optional lexical rerank)  
   - Answer generation with `flan-t5-small` (Hugging Face, free)  
   - Guardrails: fallback responses, enforced citations, sensitive query filter  

2. **Retrieval Strategy**  
   - Dense retrieval (FAISS cosine similarity)  
   - Rerank with lexical overlap to boost precision  
   - Return top-k chunks (with metadata for citation)  

3. **Guardrails and Challenges**  
   - Hallucination prevention: answers must cite retrieved snippets  
   - Low-similarity queries → graceful fallback message  
   - Sensitive queries → keyword filter  
   - Evaluation with precision@k, recall@k, and faithfulness  

4. **Scalability Plan**  
   - 10x docs → sharded FAISS or Chroma DB  
   - 100+ users → retrieval microservice and autoscaling LLM workers  
   - Cost efficiency: CPU embeddings, caching, lightweight open LLMs  

### Deliverables (Task2/)  
- `notes.md` — design notes and tradeoffs  
- `architecture_diagram.pptx` — RAG pipeline diagram  
- `prototype/` — runnable demo  
  - `rag_prototype.py` — main pipeline  
  - `requirements.txt` — dependencies  
  - `docs/` — sample input documents  
  - `evaluate.py` — optional evaluation script  
- `evaluation.csv` — retrieval metrics (precision@k, recall@k, faithfulness)  

---

## Final Presentation  
`Final_Presentation.pptx` (9 slides) covering:  
1. Data Preparation (Task 1)  
2. Analysis Strategy (Task 1)  
3. Insights (Task 1)  
4. System Architecture (Task 2)  
5. RAG Challenges and Guardrails (Task 2)  
6–9. Scalability, evaluation, and closing remarks  

---

## Tech Stack  
- **Python:** pandas, numpy, matplotlib, scikit-learn  
- **Time-series Forecasting:** ARIMA, Prophet  
- **Clustering:** KMeans  
- **Anomaly Detection:** Isolation Forest, MAD thresholds  
- **NLP and RAG:** Hugging Face Transformers, SentenceTransformers, FAISS, pdfplumber, NLTK  

---

## How to Run  

```bash
# Task 1
cd Task1
jupyter notebook task1_analysis.ipynb

# Task 2
cd Task2/prototype
pip install -r requirements.txt

Index docs:  
python rag_prototype.py --mode index --docs_dir docs/ --index_path ./task2_index/

Run a query:  
python rag_prototype.py --mode query --index_path ./task2_index/ --query "What does a sudden draft drop indicate?"

Run evaluation (optional):  
python evaluate.py

