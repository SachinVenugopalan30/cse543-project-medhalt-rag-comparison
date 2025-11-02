# PoC Plan — Med-HALT Baseline → MedGraphRAG Mitigation (Detailed Execution Plan)

> Purpose: a complete, runnable Proof-of-Concept that (1) measures LLM hallucination baseline using **Med-HALT**, (2) implements a minimal **MedGraphRAG** (TGC + U-Retrieval) pipeline, and (3) re-evaluates using Med-HALT to quantify hallucination reduction.
> Format: step-by-step execution instructions, dataset download commands & sources, project structure, and per-script responsibilities + dependencies.

---

## Table of contents

1. Overview & goals
2. High-level phases (what to do)
3. Dataset acquisition — exact sources + example commands (download & parse)
4. Implementation steps — detailed actions for each phase
5. Evaluation (Med-HALT re-evaluation) — exact procedure and scoring
6. Project structure (repo layout) and file responsibilities
7. Scripts: what each Python script does + required packages & config
8. Runbook (commands to execute the full PoC)
9. Notes: ethics, reproducibility, expected outcomes

---

## 1. Overview & goals

* Input: Med-HALT evaluation set (question prompts). ([GitHub][1])
* Evidence corpus: public biomedical corpora (PubMed abstracts / PMC-OA), MedlinePlus, CDC guidelines, OpenFDA drug labels, ClinicalTrials.gov, plus entity annotations (PubTator). ([PMC][2])
* Controlled vocabularies: ICD-10 / MeSH (open subsets) and PubTator for precomputed entity annotations. ([icdcdn.who.int][3])
* Output: two sets of model outputs (baseline LLM, MedGraphRAG-augmented), Med-HALT scores (penalized), analysis report.

---

## 2. High-level phases (what to do)

1. **Phase 0 — Environment & tooling:** create a reproducible environment (Python venv / Docker).
2. **Phase 1 — Baseline:** run Med-HALT evaluation on chosen off-the-shelf LLM(s) and record baseline metrics. ([GitHub][1])
3. **Phase 2 — Build corpus & graph (MedGraphRAG POC):** ingest corpora, chunk & embed docs, extract entities & map to CTVs, build a lightweight triple graph (TGC). ([PMC][2])
4. **Phase 3 — U-Retrieval & RAG generation:** implement top-down graph traversal + bottom-up vector re-ranking; generate answers using the same LLM but only with retrieved evidence and citation enforcement. ([PMC][2])
5. **Phase 4 — Re-evaluation:** run Med-HALT on RAG outputs and compare to baseline (penalized scoring). ([Hugging Face][4])
6. **Phase 5 — Analysis & deliverables:** metrics, ablation tables, example QA corrections, latency and abstention analyses.

---

## 3. Dataset acquisition — exact sources + example commands

> Each dataset below is publicly available and (for these components) does **not** require IRB, institutional approval, or paid access.

### 3.1 Med-HALT (benchmark & code)

* Use the canonical repo and/or Hugging Face mirror for dataset loaders and scoring harness. ([GitHub][1])
* Clone & example:

```bash
git clone https://github.com/medhalt/medhalt.git
# or
# use Hugging Face dataset loader in Python:
# from datasets import load_dataset
# ds = load_dataset("openlifescienceai/Med-HALT")
```

### 3.2 PubMed abstracts (baseline / full citations)

* Official download / FTP and E-utilities docs from NLM. Use baseline files for a snapshot of citations/abstracts. ([PubMed][5])
* Example (FTP download; replace file names with most recent baseline):

```bash
# Example — check the PubMed download page first to determine current file names
# Download baseline XML files (large)
wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.gz
```

### 3.3 PubMed Central (PMC) Open Access Subset (full-text)

* Official PMC OA list and FTP (full-text articles that allow reuse). Useful for longer contexts. ([PMC][2])
* Example:

```bash
# PMC OA manifest lists which PMCID bundles are available; follow PMC FTP instructions
# See: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
```

### 3.4 MedlinePlus (health topic summaries, XML)

* MedlinePlus provides compressed XML downloads for health topics (excellent for patient-facing definitions and authoritative summaries). ([MedlinePlus][6])
* Example:

```bash
# Download MedlinePlus health topic XML
wget https://medlineplus.gov/medlineplus_all_healthtopics.xml.zip
# (actual file path: see MedlinePlus XML page)
```

### 3.5 CDC guidelines & guideline collections

* Many CDC guideline pages provide HTML/PDF guidance you can scrape and store; CDC has guideline libraries. Use programmatic scraping carefully and respect robots.txt. ([CDC][7])

### 3.6 OpenFDA (drug labels & structured resources)

* OpenFDA provides downloadable zipped JSON sets (drug labeling, adverse events). Good for medication facts and contraindications. ([open.fda.gov][8])
* Example:

```bash
# openFDA drug label data download page provides zips you can curl/wget
# https://open.fda.gov/apis/drug/label/download/
```

### 3.7 ClinicalTrials.gov (trial descriptions)

* Full-downloadable study records / API for trial descriptions. Useful for clinical evidence retrieval. ([ClinicalTrials.gov][9])

### 3.8 PubTator / PubTator Central annotations (entity annotations)

* Precomputed biomedical entity annotations over PubMed/PMC, available via FTP or API — useful for mapping entity mentions to standardized IDs. ([NCBI][10])
* Example:

```bash
# PubTator FTP (check exact path in docs):
wget -r ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/
```

### 3.9 ICD-10 / MeSH (Controlled Vocabularies)

* ICD-10: WHO/CMS/CDC provide downloadable versions and files. Use ICD-10 (WHO) or ICD-10-CM (CDC/CMS) files for code → label mappings. ([icdcdn.who.int][3])
* MeSH: available from NLM (MeSH RDF/ XML dumps; not listed above but available via NLM). (If needed, I can add the exact link.)

---

## 4. Implementation steps — detailed actions for each phase

> Each step below assumes you are working in a project repo (see structure in section 6). Commands reference the scripts described later.

### Phase 0 — Environment & reproducibility

1. Create venv and install core deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Start Neo4j (Docker) for TGC:

```bash
docker run --name medgraph-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5
```

### Phase 1 — Baseline Med-HALT evaluation

1. Load Med-HALT dataset (use their loader or HF dataset).
2. Implement `baseline_run.py` to:

   * Read questions (subset controllable via CLI).
   * Send queries to chosen LLM(s) in a consistent prompt format (zero-shot / few-shot per experiment).
   * Save outputs in JSONL:

```json
{"id":"Q0001","question":"...","response":"...","model":"gpt-3.5-turbo","timestamp":"..."}
```

3. Run official Med-HALT scorer (from the Med-HALT repo) to compute penalized accuracy (+1 / -0.25). ([GitHub][1])

### Phase 2 — Corpus ingestion & TGC construction

1. **Download corpora** per section 3 and store under `data/corpus/`.
2. **Chunking**:

   * Implement `ingest/chunker.py`:

     * Static chunking: split docs by sections/paragraphs up to `max_tokens` (e.g., 800-1,000 tokens).
     * If section > threshold, run semantic clustering (sentence embeddings + agglomerative clustering) to produce concept-cohesive chunks.
   * Output: `data/chunks/*.jsonl` with `{"id","text","source","pubmed_id","title","entities":[]}`.
3. **Entity extraction & CTV mapping**:

   * Implement `ingest/entities.py`:

     * Use SciSpaCy (or PubTator annotations) to extract entities.
     * Map entities to ICD/MeSH using simple string / MeSH lookup or via UMLS (if used).
   * Add `entities` and `ctv_codes` fields to chunk metadata.
   * (Optional) Use PubTator downloads to speed mapping. ([NCBI][10])
4. **Graph construction (Triple Graph Construction)**:

   * Implement `graph/build_graph.py`:

     * Graph nodes: `chunk`, `source_doc` (PubMedID), `ctv` (ICD/MeSH code).
     * Graph edges: `chunk -> source_doc` (origin), `chunk -> ctv` (entity mapping), `source_doc -> ctv` (if applicable).
   * Store graph in Neo4j or on-disk NetworkX pickle for POC.

### Phase 3 — U-Retrieval & RAG generation

1. **Vector Index**:

   * Implement `index/build_index.py`:

     * Compute embeddings for every chunk (Sentence-Transformers `all-mini-lm` or biomedical model).
     * Build FAISS index with metadata linking to chunk IDs and associated CTVs.
2. **Top-down phase** (`retriever/top_down.py`):

   * Query → extract entities & map to candidate CTV nodes in the graph.
   * Traverse graph to fetch candidate `source_doc` and `chunk` IDs (e.g., use 2-hop traversal from CTV or rank by node centrality).
3. **Bottom-up refinement** (`retriever/bottom_up.py`):

   * For candidate chunk set, compute embedding similarity to the query (restrict FAISS search to candidate IDs or use metadata filtering).
   * Score = `alpha*graph_score + beta*embedding_score` (tunable).
4. **Prompt & Generation** (`rag/generate.py`):

   * Prepare the “evidence package” (top N snippets with source and CTV codes).
   * Use an enforced template instructing the model to cite snippets and ABSTAIN if insufficient.
   * Call the same LLM used for baseline (keep model constant).
5. **Abstention mechanisms**:

   * Implement simple retrieval threshold (`if best_score < tau: ABSTAIN`).
   * Implement LLM self-check: small classifier on logits / response embeddings to predict correctness (optional advanced).

### Phase 4 — Re-evaluation with Med-HALT

1. Run the exact same Med-HALT scoring pipeline on the RAG outputs (`evaluation/score.py`), using identical scoring thresholds and abstention accounting. ([Hugging Face][4])
2. Compute:

   * Penalized accuracy (overall, RHT, MHT splits)
   * Abstention rate and precision (how often ABSTAIN avoided a dangerous error)
   * False-confident rate (incorrect answers that were not ABSTAIN)
   * Latency / throughput metrics (average time per query)
3. Statistical testing (paired comparisons). Use McNemar’s test for paired binary correctness, or bootstrap confidence intervals.

---

## 5. Evaluation (Med-HALT re-evaluation) — exact scoring procedure

1. **Use Med-HALT scoring**: +1 for correct, −0.25 for incorrect, 0 for ABSTAIN (or configurable). Use canonical Med-HALT scorer from repo/HF mirror to guarantee comparability. ([GitHub][1])
2. **Per-question steps**:

   * Extract claims from model response (use the Med-HALT claim extractor or your own IE).
   * For each claim, check for supporting evidence in corpus via entailment / semantic match.
   * Classify claim as *Supported*, *Refuted*, or *Unverifiable*.
   * Aggregate support status into per-question correctness (align with Med-HALT definitions).
3. **Produce final tables**:

   * Overall penalized score (baseline vs RAG)
   * RHT & MHT breakdown
   * Top-k questions where RAG corrected hallucination (qualitative examples)
   * Latency & resource usage

---

## 6. Project structure (suggested repo layout)

```
medhalt-rag-poc/
├── README.md
├── requirements.txt
├── docker-compose.yml           # optional: Neo4j, redis, etc.
├── .env                         # API keys (OpenAI, etc.) - DO NOT COMMIT
│
├── data/
│   ├── raw/
│   │   ├── medhalt/             # med-halt dataset
│   │   ├── pubmed_baseline/     # downloaded baseline XMLs
│   │   ├── pmc_oa/              # PMC OA articles
│   │   ├── medlineplus/         # MedlinePlus XMLs
│   │   └── pubtator/            # PubTator annotations
│   └── chunks/                  # post-chunking JSONL
│
├── ingest/
│   ├── download_datasets.py     # utilities to download the datasets listed
│   ├── chunker.py               # static + semantic chunker
│   ├── entities.py              # entity extraction & CTV mapping
│   └── build_corpus.py          # pipeline to produce data/chunks/*.jsonl
│
├── index/
│   ├── build_index.py           # compute embeddings + build FAISS index
│   └── index_utils.py
│
├── graph/
│   ├── build_graph.py           # build TGC (Neo4j or NetworkX)
│   ├── query_graph.py           # helper traversals
│   └── graph_utils.py
│
├── retriever/
│   ├── top_down.py              # graph-based candidate selection
│   ├── bottom_up.py             # vector re-ranking & scoring
│   └── retrieve.py              # orchestration (returns top N snippets)
│
├── rag/
│   ├── prompt_templates.py      # prompt strings & citation enforcement templates
│   ├── generate.py              # build prompt + call LLM
│   └── abstain_detector.py      # optional classifier for abstentions
│
├── baseline/
│   ├── baseline_run.py          # run baseline LLM on Med-HALT
│   └── baseline_prompts.py
│
├── evaluation/
│   ├── medhalt_loader.py        # load Med-HALT splits
│   ├── claim_extractor.py       # extract atomic claims
│   ├── entailment.py            # entailment / support classifier
│   └── scorer.py                # compute penalized metrics & tables
│
├── experiments/
│   ├── run_all_experiments.py   # orchestrates baseline -> rag -> eval
│   └── configs/                 # JSON configs for runs (models, thresholds)
│
├── notebooks/                   # EDA and analysis notebooks
└── reports/
    └── results_summary.md
```

---

## 7. Scripts: responsibilities, inputs, outputs & requirements

> For each key script, I list purpose, inputs, outputs, and required packages.

### `ingest/download_datasets.py`

* Purpose: download Med-HALT, PubMed baseline, PMC-OA manifest, MedlinePlus XML, PubTator, OpenFDA zips, ClinicalTrials bulk, ICD files.
* Inputs: none (or config file with target paths).
* Outputs: files under `data/raw/`.
* Req: `requests`, `ftplib`, `wget`, `biopython` for parsing.

**Example use / snippet**:

```bash
python ingest/download_datasets.py --dest data/raw
```

### `ingest/chunker.py`

* Purpose: static & semantic chunking of raw docs into chunks for retrieval.
* Inputs: raw docs in `data/raw/`
* Outputs: `data/chunks/*.jsonl`
* Req: `nltk`, `sentence-transformers`, `scikit-learn`, `transformers`

### `ingest/entities.py`

* Purpose: extract medical entities from chunks and map to ICD/MeSH using exact/approx mapping or PubTator.
* Inputs: `data/chunks/*.jsonl` + PubTator or MeSH files
* Outputs: enriched chunk JSONL with `entities` and `ctv_codes`
* Req: `scispacy`, `en_core_sci_sm` (or use PubTator FTP data)

### `index/build_index.py`

* Purpose: embed chunks and build FAISS index, persist index + metadata.
* Inputs: `data/chunks/*.jsonl`
* Outputs: `index/faiss.index` + metadata `index/metadata.pkl`
* Req: `faiss-cpu`, `sentence-transformers`, `numpy`, `pickle`

### `graph/build_graph.py`

* Purpose: build Triple Graph Construction from chunks (chunk nodes, source nodes, CTV nodes).
* Inputs: enriched chunk JSONL
* Outputs: Neo4j DB (or `graph/graph.pkl` for NetworkX)
* Req: `neo4j` driver (if using Neo4j), `networkx`

### `retriever/top_down.py` & `retriever/bottom_up.py`

* Purpose: implement U-Retrieval phases described earlier.
* Inputs: user question, graph DB, FAISS index
* Outputs: ranked snippets + provenance metadata
* Req: `networkx` or `neo4j`, `faiss`, `sentence-transformers`

### `rag/generate.py`

* Purpose: format prompt with evidence, call LLM, enforce citation & ABSTAIN logic.
* Inputs: question, snippets
* Outputs: model response JSON (with `evidence_used`, `abstained` flag)
* Req: `openai` (or HF transformers client), `tqdm`

### `baseline/baseline_run.py`

* Purpose: run baseline LLMs on Med-HALT questions; store outputs.
* Inputs: Med-HALT dataset
* Outputs: `results/baseline_res.jsonl`
* Req: `openai` or `transformers` client

### `evaluation/scorer.py`

* Purpose: implement Med-HALT scoring (use repo harness if available); compute penalized accuracy and breakdown.
* Inputs: `results/*.jsonl` (baseline & rag)
* Outputs: `reports/*_scoring.csv`, summary metrics, significance tests
* Req: `pandas`, `scipy`

---

## 8. Runbook (example commands to execute the PoC)

1. Clone repo and install:

```bash
git clone <this-repo>
cd medhalt-rag-poc
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download datasets (this step may take time and disk space):

```bash
python ingest/download_datasets.py --dest data/raw
# or download only Med-HALT for quick start
```

3. Build chunks & entity annotations:

```bash
python ingest/chunker.py --input data/raw/pubmed_baseline --output data/chunks
python ingest/entities.py --chunks data/chunks --pubtator data/raw/pubtator --ctv data/raw/icd10.csv
```

4. Build index & graph:

```bash
python index/build_index.py --chunks data/chunks --out index/
python graph/build_graph.py --chunks data/chunks --out graph/ --neo4j_uri bolt://localhost:7687
```

5. Baseline run:

```bash
python baseline/baseline_run.py --dataset data/raw/medhalt --model gpt-3.5-turbo --out results/baseline_res.jsonl
python evaluation/scorer.py --pred results/baseline_res.jsonl --gold data/raw/medhalt/gold.jsonl --out reports/baseline_scores.csv
```

6. RAG run:

```bash
python retriever/retrieve.py --question_file data/raw/medhalt/questions.jsonl --out results/rag_candidates.jsonl
python rag/generate.py --candidates results/rag_candidates.jsonl --model gpt-3.5-turbo --out results/rag_res.jsonl
python evaluation/scorer.py --pred results/rag_res.jsonl --gold data/raw/medhalt/gold.jsonl --out reports/rag_scores.csv
```

7. Compare & analyze:

```bash
python experiments/run_all_experiments.py --config experiments/configs/default.json
# open notebooks/analysis.ipynb for charts
```

---

## 9. Notes: ethics, resources, expected outcomes

### Ethics & Privacy

* This PoC uses public data only (Med-HALT, PubMed / PMC OA, MedlinePlus, CDC, OpenFDA, ClinicalTrials.gov, PubTator). No PHI or private EHR data. Cite sources and keep results internal until clinically validated. ([PMC][2])

### Compute & storage

* PubMed baseline / PMC OA are large (tens to hundreds of GB). For a quick POC, use a **subset** (e.g., 1M abstracts or specific topical slices). You can also prioritize MedlinePlus + OpenFDA + a subset of PubMed (e.g., `neoplasm` or `cardio`) to reduce footprint.

### Expected results (POC)

* **Memory-based hallucination (MHT)** should improve significantly because explicit retrieval provides factual grounding (expect large gains).
* **Reasoning hallucination (RHT)** should improve but less dramatically — RAG helps by providing accurate premises but the model still synthesizes conclusions.
* **Tradeoffs**: improved factuality at the cost of latency and infrastructure complexity.

---

## Sources / dataset links (quick reference)

* Med-HALT GitHub (code & benchmark). ([GitHub][1])
* Med-HALT (Hugging Face dataset). ([Hugging Face][4])
* PMC Open Access Subset (manifest & FTP). ([PMC][2])
* PubMed data download (baseline & E-utilities). ([PubMed][5])
* MedlinePlus XML (health topic downloads). ([MedlinePlus][6])
* CDC guidelines & recommendations (guideline collections). ([CDC][7])
* openFDA data (drug labeling & zipped downloads). ([open.fda.gov][8])
* ClinicalTrials.gov data (bulk download/API). ([ClinicalTrials.gov][11])
* PubTator Central (annotated entities, FTP). ([NCBI][10])
* ICD-10 downloads / CDC CMS resources. ([icdcdn.who.int][3])

---

If you’d like, I can:

* generate the `requirements.txt` and a `Dockerfile` / `docker-compose.yml` for the stack (Neo4j + Python),
* produce the initial skeleton files and a few fully implemented scripts (e.g., `baseline_run.py` and `evaluation/scorer.py`) so you can run the baseline in <1 hour on a small subset, or
* create the exact Med-HALT scoring wrapper calling the official harness and returning CSV metrics.

Which of those should I do next?

[1]: https://github.com/medhalt/medhalt?utm_source=chatgpt.com "medhalt/medhalt"
[2]: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/?utm_source=chatgpt.com "PMC Open Access Subset"
[3]: https://icdcdn.who.int/icd10/index.html?utm_source=chatgpt.com "ICD-10 Download Page"
[4]: https://huggingface.co/datasets/openlifescienceai/Med-HALT?utm_source=chatgpt.com "openlifescienceai/Med-HALT · Datasets at Hugging Face"
[5]: https://pubmed.ncbi.nlm.nih.gov/download/?utm_source=chatgpt.com "Download PubMed Data - NIH"
[6]: https://medlineplus.gov/xml.html?utm_source=chatgpt.com "MedlinePlus XML Files"
[7]: https://www.cdc.gov/infection-control/hcp/guidance/index.html?utm_source=chatgpt.com "Guidelines and Guidance Library | Infection Control | CDC"
[8]: https://open.fda.gov/apis/drug/label/?utm_source=chatgpt.com "Drug Labeling Overview"
[9]: https://clinicaltrials.gov/data-api/how-download-study-records?utm_source=chatgpt.com "How to Download Study Records"
[10]: https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/PubTatorCentral/?utm_source=chatgpt.com "PubTator Central - NCBI - NLM - NIH"
[11]: https://clinicaltrials.gov/data-api/about-api/api-migration?utm_source=chatgpt.com "API Migration Guide"
