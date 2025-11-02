# Med-HALT RAG PoC: Medical Hallucination Mitigation with MedGraphRAG

A comprehensive Proof-of-Concept implementing MedGraphRAG to mitigate LLM hallucinations in medical question-answering, evaluated using the Med-HALT benchmark.

## 📚 Documentation

**👉 For complete step-by-step instructions, see [HowToRun.md](./HowToRun.md)**

The HowToRun.md guide provides:
- Detailed phase-by-phase execution steps
- Expected outputs and verification commands
- Troubleshooting for common issues
- Complete workflow from setup to evaluation
- Actual results from 100K PubMed corpus testing

## Overview

This project demonstrates:
1. **Baseline evaluation** of LLM hallucination rates using Med-HALT
2. **MedGraphRAG implementation** with Triple Graph Construction (TGC) and U-Retrieval
3. **Quantitative comparison** showing hallucination reduction through evidence-grounded generation

## Key Features

- **Evidence-grounded RAG** using biomedical corpora (PubMed, PMC-OA, MedlinePlus)
- **Triple Graph Construction** linking chunks → source documents → controlled vocabularies (ICD-10)
- **U-Retrieval** combining top-down graph traversal with bottom-up vector re-ranking
- **Citation enforcement** with 100% compliance and abstention mechanisms
- **Med-HALT evaluation** with penalized scoring (+1 correct, -0.25 incorrect, 0 abstain)

## Results Summary

From our 100K PubMed corpus evaluation (100 questions):

| Metric | Baseline | RAG Strict | RAG Lenient |
|--------|----------|------------|-------------|
| Accuracy (overall) | 65% | 7% | 15% |
| Hallucination Rate | 35% | **22%** ✅ | 57% |
| Abstention Rate | 0% | 71% | 28% |
| Citation Compliance | 0% | **100%** ✅ | **100%** ✅ |

**Key Finding:** RAG Strict mode achieved a **37% reduction in hallucinations** (35% → 22%) through intelligent abstention, while maintaining 100% citation transparency.

**See [RESULTS_ANALYSIS.md](./RESULTS_ANALYSIS.md) for detailed analysis.**

## Prerequisites

### Required:
- **Python 3.10+** (tested with 3.10.14)
- **pyenv** (for Python version management)
- **20GB+ disk space** for datasets and indices
- **OpenAI API key** (for GPT-3.5-turbo or GPT-4o-mini)
- **macOS, Linux, or WSL** (Windows users should use WSL)

### Optional:
- **Docker** (if using Neo4j graph database; NetworkX local graph is default)
- **GPU** (for faster embedding generation, but CPU works fine)

### Quick Installation

```bash
# Run the automated setup script
chmod +x setup.sh
./setup.sh

# This will:
# - Install Python 3.10.14 via pyenv
# - Create virtual environment
# - Install all dependencies
# - Download SciSpaCy model
# - Set up .env file

# Activate the environment
source venv/bin/activate
```

**For detailed installation and all execution phases, see [HowToRun.md](./HowToRun.md)**

### Optional: Neo4j (Graph Database)

By default, the system uses NetworkX (local Python graph). For Neo4j:

```bash
docker-compose up -d
# Access Neo4j browser at http://localhost:7474
# Default credentials: neo4j / medhalt2024
```

## Project Structure

```
Implementationv2/
├── data/                     # Data directory
│   ├── raw/                 # Raw downloaded datasets
│   │   ├── medhalt/        # Med-HALT benchmark
│   │   ├── pubmed_baseline/ # PubMed abstracts
│   │   ├── pmc_oa/         # PMC Open Access articles
│   │   ├── medlineplus/    # MedlinePlus health topics
│   │   └── pubtator/       # PubTator entity annotations
│   └── chunks/             # Processed chunks (JSONL)
│
├── ingest/                  # Data ingestion & preprocessing
├── index/                   # Vector index construction
├── graph/                   # Triple Graph Construction (TGC)
├── retriever/               # U-Retrieval implementation
├── rag/                     # RAG generation & prompting
├── baseline/                # Baseline LLM evaluation
├── evaluation/              # Med-HALT scoring & metrics
├── experiments/             # Experiment orchestration
├── notebooks/               # Analysis notebooks
└── reports/                 # Generated reports
```

## Usage Overview

**📖 For complete execution steps, see [HowToRun.md](./HowToRun.md)**

The pipeline consists of 5 phases:

### Phase 1: Download Datasets
Download Med-HALT benchmark and PubMed corpus (10K or 100K articles)

### Phase 2: Build Corpus & Knowledge Base
- Chunk documents into 256-token segments
- Extract entities with SciSpaCy + ICD-10 mapping
- Build FAISS/NumPy vector index
- Construct knowledge graph (NetworkX or Neo4j)

### Phase 3: Run Baseline Evaluation
Evaluate raw LLM performance without RAG (establishes hallucination baseline)

### Phase 4: Run RAG Evaluation
- Retrieve evidence using U-Retrieval (top-k=10-20)
- Generate responses with citation enforcement
- Test both strict and lenient abstention modes

### Phase 5: Compare & Analyze
Generate comprehensive comparison reports and analyze results

**Each phase includes:**
- Exact commands to run
- Expected outputs
- Verification steps
- Troubleshooting guidance

## Actual Results (100K Corpus)

From testing with 100K PubMed abstracts on 100 Med-HALT questions:

### Hallucination Reduction
- **Baseline:** 35% hallucination rate
- **RAG Strict:** 22% hallucination rate (**37% reduction** ✅)
- **RAG Lenient:** 57% hallucination rate (worse - validates conservative thresholds)

### Citation & Transparency
- **Baseline:** 0% citation compliance
- **RAG (both modes):** 100% citation compliance ✅

### Coverage vs Safety Trade-off
- **Baseline:** 100% coverage, 65% accuracy, 0% abstention
- **RAG Strict:** 29% coverage, 24% accuracy, 71% abstention (safe)
- **RAG Lenient:** 72% coverage, 21% accuracy, 28% abstention (risky)

### Key Insights
1. ✅ **Abstention works:** Refusing to answer prevents 13 hallucinations
2. ✅ **Citations enable verification:** All claims traceable to sources
3. ⚠️ **Corpus quality matters:** RAG accuracy depends on retrieval quality
4. ❌ **Lenient mode fails:** Forcing answers with weak evidence increases hallucinations

**For detailed metric calculations and analysis, see [RESULTS_ANALYSIS.md](./RESULTS_ANALYSIS.md)**

## Important Documentation Files

- **[HowToRun.md](./HowToRun.md)** - Complete step-by-step execution guide with all commands

## Dataset Sources

- [Med-HALT Benchmark](https://github.com/medhalt/medhalt) - Medical hallucination evaluation
- [PubMed via E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25501/) - Biomedical literature
- [ICD-10 Codes](https://www.cms.gov/medicare/coding-billing/icd-10-codes) - Disease classification
- [PubTator Central](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/PubTatorCentral/) - Entity annotations

## Ethical Considerations

- Uses **public data only** (no PHI/EHR)
- Results are for **research purposes** - not clinical use
- **Not validated for clinical decision-making**
- **Cite sources** appropriately
- Validate with domain experts before any deployment
- System should complement, not replace, medical professionals

## License

This is a research/educational project. Please cite sources and follow dataset licenses.

## Citation

If you use this code, please cite:
- Med-HALT benchmark paper
- MedGraphRAG methodology
- Original dataset sources

## Contributing

This is a PoC for educational/research purposes. For issues or suggestions, please open an issue.

## Acknowledgments

- Med-HALT team for the benchmark
- NCBI for biomedical data resources
- OpenFDA, CDC, MedlinePlus for public health data
