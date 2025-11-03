# Set variables
MODEL="gemini-2.5-pro"
PROVIDER="gemini"  # Options: openai, gemini
LIMIT=10  # Reduced for testing
QUESTIONS="data/raw/medhalt/reasoning_FCT/test.jsonl"
TOPK=10
MAX_TOKENS=8192  # Gemini's max output tokens (8192 for most models)

# Create file suffix with provider and model name (lowercase, replace dots with dashes)
PROVIDER_SUFFIX=$(echo "$PROVIDER" | tr '[:upper:]' '[:lower:]')
MODEL_SUFFIX=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '.' '-')
FILE_SUFFIX="${PROVIDER_SUFFIX}_${MODEL_SUFFIX}"

echo "=========================================="
echo "Running End-to-End RAG Pipeline"
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Questions: $LIMIT"
echo "File suffix: _${FILE_SUFFIX}"
echo "=========================================="

# Step 1: Run Baseline (No RAG)
echo ""
echo "=== STEP 1: Baseline Evaluation ==="
python baseline/baseline_run.py \
  --dataset $QUESTIONS \
  --model $MODEL \
  --provider $PROVIDER \
  --mode zero-shot \
  --limit $LIMIT \
  --max-tokens $MAX_TOKENS \
  --out results/baseline_results_${FILE_SUFFIX}.jsonl

# Step 2: Score Baseline
echo ""
echo "=== STEP 2: Score Baseline ==="
python evaluation/scorer.py \
  --pred results/baseline_results_${FILE_SUFFIX}.jsonl \
  --gold $QUESTIONS \
  --out reports/baseline_scores_${FILE_SUFFIX}.json

# Step 3: Retrieve Evidence
echo ""
echo "=== STEP 3: Retrieve Evidence ==="
python retriever/retrieve.py \
  --question-file $QUESTIONS \
  --index-dir index \
  --graph graph/graph.pkl \
  --icd data/raw/icd10/icd10_codes_enhanced.csv \
  --out results/retrieval_results_${FILE_SUFFIX}.jsonl \
  --top-k $TOPK \
  --limit $LIMIT \
  --alpha 0.5 \
  --beta 0.5

# Step 4: Generate RAG Responses (Strict Mode)
echo ""
echo "=== STEP 4: RAG Generation (Strict) ==="
python rag/generate.py \
  --candidates results/retrieval_results_${FILE_SUFFIX}.jsonl \
  --model $MODEL \
  --provider $PROVIDER \
  --template strict \
  --max-tokens $MAX_TOKENS \
  --out results/rag_results_strict_${FILE_SUFFIX}.jsonl

# Step 5: Score RAG Strict
echo ""
echo "=== STEP 5: Score RAG Strict ==="
python evaluation/scorer.py \
  --pred results/rag_results_strict_${FILE_SUFFIX}.jsonl \
  --gold $QUESTIONS \
  --out reports/rag_scores_strict_${FILE_SUFFIX}.json

# Step 6: Generate RAG Responses (Lenient Mode)
echo ""
echo "=== STEP 6: RAG Generation (Lenient) ==="
python rag/generate.py \
  --candidates results/retrieval_results_${FILE_SUFFIX}.jsonl \
  --model $MODEL \
  --provider $PROVIDER \
  --template lenient \
  --max-tokens $MAX_TOKENS \
  --out results/rag_results_lenient_${FILE_SUFFIX}.jsonl

# Step 7: Score RAG Lenient
echo ""
echo "=== STEP 7: Score RAG Lenient ==="
python evaluation/scorer.py \
  --pred results/rag_results_lenient_${FILE_SUFFIX}.jsonl \
  --gold $QUESTIONS \
  --out reports/rag_scores_lenient_${FILE_SUFFIX}.json

# Step 8: Generate Comparison Reports
echo ""
echo "=== STEP 8: Comparison Reports ==="
python evaluation/scorer.py \
  --pred results/rag_results_strict_${FILE_SUFFIX}.jsonl \
  --gold $QUESTIONS \
  --baseline-pred results/baseline_results_${FILE_SUFFIX}.jsonl \
  --out reports/comparison_strict_${FILE_SUFFIX}.json

python evaluation/scorer.py \
  --pred results/rag_results_lenient_${FILE_SUFFIX}.jsonl \
  --gold $QUESTIONS \
  --baseline-pred results/baseline_results_${FILE_SUFFIX}.jsonl \
  --out reports/comparison_lenient_${FILE_SUFFIX}.json

# Step 9: Display Results Summary
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "Provider: ${PROVIDER}"
echo "Model: ${MODEL}"
echo "=========================================="
echo ""
echo "=== BASELINE ==="
cat reports/baseline_scores_${FILE_SUFFIX}.json
echo ""
echo "=== RAG STRICT ==="
cat reports/rag_scores_strict_${FILE_SUFFIX}.json
echo ""
echo "=== RAG LENIENT ==="
cat reports/rag_scores_lenient_${FILE_SUFFIX}.json
echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved to:"
echo "  results/*_${FILE_SUFFIX}.jsonl"
echo "  reports/*_${FILE_SUFFIX}.json"
echo "=========================================="