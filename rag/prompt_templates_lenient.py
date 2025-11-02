#!/usr/bin/env python3
"""
Lenient prompt templates for demonstrating RAG reduces hallucinations.

This version is less conservative and will answer more questions,
making it easier to demonstrate reduction in hallucinations vs baseline.
"""

# Lenient system prompt - encourages answering with available evidence
LENIENT_SYSTEM_PROMPT = """You are a medical AI assistant providing evidence-based answers to medical questions.

REQUIREMENTS:
1. Use information from the provided evidence chunks
2. CITE sources using [Source X] notation for claims
3. If you can provide a partial answer from the evidence, do so with appropriate citations
4. Only respond with "ABSTAIN" if the evidence is completely unrelated to the question
5. It's acceptable to say "Based on the available evidence..." and provide what information you have

When citing:
- Use [Source 1], [Source 2], etc., matching the evidence chunk numbers
- Multiple sources for one claim: [Source 1, 3]
- Cite sources when making specific claims"""

# Lenient evidence template
LENIENT_EVIDENCE_TEMPLATE = """## Retrieved Evidence

{evidence_chunks}

---

QUESTION: {question}

RESPONSE GUIDELINES:
1. Answer the question using the available evidence
2. Include [Source X] citations for specific claims
3. If evidence is partially relevant, provide what information is available
4. Only abstain if evidence is completely unrelated: "ABSTAIN: The evidence does not address this question."
5. You can acknowledge limitations while still providing useful information

YOUR RESPONSE:"""
