#!/usr/bin/env python3
"""
Prompt templates for MedGraphRAG generation.

Provides:
- Citation-enforcing prompts
- Abstention instructions
- Evidence formatting
"""

# Base system prompt for medical Q&A
MEDICAL_SYSTEM_PROMPT = """You are a medical AI assistant providing evidence-based answers to medical questions.

CRITICAL REQUIREMENTS:
1. ONLY use information from the provided evidence chunks
2. CITE sources using [Source X] notation for every claim
3. If the evidence is insufficient or contradictory, you MUST respond with "ABSTAIN" and explain why
4. Do NOT use external knowledge - rely solely on the provided evidence
5. Clearly distinguish between established facts and areas of uncertainty

When citing:
- Use [Source 1], [Source 2], etc., matching the evidence chunk numbers
- Multiple sources for one claim: [Source 1, 3]
- Every factual claim requires a citation"""

# Evidence formatting template
EVIDENCE_TEMPLATE = """## Retrieved Evidence

{evidence_chunks}

---

Question: {question}

Instructions:
- Answer using ONLY the evidence above
- Cite sources for every claim using [Source X]
- If evidence is insufficient, unclear, or contradictory, respond with: "ABSTAIN: [reason]"
- Be concise but thorough
- Acknowledge limitations when appropriate

Answer:"""

# Alternative template with stricter abstention
STRICT_EVIDENCE_TEMPLATE = """## Medical Evidence Database Query Results

You have retrieved {num_chunks} evidence chunks for the question below.

{evidence_chunks}

---

QUESTION: {question}

RESPONSE GUIDELINES:
1. Answer ONLY if the evidence clearly supports a response
2. Every sentence must include [Source X] citations
3. If ANY of these conditions are met, you MUST respond with "ABSTAIN: [specific reason]":
   - Evidence is insufficient or incomplete
   - Evidence contains contradictions
   - Evidence is outdated or uncertain
   - Question asks about something not covered in evidence
   - You cannot provide a safe, accurate answer

4. Abstention format: "ABSTAIN: The evidence does not provide sufficient information about [specific aspect]. Available sources discuss [what is covered] but do not address [what is missing]."

YOUR RESPONSE:"""

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        "question": "What are the common side effects of metformin?",
        "evidence": [
            {
                "source": 1,
                "text": "Metformin commonly causes gastrointestinal side effects including nausea, vomiting, and diarrhea in approximately 25-30% of patients. These effects are usually mild and transient.",
                "source_id": "PMID:12345678"
            },
            {
                "source": 2,
                "text": "Long-term metformin use has been associated with vitamin B12 deficiency in some patients. Regular monitoring is recommended.",
                "source_id": "PMID:23456789"
            }
        ],
        "response": "Common side effects of metformin include gastrointestinal symptoms such as nausea, vomiting, and diarrhea, affecting approximately 25-30% of patients [Source 1]. These effects are typically mild and temporary [Source 1]. Additionally, long-term use may lead to vitamin B12 deficiency, necessitating regular monitoring [Source 2]."
    },
    {
        "question": "What is the recommended treatment for stage 4 pancreatic cancer with liver metastases?",
        "evidence": [
            {
                "source": 1,
                "text": "Pancreatic cancer is the fourth leading cause of cancer death in the United States.",
                "source_id": "PMID:34567890"
            },
            {
                "source": 2,
                "text": "Early detection of pancreatic cancer significantly improves outcomes.",
                "source_id": "PMID:45678901"
            }
        ],
        "response": "ABSTAIN: The provided evidence discusses pancreatic cancer statistics and early detection [Source 1, 2] but does not contain specific information about treatment protocols for stage 4 pancreatic cancer with liver metastases. Clinical treatment decisions for advanced pancreatic cancer require current oncology guidelines and individualized treatment plans not present in the available evidence."
    }
]


def format_evidence_chunks(chunks: list) -> str:
    """
    Format retrieved chunks into evidence string.

    Args:
        chunks: List of chunk dictionaries with 'text', 'source_id', etc.

    Returns:
        Formatted evidence string
    """
    # Safety check: ensure chunks is iterable
    if not isinstance(chunks, (list, tuple)):
        raise TypeError(f"chunks must be a list or tuple, got {type(chunks).__name__}: {chunks}")
    
    formatted_chunks = []

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        source_id = chunk.get('source_id', 'Unknown')
        source = chunk.get('source', 'Unknown')

        chunk_str = f"[Source {i}] ({source}: {source_id})\n{text}"
        formatted_chunks.append(chunk_str)

    return "\n\n".join(formatted_chunks)


def build_prompt(
    question: str,
    evidence_chunks: list,
    template: str = "default",
    include_few_shot: bool = False,
    system_prompt: str = MEDICAL_SYSTEM_PROMPT
) -> dict:
    """
    Build complete prompt for generation.

    Args:
        question: User question
        evidence_chunks: Retrieved evidence chunks
        template: Template type ("default", "strict", or "lenient")
        include_few_shot: Include few-shot examples
        system_prompt: System prompt (default: MEDICAL_SYSTEM_PROMPT)

    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    # Format evidence
    formatted_evidence = format_evidence_chunks(evidence_chunks)

    # Select template and system prompt based on mode
    if template == "lenient":
        from prompt_templates_lenient import LENIENT_SYSTEM_PROMPT, LENIENT_EVIDENCE_TEMPLATE
        system_prompt = LENIENT_SYSTEM_PROMPT
        user_template = LENIENT_EVIDENCE_TEMPLATE
    elif template == "strict":
        user_template = STRICT_EVIDENCE_TEMPLATE
    else:
        user_template = EVIDENCE_TEMPLATE

    # Build user prompt
    user_prompt = user_template.format(
        evidence_chunks=formatted_evidence,
        question=question,
        num_chunks=len(evidence_chunks)
    )

    # Add few-shot examples if requested
    if include_few_shot:
        few_shot_text = "\n\n## Examples of Good Responses:\n\n"

        for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
            few_shot_text += f"### Example {i}\n"
            few_shot_text += f"Question: {example['question']}\n\n"

            example_evidence = format_evidence_chunks(example['evidence'])
            few_shot_text += f"Evidence:\n{example_evidence}\n\n"

            few_shot_text += f"Response: {example['response']}\n\n---\n\n"

        # Insert examples before the actual question
        user_prompt = few_shot_text + user_prompt

    return {
        "system": system_prompt,
        "user": user_prompt
    }


def extract_citations(response: str) -> list:
    """
    Extract [Source X] citations from response.

    Args:
        response: Model response text

    Returns:
        List of cited source numbers
    """
    import re

    # Pattern: [Source 1], [Source 2, 3], etc.
    pattern = r'\[Source\s+([\d,\s]+)\]'
    matches = re.findall(pattern, response, re.IGNORECASE)

    citations = []
    for match in matches:
        # Parse comma-separated numbers
        numbers = [int(n.strip()) for n in match.split(',') if n.strip().isdigit()]
        citations.extend(numbers)

    # Return unique citations
    return sorted(set(citations))


def is_abstention(response: str) -> bool:
    """
    Check if response is an abstention.

    Args:
        response: Model response text

    Returns:
        True if response contains abstention
    """
    response_lower = response.lower().strip()

    # Check for explicit abstention
    if response_lower.startswith('abstain'):
        return True

    # Check for abstention indicators
    abstention_phrases = [
        'abstain:',
        'i must abstain',
        'i cannot provide',
        'insufficient evidence',
        'insufficient information'
    ]

    return any(phrase in response_lower for phrase in abstention_phrases)


def validate_response(response: str, num_evidence_chunks: int) -> dict:
    """
    Validate response for citation compliance.

    Args:
        response: Model response
        num_evidence_chunks: Number of evidence chunks provided

    Returns:
        Validation dictionary with issues and warnings
    """
    issues = []
    warnings = []

    # Check if abstention
    if is_abstention(response):
        return {
            'valid': True,
            'abstained': True,
            'issues': [],
            'warnings': []
        }

    # Extract citations
    citations = extract_citations(response)

    # Check for citations
    if not citations:
        issues.append("Response contains no citations")

    # Check for out-of-range citations
    for cite in citations:
        if cite > num_evidence_chunks:
            issues.append(f"Citation [Source {cite}] exceeds available sources (max: {num_evidence_chunks})")

    # Check for potentially unsupported claims (heuristic)
    sentences = response.split('.')
    uncited_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Skip very short fragments
            # Check if sentence contains a citation marker
            if '[source' not in sentence.lower():
                uncited_sentences.append(sentence[:50] + '...')

    if len(uncited_sentences) > len(sentences) * 0.3:  # >30% uncited
        warnings.append(f"Many sentences lack citations ({len(uncited_sentences)}/{len(sentences)})")

    return {
        'valid': len(issues) == 0,
        'abstained': False,
        'issues': issues,
        'warnings': warnings,
        'num_citations': len(citations),
        'cited_sources': citations
    }


if __name__ == "__main__":
    # Example usage
    example_question = "What are the symptoms of type 2 diabetes?"

    example_chunks = [
        {
            "text": "Type 2 diabetes symptoms include increased thirst, frequent urination, and unexplained weight loss.",
            "source_id": "PMID:11111111",
            "source": "pubmed"
        },
        {
            "text": "Patients with type 2 diabetes may also experience fatigue, blurred vision, and slow-healing wounds.",
            "source_id": "PMID:22222222",
            "source": "pubmed"
        }
    ]

    # Build prompt
    prompt = build_prompt(example_question, example_chunks, template="strict")

    print("="*80)
    print("SYSTEM PROMPT:")
    print("="*80)
    print(prompt['system'])
    print("\n" + "="*80)
    print("USER PROMPT:")
    print("="*80)
    print(prompt['user'])

    # Example response validation
    example_response = "Type 2 diabetes presents with increased thirst, frequent urination, and unexplained weight loss [Source 1]. Additional symptoms include fatigue, blurred vision, and slow-healing wounds [Source 2]."

    print("\n" + "="*80)
    print("VALIDATION:")
    print("="*80)
    validation = validate_response(example_response, len(example_chunks))
    print(f"Valid: {validation['valid']}")
    print(f"Citations: {validation['cited_sources']}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
