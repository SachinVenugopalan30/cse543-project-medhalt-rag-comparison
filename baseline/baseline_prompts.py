#!/usr/bin/env python3
"""
Baseline prompts for LLM evaluation without RAG.

Simple prompts that don't provide evidence - used to measure baseline hallucination.
"""

# Zero-shot baseline prompt
ZERO_SHOT_SYSTEM_PROMPT = """You are a helpful medical AI assistant. Answer medical questions accurately and concisely based on your training knowledge.

For multiple choice questions, respond with ONLY the letter of your chosen answer (A, B, C, or D)."""

ZERO_SHOT_USER_TEMPLATE = """Question: {question}

Answer:"""

ZERO_SHOT_MC_TEMPLATE = """Question: {question}

Options:
{options}

Select the correct answer. Respond with ONLY the letter (A, B, C, or D):"""

# Few-shot baseline prompt with examples
FEW_SHOT_SYSTEM_PROMPT = """You are a medical AI assistant. Answer medical questions accurately based on established medical knowledge.

Provide clear, evidence-based answers. If you're uncertain about something, acknowledge the uncertainty."""

FEW_SHOT_EXAMPLES = """Here are some examples:

Q: What is the normal range for blood glucose levels?
A: Normal fasting blood glucose levels are typically between 70-100 mg/dL (3.9-5.6 mmol/L). Levels between 100-125 mg/dL may indicate prediabetes, while levels of 126 mg/dL or higher on two separate tests indicate diabetes.

Q: What are the primary symptoms of appendicitis?
A: The primary symptoms of appendicitis include abdominal pain (often starting near the belly button and moving to the lower right abdomen), loss of appetite, nausea and vomiting, low-grade fever, and inability to pass gas. The pain typically worsens over 6-12 hours. Appendicitis is a medical emergency requiring immediate attention.

Q: What is the mechanism of action of statins?
A: Statins work by inhibiting HMG-CoA reductase, an enzyme involved in cholesterol synthesis in the liver. This reduces the production of cholesterol and leads to increased uptake of LDL cholesterol from the bloodstream, thereby lowering total cholesterol and LDL levels.

---

Now answer this question:"""

FEW_SHOT_USER_TEMPLATE = """{question}

Answer:"""


def format_options(options: dict) -> str:
    """
    Format multiple choice options for display.

    Args:
        options: Dict with keys '0', '1', '2', '3' mapping to answer choices

    Returns:
        Formatted string like "A) Option 1\nB) Option 2\n..."
    """
    if not options:
        return ""

    letters = ['A', 'B', 'C', 'D']
    formatted = []

    for i, letter in enumerate(letters):
        option_key = str(i)
        if option_key in options and options[option_key]:
            # Skip the 'correct answer' key
            if option_key != 'correct answer':
                formatted.append(f"{letter}) {options[option_key]}")

    return "\n".join(formatted)


def build_baseline_prompt(
    question: str,
    mode: str = "zero-shot",
    options: dict = None
) -> dict:
    """
    Build baseline prompt without RAG evidence.

    Args:
        question: Medical question
        mode: "zero-shot" or "few-shot"
        options: Multiple choice options (optional)

    Returns:
        Dict with 'system' and 'user' prompts
    """
    if mode == "few-shot":
        system_prompt = FEW_SHOT_SYSTEM_PROMPT
        user_prompt = FEW_SHOT_EXAMPLES + "\n" + FEW_SHOT_USER_TEMPLATE.format(question=question)
    else:  # zero-shot
        system_prompt = ZERO_SHOT_SYSTEM_PROMPT

        # Use multiple choice template if options provided
        if options:
            formatted_options = format_options(options)
            user_prompt = ZERO_SHOT_MC_TEMPLATE.format(
                question=question,
                options=formatted_options
            )
        else:
            user_prompt = ZERO_SHOT_USER_TEMPLATE.format(question=question)

    return {
        "system": system_prompt,
        "user": user_prompt
    }


# Med-HALT specific prompt (if needed)
MEDHALT_SYSTEM_PROMPT = """You are a medical AI assistant being evaluated for accuracy and hallucination.

Answer the following medical question based on your knowledge. Be as accurate as possible. If you are uncertain or don't know the answer, you may say "I don't know" or "I'm not certain.

For multiple choice questions, respond with ONLY the letter of your chosen answer (A, B, C, or D)."""

MEDHALT_USER_TEMPLATE = """Medical Question: {question}

Provide your answer:"""

MEDHALT_MC_TEMPLATE = """Medical Question: {question}

Options:
{options}

Select the correct answer. Respond with ONLY the letter (A, B, C, or D):"""


def build_medhalt_baseline_prompt(question: str, options: dict = None) -> dict:
    """
    Build prompt for Med-HALT baseline evaluation.

    Args:
        question: Med-HALT question
        options: Multiple choice options (optional)

    Returns:
        Dict with 'system' and 'user' prompts
    """
    system_prompt = MEDHALT_SYSTEM_PROMPT

    if options:
        formatted_options = format_options(options)
        user_prompt = MEDHALT_MC_TEMPLATE.format(
            question=question,
            options=formatted_options
        )
    else:
        user_prompt = MEDHALT_USER_TEMPLATE.format(question=question)

    return {
        "system": system_prompt,
        "user": user_prompt
    }


if __name__ == "__main__":
    # Example usage
    example_question = "What are the contraindications for metformin?"

    print("="*80)
    print("ZERO-SHOT BASELINE")
    print("="*80)
    zero_shot = build_baseline_prompt(example_question, mode="zero-shot")
    print("\nSystem:", zero_shot['system'])
    print("\nUser:", zero_shot['user'])

    print("\n" + "="*80)
    print("FEW-SHOT BASELINE")
    print("="*80)
    few_shot = build_baseline_prompt(example_question, mode="few-shot")
    print("\nSystem:", few_shot['system'])
    print("\nUser:", few_shot['user'])

    print("\n" + "="*80)
    print("MED-HALT BASELINE")
    print("="*80)
    medhalt = build_medhalt_baseline_prompt(example_question)
    print("\nSystem:", medhalt['system'])
    print("\nUser:", medhalt['user'])
