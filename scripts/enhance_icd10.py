#!/usr/bin/env python3
"""
Enhance ICD-10 CSV with American spelling variants for better entity matching.

Converts British spellings to include American variants:
- anaemia -> anemia
- haemolytic -> hemolytic
- haematology -> hematology
- oesophagus -> esophagus
etc.
"""

import sys
from pathlib import Path
import re

# British to American spelling patterns
SPELLING_REPLACEMENTS = [
    (r'\banaemia\b', 'anemia'),
    (r'\banaemias\b', 'anemias'),
    (r'\bhaemolytic\b', 'hemolytic'),
    (r'\bhaematology\b', 'hematology'),
    (r'\bhaematoma\b', 'hematoma'),
    (r'\bhaemorrhage\b', 'hemorrhage'),
    (r'\bhaemorrhagic\b', 'hemorrhagic'),
    (r'\boedema\b', 'edema'),
    (r'\boesophag', 'esophag'),
    (r'\bdiarrhoea\b', 'diarrhea'),
    (r'\bleukaemia\b', 'leukemia'),
    (r'\bfoetus\b', 'fetus'),
    (r'\bfoetal\b', 'fetal'),
]

def british_to_american(text: str) -> str:
    """Convert British spelling to American."""
    american = text
    for pattern, replacement in SPELLING_REPLACEMENTS:
        american = re.sub(pattern, replacement, american, flags=re.IGNORECASE)
    return american

def remove_modifiers(text: str) -> str:
    """Remove common medical modifiers to improve matching."""
    # Remove leading modifiers
    prefixes_to_remove = [
        r'^other\s+',
        r'^drug-induced\s+',
        r'^acquired\s+',
        r'^congenital\s+',
        r'^hereditary\s+',
        r'^primary\s+',
        r'^secondary\s+',
        r'^chronic\s+',
        r'^acute\s+',
        r'^idiopathic\s+',
        r'^specified\s+',
        r'^unspecified\s+',
    ]

    cleaned = text
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

    # Remove trailing qualifiers like "unspecified", "NOS"
    suffixes_to_remove = [
        r',?\s+unspecified$',
        r',?\s+NOS$',
        r',?\s+so stated$',
    ]

    for suffix in suffixes_to_remove:
        cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)

    return cleaned.strip()

def enhance_icd10_csv(input_file: Path, output_file: Path):
    """
    Create enhanced ICD-10 CSV with American spelling variants and modifier-stripped versions.

    For each entry:
    1. Add original description
    2. Add American spelling variant (if different)
    3. Add modifier-stripped version for better entity matching
    """

    entries = []
    codes_added = 0
    seen_descriptions = set()  # Track unique (code, description) pairs

    # Read original entries
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',', 1)
            if len(parts) != 2:
                continue

            code, description = parts
            entries.append((code.strip(), description.strip()))

    # Write original + variants
    with open(output_file, 'w', encoding='utf-8') as f:
        for code, description in entries:
            variants = set()

            # Variant 1: Original (British spelling)
            variants.add(description)

            # Variant 2: American spelling
            american_desc = british_to_american(description)
            variants.add(american_desc)

            # Variant 3: Modifier-stripped with British spelling
            stripped_british = remove_modifiers(description)
            if stripped_british and len(stripped_british) > 5:  # Avoid too-short terms
                variants.add(stripped_british)

            # Variant 4: Modifier-stripped with American spelling
            stripped_american = remove_modifiers(american_desc)
            if stripped_american and len(stripped_american) > 5:
                variants.add(stripped_american)

            # Write all unique variants
            for variant in variants:
                # Avoid duplicate entries
                key = (code, variant.lower())
                if key not in seen_descriptions:
                    f.write(f"{code},{variant}\n")
                    seen_descriptions.add(key)
                    codes_added += 1

    print(f"Enhanced ICD-10 CSV with {codes_added} total entries (original + variants)")
    return codes_added

def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data/raw/icd10/icd10_codes_full.csv"
    output_file = base_dir / "data/raw/icd10/icd10_codes_enhanced.csv"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Run convert_icd10.py first to generate icd10_codes_full.csv")
        sys.exit(1)

    print(f"Enhancing {input_file}")
    print(f"Output: {output_file}")
    print(f"\nAdding American spelling variants for:")
    print("  - anaemia -> anemia")
    print("  - haemolytic -> hemolytic")
    print("  - oesophagus -> esophagus")
    print("  - etc.\n")

    codes_added = enhance_icd10_csv(input_file, output_file)

    print(f"\nSuccess! Generated {output_file}")
    print("\nTo use this enhanced file, update your commands to:")
    print(f"  --icd {output_file}")
    print("\nOr create a symlink:")
    print(f"  ln -sf icd10_codes_enhanced.csv data/raw/icd10/icd10_codes.csv")

if __name__ == "__main__":
    main()
