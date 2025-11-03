#!/usr/bin/env python3
"""
Convert ICD-10 codes from full format to simplified CSV.

Input format (semicolon-delimited):
Field 0: Level
Field 6: ICD Code (e.g., D59.0)
Field 8: Description

Output format:
CODE,DESCRIPTION
"""

import sys
from pathlib import Path

def convert_icd10_to_csv(input_file: Path, output_file: Path):
    """Convert ICD-10 full format to simplified CSV."""

    codes_written = 0

    with open(output_file, 'w', encoding='utf-8') as outf:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as inf:
            for line_num, line in enumerate(inf, 1):
                line = line.strip()
                if not line:
                    continue

                # Split by semicolon
                parts = line.split(';')

                # Need at least 9 fields
                if len(parts) < 9:
                    continue

                # Extract code (field 6) and description (field 8)
                code = parts[6].strip()
                description = parts[8].strip()

                # Skip if empty
                if not code or not description:
                    continue

                # Write to CSV
                outf.write(f"{code},{description}\n")
                codes_written += 1

    print(f"Converted {codes_written} ICD-10 codes to {output_file}")
    return codes_written

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data/raw/icd10/icd102019syst_codes.txt"
    output_file = base_dir / "data/raw/icd10/icd10_codes_full.csv"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Converting {input_file}")
    print(f"Output: {output_file}")

    codes_written = convert_icd10_to_csv(input_file, output_file)

    print(f"\nSuccess! Generated {output_file} with {codes_written} codes")
    print("\nTo use this file, update your commands to use:")
    print(f"  --icd {output_file}")

if __name__ == "__main__":
    main()
