#!/usr/bin/env python3
import json
import re

def remove_citations_from_jsonl(input_file, output_file):
    """
    Remove all occurrences of ' :contentReference' and everything after it up to the next '}' or end of line, anywhere in the line.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        # Remove all occurrences of ' :contentReference...' up to the next '}' or end of line
        # Handles multiple citations per line
        line = re.sub(r' :contentReference[^}}]*\}', '}', line)
        # Also remove any trailing ' :contentReference...' at end of line (no closing brace)
        line = re.sub(r' :contentReference[^\n\r]*$', '', line)
        processed_lines.append(line.rstrip())
    with open(output_file, 'w', encoding='utf-8') as f:
        for pline in processed_lines:
            f.write(pline + '\n')
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    input_file = "serenity_qa_batch1_clean.jsonl"
    output_file = "serenity_qa_batch1_clean_no_citations.jsonl"
    remove_citations_from_jsonl(input_file, output_file) 