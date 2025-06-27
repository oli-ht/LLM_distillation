#!/usr/bin/env python3
"""
Data preparation script for fine-tuning Llama 3.2 1B on Serenity QA dataset.
This script converts JSONL QA pairs to instruction format that the model expects.
"""

import json

def convert_to_instruction_format(jsonl_file, output_file):
    """
    Convert JSONL QA pairs to instruction format for fine-tuning.
    
    Args:
        jsonl_file (str): Path to input JSONL file with QA pairs
        output_file (str): Path to output JSONL file in instruction format
    """
    
    instructions = []
    
    # Step 1: Read the JSONL file line by line
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines
            if not line.strip():
                continue
                
            try:
                # Parse each line as JSON
                data = json.loads(line.strip())
                
                # Extract question and answer
                question = data['question']
                answer = data['answer']
                
                # Step 2: Create instruction format
                instruction = {
                    "instruction": f"Based on the Serenity development project information, answer the following question: {question}",
                    "input": "",  # No additional context needed for this dataset
                    "output": answer
                }
                
                instructions.append(instruction)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key on line {line_num}: {e}")
                continue
    
    # Step 3: Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in instructions:
            f.write(json.dumps(item) + '\n')
    
    print(f"Successfully converted {len(instructions)} QA pairs to instruction format")
    print(f"Output saved to: {output_file}")
    
    return instructions

def preview_conversion(jsonl_file, num_samples=3):
    """
    Preview the conversion by showing a few examples.
    
    Args:
        jsonl_file (str): Path to input JSONL file
        num_samples (int): Number of samples to preview
    """
    
    print(f"\n=== PREVIEWING CONVERSION (first {num_samples} samples) ===")
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
                
            data = json.loads(line.strip())
            question = data['question']
            answer = data['answer']
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Original Question: {question}")
            print(f"Original Answer: {answer}")
            
            # Show what it will become
            instruction = f"Based on the Serenity development project information, answer the following question: {question}"
            print(f"Will become instruction: {instruction}")
            print(f"Output: {answer}")

def main():
    """Main function to run the data preparation."""
    
    # File paths
    input_file = "serenity_qas_dataset.jsonl"
    output_file = "serenity_instructions.jsonl"
    
    print("=== SERENITY QA DATASET PREPARATION ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Step 1: Preview the conversion
    try:
        preview_conversion(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    # Step 2: Convert the data
    print(f"\n=== CONVERTING DATA ===")
    instructions = convert_to_instruction_format(input_file, output_file)
    
    # Step 3: Show summary
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"Total QA pairs processed: {len(instructions)}")
    print(f"Output file created: {output_file}")
    
    # Step 4: Show a sample of the output
    print(f"\n=== SAMPLE OUTPUT FORMAT ===")
    if instructions:
        sample = instructions[0]
        print("First converted item:")
        print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
