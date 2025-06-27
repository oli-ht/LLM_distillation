#!/usr/bin/env python3
"""
Email document preparation for RAG (Retrieval-Augmented Generation).
This script processes email documents and prepares them for semantic search.
"""

import re
import json
from typing import List, Dict
from pathlib import Path

def clean_email_text(text: str) -> str:
    """
    Clean and normalize email text.
    
    Args:
        text (str): Raw email text
        
    Returns:
        str: Cleaned text
    """
    # Split into lines for better processing
    lines = text.split('\n')
    cleaned_lines = []
    
    # Track if we're in email body content
    in_body = False
    skip_next = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip email headers (but be more selective)
        if re.match(r'^(From|To|Subject|Date|Cc|Bcc):', line, re.IGNORECASE):
            continue
            
        # Skip Outlook-specific lines
        if 'Get Outlook for iOS' in line or 'Get Outlook for' in line:
            continue
            
        # Skip email separators
        if line.startswith('_') * 20 or line.startswith('=') * 20:
            continue
            
        # Skip email signatures (but be more careful)
        if line.startswith('--') and len(line) <= 5:
            continue
            
        # Skip very short lines that are likely formatting
        if len(line) <= 3 and line in ['--', '==', '**']:
            continue
            
        # If we find "Body:" or similar, mark that we're in the content
        if re.match(r'^(Body|Message):', line, re.IGNORECASE):
            in_body = True
            continue
            
        # If we find "Sent:" followed by date, we're likely in email metadata
        if re.match(r'^Sent:\s*\w+,\s*\w+\s+\d+', line):
            continue
            
        # If we find "From:" followed by email address, skip
        if re.match(r'^From:\s*\w+\s+<.*@.*>', line):
            continue
            
        # If we find "To:" followed by email addresses, skip
        if re.match(r'^To:\s*.*@.*', line):
            continue
            
        # If we find "Subject:", skip
        if re.match(r'^Subject:\s*', line):
            continue
            
        # If we find "Cc:", skip
        if re.match(r'^Cc:\s*', line):
            continue
            
        # If we find "Bcc:", skip
        if re.match(r'^Bcc:\s*', line):
            continue
            
        # If we find "Date:", skip
        if re.match(r'^Date:\s*', line):
            continue
            
        # Skip lines that are just email addresses
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', line):
            continue
            
        # Skip lines that are just URLs
        if re.match(r'^https?://', line):
            continue
            
        # If we get here, this is likely content we want to keep
        cleaned_lines.append(line)
    
    # Join lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Clean up multiple newlines
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

def process_email_file(email_file: str, output_file: str) -> Dict:
    """
    Process email file and create chunks for RAG.
    
    Args:
        email_file (str): Path to email text file
        output_file (str): Path to output JSON file
        
    Returns:
        Dict: Processing statistics
    """
    print(f"Processing email file: {email_file}")
    
    # Read the email file
    with open(email_file, 'r', encoding='utf-8') as f:
        email_text = f.read()
    
    print(f"Original file size: {len(email_text):,} characters")
    
    # Clean the text
    cleaned_text = clean_email_text(email_text)
    
    print(f"After cleaning: {len(cleaned_text):,} characters")
    print(f"Content preserved: {len(cleaned_text)/len(email_text)*100:.1f}%")
    
    # Split into chunks
    chunks = chunk_text(cleaned_text)
    
    # Create documents for RAG
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "id": f"email_chunk_{i}",
            "content": chunk,
            "source": email_file,
            "chunk_index": i
        }
        documents.append(doc)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    stats = {
        "total_chunks": len(chunks),
        "total_characters": len(cleaned_text),
        "avg_chunk_size": len(cleaned_text) // len(chunks) if chunks else 0,
        "original_size": len(email_text),
        "preservation_rate": len(cleaned_text) / len(email_text) * 100
    }
    
    print(f"Created {len(chunks)} chunks from email file")
    print(f"Output saved to: {output_file}")
    print(f"Statistics: {stats}")
    
    return stats

def preview_chunks(json_file: str, num_samples: int = 3):
    """
    Preview the created chunks.
    
    Args:
        json_file (str): Path to chunks JSON file
        num_samples (int): Number of samples to preview
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"\n=== PREVIEWING CHUNKS (first {num_samples} samples) ===")
    
    for i, doc in enumerate(documents[:num_samples]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {doc['id']}")
        print(f"Content preview: {doc['content'][:200]}...")
        print(f"Length: {len(doc['content'])} characters")

def main():
    """Main function to process email documents."""
    
    # File paths (you'll need to update these)
    email_file = "emails.txt"  # Your email text file
    output_file = "email_chunks.json"
    
    print("=== EMAIL DOCUMENT PREPARATION FOR RAG ===")
    print(f"Input file: {email_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not Path(email_file).exists():
        print(f"Error: Email file '{email_file}' not found!")
        print("Please create a text file with your emails or update the file path.")
        return
    
    # Process the email file
    stats = process_email_file(email_file, output_file)
    
    # Preview the results
    preview_chunks(output_file)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average chunk size: {stats['avg_chunk_size']} characters")
    print(f"Content preservation: {stats['preservation_rate']:.1f}%")

if __name__ == "__main__":
    main() 