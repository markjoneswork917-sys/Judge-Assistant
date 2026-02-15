import argparse
import os
from langchain_groq import ChatGroq
from node_0 import Node0_DocumentIntake
from dotenv import load_dotenv

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

def main():


    file_path = r"D:\FUCK!!\Grad\Code\Docs\صحيفة دعوى.txt"
    
    # 1. Read the file
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Try UTF-8 first
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        print("Warning: Could not decode as UTF-8. Trying 'latin-1'...")
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"\n--- Loaded file: {file_path} ---")
    print(f"Length: {len(raw_text)} characters")

    # 2. Setup Node 0
    # Check for API key but don't crash immediately; let langchain handle missing key if it must
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nWARNING: GROQ_API_KEY not found in environment variables.")
        print("Ensure you have a .env file or set the variable.")
    
    try:
        # Initialize LLM
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        node_0 = Node0_DocumentIntake(llm)
    except Exception as e:
        print(f"Failed to initialize Node 0 or LLM: {e}")
        return

    # 3. Process
    inputs = {
        "raw_text": raw_text,
        "doc_id": os.path.basename(file_path)
    }

    print("\n--- Processing Document ---")
    try:
        result = node_0.process(inputs)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Output Results
    chunks = result.get("chunks", [])
    
    if not chunks:
        print("No chunks generated.")
        return

    # Extract global metadata from the first chunk (assuming consistent for the doc)
    first_chunk = chunks[0]
    print("\n=== Extracted Metadata ===")
    print(f"Document Type: {first_chunk.get('doc_type', 'N/A')}")
    print(f"Party: {first_chunk.get('party', 'N/A')}")
    
    print(f"\n=== Generated Chunks ({len(chunks)}) ===")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}]")
        print(f"ID: {chunk.get('chunk_id')}")
        print(f"Page (Est): {chunk.get('page_number')}")
        print(f"Para: {chunk.get('paragraph_number')}")
        content = chunk.get('clean_text', '')
        # Preview first 100 chars
        preview = content.replace('\n', ' ')
        print(f"Content Preview: {preview}...")
        print("-" * 40)

if __name__ == "__main__":
    main()
