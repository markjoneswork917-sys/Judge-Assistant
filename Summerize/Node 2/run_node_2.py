import json
import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from node_2 import Node2_BulletExtractor

# Load environment variables (for GROQ_API_KEY)
load_dotenv()


# Sample classified chunks for testing (Node 1 output format)
SAMPLE_INPUT = {
    "classified_chunks": [
        {
            "chunk_id": "test-chunk-001",
            "doc_id": "صحيفة دعوى.txt",
            "page_number": 1,
            "paragraph_number": 1,
            "clean_text": "بتاريخ 5/6/2021 أبرم المدعي عقد بيع مع المدعى عليه بشأن شقة سكنية بمساحة 120 متراً مربعاً بمبلغ إجمالي قدره 500,000 جنيه مصري، سدد منها المدعي مبلغ 200,000 جنيه كمقدم تعاقد.",
            "doc_type": "صحيفة دعوى",
            "party": "المدعي",
            "role": "الوقائع",
            "confidence": 1.0
        },
        {
            "chunk_id": "test-chunk-002",
            "doc_id": "صحيفة دعوى.txt",
            "page_number": 1,
            "paragraph_number": 2,
            "clean_text": "امتنع المدعى عليه عن تسليم الشقة المبيعة في الموعد المتفق عليه وهو 1/1/2022 رغم إنذاره رسمياً بتاريخ 15/2/2022 على يد محضر.",
            "doc_type": "صحيفة دعوى",
            "party": "المدعي",
            "role": "الوقائع",
            "confidence": 1.0
        },
        {
            "chunk_id": "test-chunk-003",
            "doc_id": "صحيفة دعوى.txt",
            "page_number": 2,
            "paragraph_number": 3,
            "clean_text": "يلتمس المدعي الحكم بإلزام المدعى عليه بتسليم الشقة محل التعاقد وتعويض المدعي بمبلغ 50,000 جنيه عن الأضرار المادية والأدبية.",
            "doc_type": "صحيفة دعوى",
            "party": "المدعي",
            "role": "الطلبات",
            "confidence": 1.0
        },
        {
            "chunk_id": "test-chunk-004",
            "doc_id": "صحيفة دعوى.txt",
            "page_number": 2,
            "paragraph_number": 4,
            "clean_text": "سند المدعي في دعواه المادة 418 من القانون المدني التي تنص على أن البيع عقد يلتزم به البائع أن ينقل للمشتري ملكية شيء أو حقاً مالياً آخر في مقابل ثمن نقدي.",
            "doc_type": "صحيفة دعوى",
            "party": "المدعي",
            "role": "الأساس القانوني",
            "confidence": 1.0
        }
    ]
}


def main():
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nWARNING: GROQ_API_KEY not found in environment variables.")
        print("Ensure you have a .env file or set the variable.")

    try:
        # Initialize LLM
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        node_2 = Node2_BulletExtractor(llm)
    except Exception as e:
        print(f"Failed to initialize Node 2 or LLM: {e}")
        return

    # Use sample data or load from file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
        print(f"\n--- Loaded input from: {file_path} ---")
    else:
        input_data = SAMPLE_INPUT
        print("\n--- Using sample classified chunks ---")

    chunks = input_data.get("classified_chunks", [])
    print(f"Input chunks: {len(chunks)}")

    # Process
    print("\n--- Extracting Bullets ---")
    try:
        result = node_2.process(input_data)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Output Results
    bullets = result.get("bullets", [])

    if not bullets:
        print("No bullets extracted.")
        return

    print(f"\n=== Extracted Bullets ({len(bullets)}) ===")
    for i, bullet in enumerate(bullets):
        print(f"\n[Bullet {i + 1}]")
        print(f"  ID:     {bullet.get('bullet_id')}")
        print(f"  Role:   {bullet.get('role')}")
        print(f"  Party:  {bullet.get('party')}")
        print(f"  Source: {bullet.get('source')}")
        print(f"  Chunk:  {bullet.get('chunk_id')}")
        print(f"  Text:   {bullet.get('bullet')}")
        print("-" * 50)

    # Optionally save output
    output_path = os.path.join(os.path.dirname(__file__), "node_2_output.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nOutput saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save output file: {e}")


if __name__ == "__main__":
    main()
