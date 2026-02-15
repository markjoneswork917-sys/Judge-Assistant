import json
import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from node_3 import Node3_Aggregator

# Load environment variables (for GROQ_API_KEY)
load_dotenv()


# Sample bullets for testing (Node 2 output format)
# Covers: multi-party dispute, agreed facts, party-specific claims
SAMPLE_INPUT = {
    "bullets": [
        # --- الوقائع (Facts) - multi-party with agreements and disputes ---
        {
            "bullet_id": "fact-001",
            "role": "الوقائع",
            "bullet": "أبرم المدعي عقد بيع مع المدعى عليه بتاريخ 5/6/2021 بشأن شقة سكنية",
            "source": ["صحيفة دعوى.txt ص1 ف1"],
            "party": "المدعي",
            "chunk_id": "chunk-001"
        },
        {
            "bullet_id": "fact-002",
            "role": "الوقائع",
            "bullet": "لا ينكر المدعى عليه وجود عقد بيع مؤرخ 5/6/2021",
            "source": ["مذكرة دفاع.txt ص1 ف1"],
            "party": "المدعى عليه",
            "chunk_id": "chunk-010"
        },
        {
            "bullet_id": "fact-003",
            "role": "الوقائع",
            "bullet": "تم دفع مبلغ 500,000 جنيه كثمن للعقار",
            "source": ["صحيفة دعوى.txt ص1 ف3"],
            "party": "المدعي",
            "chunk_id": "chunk-002"
        },
        {
            "bullet_id": "fact-004",
            "role": "الوقائع",
            "bullet": "المبلغ المدفوع لا يتجاوز 300,000 جنيه",
            "source": ["مذكرة دفاع.txt ص2 ف1"],
            "party": "المدعى عليه",
            "chunk_id": "chunk-011"
        },
        {
            "bullet_id": "fact-005",
            "role": "الوقائع",
            "bullet": "يعاني المدعي من أضرار مادية نتيجة التأخير في التسليم",
            "source": ["صحيفة دعوى.txt ص2 ف2"],
            "party": "المدعي",
            "chunk_id": "chunk-003"
        },
        # --- الطلبات (Requests) - single party only ---
        {
            "bullet_id": "req-001",
            "role": "الطلبات",
            "bullet": "يطلب المدعي إلزام المدعى عليه بتسليم الشقة محل التعاقد",
            "source": ["صحيفة دعوى.txt ص3 ف1"],
            "party": "المدعي",
            "chunk_id": "chunk-004"
        },
        {
            "bullet_id": "req-002",
            "role": "الطلبات",
            "bullet": "يطلب المدعي تعويضاً بمبلغ 50,000 جنيه عن الأضرار المادية والأدبية",
            "source": ["صحيفة دعوى.txt ص3 ف2"],
            "party": "المدعي",
            "chunk_id": "chunk-005"
        },
        # --- الأساس القانوني (Legal Basis) - multi-party ---
        {
            "bullet_id": "legal-001",
            "role": "الأساس القانوني",
            "bullet": "يستند المدعي إلى المادة 418 من القانون المدني بشأن التزام البائع بنقل الملكية",
            "source": ["صحيفة دعوى.txt ص4 ف1"],
            "party": "المدعي",
            "chunk_id": "chunk-006"
        },
        {
            "bullet_id": "legal-002",
            "role": "الأساس القانوني",
            "bullet": "يدفع المدعى عليه بنص المادة 160 من القانون المدني بشأن الفسخ لعدم تنفيذ الالتزام",
            "source": ["مذكرة دفاع.txt ص3 ف2"],
            "party": "المدعى عليه",
            "chunk_id": "chunk-012"
        },
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
        node_3 = Node3_Aggregator(llm)
    except Exception as e:
        print(f"Failed to initialize Node 3 or LLM: {e}")
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
        print("\n--- Using sample bullets ---")

    bullets = input_data.get("bullets", [])
    print(f"Input bullets: {len(bullets)}")

    # Process
    print("\n--- Running Node 3 Aggregation ---")
    try:
        result = node_3.process(input_data)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Output Results
    aggregations = result.get("role_aggregations", [])

    if not aggregations:
        print("No role aggregations produced.")
        return

    print(f"\n=== Role Aggregations ({len(aggregations)}) ===")
    for agg in aggregations:
        role = agg.get("role", "?")
        agreed = agg.get("agreed", [])
        disputed = agg.get("disputed", [])
        party_specific = agg.get("party_specific", [])

        print(f"\n{'=' * 60}")
        print(f"Role: {role}")
        print(f"  Agreed: {len(agreed)} | Disputed: {len(disputed)} | Party-specific: {len(party_specific)}")

        if agreed:
            print(f"\n  --- Agreed ---")
            for i, item in enumerate(agreed, 1):
                print(f"  [{i}] {item.get('text')}")
                print(f"      Sources: {item.get('sources')}")

        if disputed:
            print(f"\n  --- Disputed ---")
            for i, item in enumerate(disputed, 1):
                print(f"  [{i}] Subject: {item.get('subject')}")
                for pos in item.get("positions", []):
                    print(f"      {pos.get('party')}:")
                    for bt in pos.get("bullets", []):
                        print(f"        - {bt}")
                    print(f"      Sources: {pos.get('sources')}")

        if party_specific:
            print(f"\n  --- Party-specific ---")
            for i, item in enumerate(party_specific, 1):
                print(f"  [{i}] [{item.get('party')}] {item.get('text')}")
                print(f"      Sources: {item.get('sources')}")

        print("-" * 60)

    # Optionally save output
    output_path = os.path.join(os.path.dirname(__file__), "node_3_output.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nOutput saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save output file: {e}")


if __name__ == "__main__":
    main()
