import json
import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from node_5 import Node5_BriefGenerator

# Load environment variables (for GROQ_API_KEY)
load_dotenv()


# Sample Node 4B output for testing
SAMPLE_INPUT = {
    "role_theme_summaries": [
        {
            "role": "الوقائع",
            "theme_summaries": [
                {
                    "theme": "الوقائع التعاقدية",
                    "summary": "أبرم الطرفان عقد بيع ابتدائي بتاريخ 5/6/2021 بشأن شقة سكنية بالعقار رقم 15 شارع النيل بمبلغ إجمالي قدره 500,000 جنيه مصري. تم تسجيل العقد لدى الشهر العقاري بتاريخ 10/6/2021 وسدد المدعي دفعة أولى بمبلغ 200,000 جنيه بموجب إيصال مؤرخ 5/6/2021.\n\nيتمسك المدعي بأنه سدد مبلغ 400,000 جنيه من إجمالي الثمن، بينما يدفع المدعى عليه بأنه لم يتسلم سوى الدفعة الأولى بمبلغ 200,000 جنيه فقط ولا يوجد ما يثبت سداد أي مبالغ إضافية.",
                    "key_disputes": [
                        "مقدار المبالغ المسددة من الثمن",
                        "تاريخ التسليم المتفق عليه"
                    ],
                    "sources": [
                        "صحيفة دعوى.txt ص1 ف1",
                        "مذكرة دفاع.txt ص1 ف1",
                        "صحيفة دعوى.txt ص1 ف3",
                        "مذكرة دفاع.txt ص1 ف2",
                        "صحيفة دعوى.txt ص2 ف1",
                        "مذكرة دفاع.txt ص2 ف1"
                    ]
                },
                {
                    "theme": "الوقائع المادية والأضرار",
                    "summary": "يدعي المدعي أن العقار غير مطابق للمواصفات المتفق عليها ويوجد عيوب إنشائية، بينما يؤكد المدعى عليه أن العقار سليم ومطابق وتم معاينته قبل التعاقد. كما يدعي المدعي أنه تعرض لأضرار مادية جسيمة نتيجة التأخير تقدر بمبلغ 50,000 جنيه واضطر لاستئجار مسكن بديل بإيجار شهري 3,000 جنيه.\n\nمن جانبه، يتمسك المدعى عليه بتعرضه لظروف قاهرة تمثلت في ارتفاع أسعار مواد البناء بنسبة 40% وأنه أخطر المدعي بالتأخير بخطاب مسجل بتاريخ 15/11/2021.",
                    "key_disputes": [
                        "حالة العقار عند المعاينة",
                        "مدى توافر القوة القاهرة"
                    ],
                    "sources": [
                        "صحيفة دعوى.txt ص4 ف1",
                        "مذكرة دفاع.txt ص4 ف1",
                        "صحيفة دعوى.txt ص4 ف2",
                        "صحيفة دعوى.txt ص4 ف3",
                        "مذكرة دفاع.txt ص4 ف2",
                        "مذكرة دفاع.txt ص5 ف1"
                    ]
                }
            ]
        },
        {
            "role": "الأساس القانوني",
            "theme_summaries": [
                {
                    "theme": "الأساس القانوني للفسخ والتعويض",
                    "summary": "يستند المدعي إلى المادة 418 من القانون المدني بشأن التزام البائع بنقل الملكية والمادة 157 بشأن الفسخ القضائي لعدم التنفيذ، فضلاً عن حكم محكمة النقض في الطعن رقم 1234 لسنة 85 ق بشأن التزام البائع بالتسليم والمادة 215 بشأن التعويض عن عدم التنفيذ.\n\nبينما يدفع المدعى عليه بنص المادة 160 بشأن انفساخ العقد بقوة القانون ويتمسك بالمادة 165 بشأن القوة القاهرة كسبب أجنبي، مع الاستناد إلى المادة 373 بشأن شروط القوة القاهرة ومبدأ حسن النية في تنفيذ العقود وفقاً للمادة 148 مدني.",
                    "key_disputes": [
                        "الأساس القانوني للفسخ",
                        "مدى انطباق أحكام القوة القاهرة"
                    ],
                    "sources": [
                        "صحيفة دعوى.txt ص5 ف1",
                        "صحيفة دعوى.txt ص5 ف2",
                        "صحيفة دعوى.txt ص6 ف1",
                        "صحيفة دعوى.txt ص6 ف2",
                        "مذكرة دفاع.txt ص5 ف2",
                        "مذكرة دفاع.txt ص5 ف3",
                        "مذكرة دفاع.txt ص6 ف1",
                        "مذكرة دفاع.txt ص6 ف2"
                    ]
                }
            ]
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
        node_5 = Node5_BriefGenerator(llm)
    except Exception as e:
        print(f"Failed to initialize Node 5 or LLM: {e}")
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
        print("\n--- Using sample Node 4B output ---")

    # Print input summary
    role_summaries = input_data.get("role_theme_summaries", [])
    total_themes = sum(
        len(rs.get("theme_summaries", []))
        for rs in role_summaries
    )
    print(f"Input roles: {len(role_summaries)}, Total themes: {total_themes}")

    # Run Node 5
    print("\n" + "=" * 60)
    print("Node 5: Judge-Facing Case Brief")
    print("=" * 60)
    try:
        result = node_5.process(input_data)
    except Exception as e:
        print(f"Error during Node 5 processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print the rendered brief
    rendered = result.get("rendered_brief", "")
    if rendered:
        print("\n" + "=" * 60)
        print("RENDERED BRIEF:")
        print("=" * 60)
        print(rendered)

    # Print sources
    all_sources = result.get("all_sources", [])
    print(f"\nTotal unique sources: {len(all_sources)}")

    # Save JSON output
    output_json_path = os.path.join(os.path.dirname(__file__), "node_5_output.json")
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nJSON output saved to: {output_json_path}")
    except Exception as e:
        print(f"Warning: Could not save JSON output: {e}")

    # Save rendered brief as markdown
    output_md_path = os.path.join(os.path.dirname(__file__), "case_brief.md")
    try:
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"Rendered brief saved to: {output_md_path}")
    except Exception as e:
        print(f"Warning: Could not save rendered brief: {e}")


if __name__ == "__main__":
    main()
