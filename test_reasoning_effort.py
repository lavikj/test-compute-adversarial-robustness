"""Test reasoning_effort parameter with o1."""
import os
from dotenv import load_dotenv
from models.openai_client import OpenAIClient

load_dotenv()

model = OpenAIClient(model_name="o1", api_key=os.getenv("OPENAI_API_KEY"))

prompt = "What is 2 + 2? Write a single number as the answer."

print("Testing reasoning_effort parameter...")
print("="*80)

for effort in ["low", "medium", "high"]:
    print(f"\nTesting reasoning_effort={effort}:")
    try:
        response = model.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.0,
            reasoning_effort=effort
        )
        print(f"  Success! Response: '{response}'")
        print(f"  Length: {len(response)} chars")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*80)
print("Testing without reasoning_effort:")
try:
    response = model.generate(
        prompt=prompt,
        max_tokens=50,
        temperature=0.0
    )
    print(f"  Success! Response: '{response}'")
    print(f"  Length: {len(response)} chars")
except Exception as e:
    print(f"  ERROR: {e}")

