"""Test that few-shot examples are being generated."""
from attacks.many_shot import build_many_shot_prompt

# Test output_42
print("="*80)
print("TEST 1: output_42 with 500 tokens")
print("="*80)
prompt = build_many_shot_prompt(
    question="50 + 50 =",
    answer=100,
    attacker_goal="output_42",
    attacker_strength_tokens=500,
    seed=42
)
print(prompt)
print(f"\nNumber of '= 42' in prompt: {prompt.count('= 42')}")
print(f"Number of examples: {prompt.count('Answer')}")

# Test answer_plus_1
print("\n" + "="*80)
print("TEST 2: answer_plus_1 with 300 tokens")
print("="*80)
prompt = build_many_shot_prompt(
    question="20 + 30 =",
    answer=50,
    attacker_goal="answer_plus_1",
    attacker_strength_tokens=300,
    seed=42
)
print(prompt[:1000])  # First 1000 chars
print("\n...")
print(f"\nNumber of '+ 1' in prompt: {prompt.count('+ 1')}")
print(f"Number of examples: {prompt.count('Answer')}")

# Test answer_times_7
print("\n" + "="*80)
print("TEST 3: answer_times_7 with 300 tokens")
print("="*80)
prompt = build_many_shot_prompt(
    question="10 + 5 =",
    answer=15,
    attacker_goal="answer_times_7",
    attacker_strength_tokens=300,
    seed=42
)
print(prompt[:1000])  # First 1000 chars
print("\n...")
print(f"\nNumber of 'x 7' in prompt: {prompt.count('x 7')}")
print(f"Number of examples: {prompt.count('Answer')}")

