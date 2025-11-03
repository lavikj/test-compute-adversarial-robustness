"""Check what the attack prompt looks like (no API calls)."""
from data.gen_math import sample_add
from attacks.many_shot import build_many_shot_prompt, get_attacker_goal_value

# Generate one problem
problems = sample_add(1, digits=2, seed=42)
question, answer = problems[0]

print("="*80)
print("CHECKING ATTACK PROMPT FORMAT")
print("="*80)
print(f"\nOriginal problem: {question}")
print(f"Correct answer: {answer}")

# Build attack for output_42 with 1000 tokens
attacker_goal = "output_42"
goal_value = get_attacker_goal_value(answer, attacker_goal)

print(f"\nAttack goal: {attacker_goal}")
print(f"Target value (what attacker wants): {goal_value}")

prompt = build_many_shot_prompt(
    question=question,
    answer=answer,
    attacker_goal=attacker_goal,
    attacker_strength_tokens=1000,
    seed=42
)

print("\n" + "="*80)
print("FULL PROMPT:")
print("="*80)
print(prompt)
print("="*80)
print(f"\nPrompt length: {len(prompt)} characters")
print(f"Number of example lines: {prompt.count('=')}")

# Count how many times "42" appears in the attack examples
attack_section = prompt.split("<BEGIN IGNORE>")[1].split("</END IGNORE>")[0]
count_42 = attack_section.count("= 42")
print(f"Number of '= 42' patterns in attack: {count_42}")

