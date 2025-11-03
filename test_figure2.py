"""Quick test to verify Figure 2 implementation."""
import sys
import pandas as pd
from data.gen_math import sample_add, sample_mul, sample_math
from attacks.many_shot import build_many_shot_prompt, get_attacker_goal_value


def test_data_generation():
    """Test that all three task types generate problems correctly."""
    print("Testing data generation...")
    
    # Test addition
    add_problems = sample_add(5, digits=2, seed=42)
    assert len(add_problems) == 5
    assert all(isinstance(q, str) and isinstance(a, int) for q, a in add_problems)
    print(f"  ✓ Addition: {add_problems[0]}")
    
    # Test multiplication
    mul_problems = sample_mul(5, digits=2, seed=42)
    assert len(mul_problems) == 5
    assert all(isinstance(q, str) and isinstance(a, int) for q, a in mul_problems)
    print(f"  ✓ Multiplication: {mul_problems[0]}")
    
    # Test MATH problems
    math_problems = sample_math(5, seed=42)
    assert len(math_problems) == 5
    assert all(isinstance(q, str) and isinstance(a, int) for q, a in math_problems)
    print(f"  ✓ MATH: {math_problems[0]}")
    
    print("✓ Data generation tests passed\n")


def test_attacker_goals():
    """Test that all three attacker goals work correctly."""
    print("Testing attacker goals...")
    
    correct_answer = 100
    
    # Test output_42
    goal_val = get_attacker_goal_value(correct_answer, "output_42")
    assert goal_val == 42
    print(f"  ✓ output_42: {goal_val}")
    
    # Test answer_plus_1
    goal_val = get_attacker_goal_value(correct_answer, "answer_plus_1")
    assert goal_val == 101
    print(f"  ✓ answer_plus_1: {goal_val}")
    
    # Test answer_times_7
    goal_val = get_attacker_goal_value(correct_answer, "answer_times_7")
    assert goal_val == 700
    print(f"  ✓ answer_times_7: {goal_val}")
    
    print("✓ Attacker goal tests passed\n")


def test_prompt_building():
    """Test that prompts are built correctly for all goals."""
    print("Testing prompt building...")
    
    question = "23 + 45 ="
    answer = 68
    
    for goal in ["output_42", "answer_plus_1", "answer_times_7"]:
        prompt = build_many_shot_prompt(
            question=question,
            answer=answer,
            attacker_goal=goal,
            attacker_strength_tokens=100,
            seed=42
        )
        assert isinstance(prompt, str)
        assert question in prompt
        assert "BEGIN IGNORE" in prompt
        assert "END IGNORE" in prompt
        print(f"  ✓ Prompt built for {goal} (length: {len(prompt)} chars)")
    
    print("✓ Prompt building tests passed\n")


def test_plotting_imports():
    """Test that plotting functions can be imported."""
    print("Testing plotting imports...")
    
    try:
        from eval.plotting import plot_figure2_grid
        print("  ✓ plot_figure2_grid imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import plot_figure2_grid: {e}")
        return False
    
    print("✓ Plotting import tests passed\n")
    return True


def test_csv_structure():
    """Test that we can create the expected DataFrame structure."""
    print("Testing CSV structure...")
    
    # Create sample data
    data = {
        'k': [1, 2, 4],
        'attacker_strength': [100, 100, 100],
        'attacker_goal': ['output_42', 'output_42', 'output_42'],
        'attack_success_rate': [0.8, 0.5, 0.2],
        'accuracy': [0.2, 0.5, 0.8],
        'variation': ['test', 'test', 'test'],
    }
    df = pd.DataFrame(data)
    
    # Check columns exist
    required_cols = ['k', 'attacker_strength', 'attacker_goal', 'attack_success_rate', 'accuracy']
    assert all(col in df.columns for col in required_cols)
    print(f"  ✓ DataFrame has all required columns: {required_cols}")
    
    # Check pivot works
    pivot = df.pivot_table(
        index='attacker_strength',
        columns='k',
        values='attack_success_rate',
        aggfunc='mean'
    )
    assert pivot.shape == (1, 3)  # 1 attack strength, 3 k values
    print(f"  ✓ Pivot table works correctly")
    
    print("✓ CSV structure tests passed\n")


def main():
    """Run all tests."""
    print("="*80)
    print("Figure 2 Implementation Test Suite")
    print("="*80)
    print()
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Attacker Goals", test_attacker_goals),
        ("Prompt Building", test_prompt_building),
        ("Plotting Imports", test_plotting_imports),
        ("CSV Structure", test_csv_structure),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            failed += 1
    
    print("="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("  1. Run: python run_figure2.py --n_samples 10  (quick test)")
        print("  2. Run: python run_figure2.py --n_samples 100 (full experiment)")
        sys.exit(0)


if __name__ == "__main__":
    main()

