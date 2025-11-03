"""Integration with the real MATH dataset from Hendrycks et al."""
from datasets import load_dataset
from typing import List, Tuple, Optional
import re


def load_hendrycks_math(
    n: int,
    subset: Optional[str] = None,
    split: str = "test",
    seed: Optional[int] = None
) -> List[Tuple[str, int]]:
    """
    Load problems from the Hendrycks MATH dataset.
    
    Args:
        n: Number of problems to load
        subset: Specific subset to use (algebra, counting_and_probability, etc.)
                If None, samples from all subsets
        split: "train" or "test"
        seed: Random seed for sampling
        
    Returns:
        List of (question_str, answer_int) tuples
        
    Note:
        Only returns problems with integer answers. Some MATH problems
        have non-integer answers and are filtered out.
    """
    try:
        # Load dataset from HuggingFace
        if subset:
            dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)
        else:
            # Load all subsets and concatenate
            # Available subsets: algebra, counting_and_probability, geometry, 
            # intermediate_algebra, number_theory, prealgebra, precalculus
            all_subsets = [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
            ]
            datasets = []
            for subset_name in all_subsets:
                ds = load_dataset("EleutherAI/hendrycks_math", subset_name, split=split)
                datasets.append(ds)
            
            # Concatenate all datasets
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MATH dataset. Make sure you have internet connection "
            f"and datasets library installed: pip install datasets\nError: {e}"
        )
    
    # Shuffle dataset
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    problems = []
    
    for item in dataset:
        if len(problems) >= n:
            break
        
        problem = item['problem']
        solution = item['solution']
        
        # Try to extract the final answer (usually in \boxed{...})
        answer = extract_answer(solution)
        
        if answer is not None and isinstance(answer, int):
            problems.append((problem, answer))
    
    # If we couldn't find enough problems with integer answers, warn user
    if len(problems) < n:
        print(f"Warning: Only found {len(problems)} problems with integer answers out of {n} requested")
    
    return problems[:n]


def extract_answer(solution: str) -> Optional[int]:
    """
    Extract the final answer from a MATH solution string.
    
    The solution typically contains the answer in \boxed{...}.
    This function tries to extract and convert it to an integer.
    
    Args:
        solution: Solution string from MATH dataset
        
    Returns:
        Integer answer if found and parseable, None otherwise
    """
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, solution)
    
    if not matches:
        return None
    
    # Get the last boxed answer (final answer)
    answer_str = matches[-1]
    
    # Try to extract integer from answer string
    # Remove common formatting
    answer_str = answer_str.replace(',', '')  # Remove thousands separators
    answer_str = answer_str.replace('$', '')  # Remove dollar signs
    answer_str = answer_str.strip()
    
    # Handle negative numbers
    negative = answer_str.startswith('-')
    if negative:
        answer_str = answer_str[1:]
    
    # Try to extract just the number
    number_match = re.search(r'\d+', answer_str)
    if number_match:
        try:
            value = int(number_match.group())
            return -value if negative else value
        except ValueError:
            return None
    
    return None


def sample_math_hendrycks(n: int, seed: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Convenience function matching the signature of sample_math() in gen_math.py
    but using the real Hendrycks MATH dataset.
    
    Args:
        n: Number of problems to sample
        seed: Random seed
        
    Returns:
        List of (question_str, answer_int) tuples
    """
    return load_hendrycks_math(n=n, split="test", seed=seed)


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing MATH dataset loading...")
    
    try:
        problems = load_hendrycks_math(n=5, seed=42)
        
        print(f"\nLoaded {len(problems)} problems:")
        for i, (question, answer) in enumerate(problems, 1):
            print(f"\n{i}. Question: {question[:100]}...")
            print(f"   Answer: {answer}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to install: pip install datasets")

