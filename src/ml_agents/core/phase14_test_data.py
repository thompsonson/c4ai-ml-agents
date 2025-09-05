"""Test data module for Phase 14 reasoning approaches comparison study."""

from typing import Dict, List

# Basic questions from Phase 13 for baseline comparison
BASIC_QUESTIONS = [
    {"INPUT": "What is 2 + 2?", "OUTPUT": "4"},
    {"INPUT": "What is the capital of France?", "OUTPUT": "Paris"},
    {"INPUT": "Who won the world series in 2020?", "OUTPUT": "Los Angeles Dodgers"},
    {"INPUT": "What is the largest planet in our solar system?", "OUTPUT": "Jupiter"},
    {"INPUT": "How many days are in a week?", "OUTPUT": "7"},
]

# Reasoning-intensive questions for approach comparison
REASONING_QUESTIONS = [
    {
        "INPUT": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "OUTPUT": "No, we cannot conclude that some roses fade quickly. While all roses are flowers, we only know that some flowers fade quickly, not which specific types of flowers.",
    },
    {
        "INPUT": "A train travels 60 miles in 1 hour. How far will it travel in 2.5 hours at the same speed?",
        "OUTPUT": "150 miles (60 miles/hour × 2.5 hours = 150 miles)",
    },
    {
        "INPUT": "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "OUTPUT": "42 (the pattern is n × (n + 1), where n increases by 1 each time: 1×2=2, 2×3=6, 3×4=12, 4×5=20, 5×6=30, 6×7=42)",
    },
    {
        "INPUT": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "OUTPUT": "5 minutes (each machine makes 1 widget in 5 minutes, so 100 machines can make 100 widgets in 5 minutes)",
    },
    {
        "INPUT": "Mary has twice as many apples as John. John has 3 more apples than Sarah. If Sarah has 5 apples, how many does Mary have?",
        "OUTPUT": "16 apples (Sarah has 5, John has 5+3=8, Mary has 8×2=16)",
    },
]

# Extended reasoning challenges for comprehensive testing
EXTENDED_REASONING_QUESTIONS = [
    {
        "INPUT": "In a room, there are 3 switches outside and 3 light bulbs inside. You can only enter the room once. How can you determine which switch controls which bulb?",
        "OUTPUT": "Turn on the first switch for 10 minutes, then turn it off. Turn on the second switch and enter the room. The bulb that is on corresponds to the second switch. The bulb that is off but warm corresponds to the first switch. The bulb that is off and cool corresponds to the third switch.",
    },
    {
        "INPUT": "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. The boat can only carry the farmer and one item. The fox cannot be left alone with the chicken, and the chicken cannot be left alone with the grain. How does the farmer get everything across?",
        "OUTPUT": "1. Take the chicken across. 2. Return alone. 3. Take the fox across. 4. Bring the chicken back. 5. Take the grain across. 6. Return alone. 7. Take the chicken across again.",
    },
    {
        "INPUT": "If today is Wednesday and it rained 2 days ago, what day will it be 5 days from when it rained?",
        "OUTPUT": "Saturday (It rained on Monday, 5 days from Monday is Saturday)",
    },
    {
        "INPUT": "Three friends share a pizza. Alice eats 1/4, Bob eats 1/3, and Charlie eats the rest. What fraction did Charlie eat?",
        "OUTPUT": "5/12 (Alice: 1/4 = 3/12, Bob: 1/3 = 4/12, Charlie: 12/12 - 3/12 - 4/12 = 5/12)",
    },
    {
        "INPUT": "If a=3, b=5, and c=a+b, what is the value of 2c - a?",
        "OUTPUT": "13 (c = 3 + 5 = 8, so 2c - a = 2(8) - 3 = 16 - 3 = 13)",
    },
    {
        "INPUT": "A clock shows 3:15. What is the angle between the hour and minute hands?",
        "OUTPUT": "7.5 degrees (Hour hand moves 0.5° per minute: 15 × 0.5 = 7.5°. At 3:15, hour hand is 7.5° past 3, minute hand is at 15 minutes = 90°. Angle = 90° - (90° + 7.5°) = 7.5°)",
    },
    {
        "INPUT": "You have 12 balls, 11 are identical in weight, 1 is different (heavier or lighter). Using a balance scale only 3 times, how can you identify the different ball?",
        "OUTPUT": "Divide into groups of 4. Compare group A vs B: if balanced, different ball is in C; if unbalanced, it's in the heavier/lighter group. Then divide that group into 1+1+2, balance the singles - if balanced, weigh the 2 against 2 normal balls to determine if heavier or lighter.",
    },
    {
        "INPUT": "A snail climbs up a 10-meter wall. Each day it climbs 3 meters up, but each night it slides 2 meters down. On which day will it reach the top?",
        "OUTPUT": "Day 8 (Days 1-7: net +1m per day = 7m. Day 8: starts at 7m, climbs 3m to reach 10m before sliding back)",
    },
    {
        "INPUT": "In a group of 100 people, 70 like coffee, 80 like tea, and everyone likes at least one. How many like both coffee and tea?",
        "OUTPUT": "50 people (Using inclusion-exclusion: |Coffee ∪ Tea| = |Coffee| + |Tea| - |Coffee ∩ Tea|. So 100 = 70 + 80 - x, therefore x = 50)",
    },
    {
        "INPUT": "A rectangular garden is 20m long and 15m wide. If you walk diagonally across it, how far do you walk?",
        "OUTPUT": "25 meters (Using Pythagorean theorem: √(20² + 15²) = √(400 + 225) = √625 = 25m)",
    },
]


def get_test_dataset(
    dataset_type: str = "all", sample_size: int = None
) -> List[Dict[str, str]]:
    """Get test dataset for Phase 14 experiments.

    Args:
        dataset_type: Type of questions to include ('basic', 'reasoning', 'extended', 'all')
        sample_size: Number of samples to return (None for all)

    Returns:
        List of dictionaries with INPUT and OUTPUT keys
    """
    datasets = {
        "basic": BASIC_QUESTIONS,
        "reasoning": REASONING_QUESTIONS,
        "extended": EXTENDED_REASONING_QUESTIONS,
        "all": BASIC_QUESTIONS + REASONING_QUESTIONS + EXTENDED_REASONING_QUESTIONS,
    }

    if dataset_type not in datasets:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    data = datasets[dataset_type]

    if sample_size and sample_size < len(data):
        return data[:sample_size]

    return data


def get_evaluation_rubric() -> Dict[str, Dict[str, float]]:
    """Get evaluation rubric for Phase 14 experiments.

    Returns:
        Dictionary mapping criteria to weights and descriptions
    """
    return {
        "correctness": {
            "weight": 0.4,
            "description": "Did the approach reach the right answer?",
        },
        "reasoning_quality": {
            "weight": 0.3,
            "description": "Was the reasoning logical and clear?",
        },
        "efficiency": {
            "weight": 0.2,
            "description": "Token usage and response time",
        },
        "robustness": {
            "weight": 0.1,
            "description": "Handling of edge cases and ambiguity",
        },
    }


def get_approach_hypotheses() -> Dict[str, Dict[str, str]]:
    """Get hypotheses for each reasoning approach.

    Returns:
        Dictionary of approach names to expected performance metrics
    """
    return {
        "ChainOfThought": {
            "token_usage": "2-3x",
            "correctness_improvement": "20-30%",
            "response_time": "1.5x",
        },
        "TreeOfThought": {
            "token_usage": "3-5x",
            "correctness_improvement": "30-40%",
            "response_time": "2-3x",
        },
        "ReasoningAsPlanning": {
            "token_usage": "2-4x",
            "correctness_improvement": "25-35%",
            "response_time": "2x",
        },
    }


def create_local_test_dataset() -> Dict[str, List[Dict[str, str]]]:
    """Create LOCAL_TEST dataset for Phase 14 experiments.

    Returns:
        Dictionary with train split containing test questions
    """
    all_questions = get_test_dataset("all")

    # Add unique IDs to each question
    for i, question in enumerate(all_questions):
        question["id"] = f"LOCAL_TEST_{i+1:03d}"

    return {"train": all_questions}
