"""
Dataset utilities for GEPA optimization.
Handles loading and splitting dataset_unified.csv into train/val sets.
"""
import argparse
import csv
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.DEBUG)


def load_tasks_from_csv(csv_path: str, web_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Load enabled tasks from CSV file optionally filter by web_names.
    
    Args:
        csv_path: Path to dataset_unified.csv or similar CSV with tasks
    
    Returns:
        List of task dictionaries with keys: id, input, ground_truth, web, etc.
    """
    tasks: List[Dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Only include enabled tasks
            if row.get("enabled", "true").lower() != "true":
                continue

            # If web_names filter is provided, skip tasks not in the list
            if web_names and row.get("web_name") not in web_names:
                continue

            tasks.append(row)

    
    logging.info(f"Loaded {len(tasks)} enabled tasks from {csv_path}")
    return tasks


def create_train_val_split(
    tasks: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split tasks into train and validation sets.
    
    Args:
        tasks: List of task dictionaries
        train_ratio: Fraction of tasks to use for training (default 0.7)
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_tasks, val_tasks)
    """
    random.seed(random_seed)
    
    # Shuffle and split
    shuffled = tasks.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_tasks = shuffled[:split_idx]
    val_tasks = shuffled[split_idx:]

    logging.info("Split dataset into %s train and %s val tasks", len(train_tasks), len(val_tasks))
    return train_tasks, val_tasks


def prepare_example_for_gepa(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a CSV task row to a GEPA example dict.
    
    GEPA expects examples to have at least:
    - 'id': unique identifier
    - Any other fields needed by the evaluator
    
    Args:
        task: Task dictionary from CSV
    
    Returns:
        GEPA-formatted example
    """
    return {
        "id": task.get("id", ""),
        "input": task.get("input", ""),
        "ground_truth": task.get("ground_truth", ""),
        "web": task.get("web", ""),
        "web_name": task.get("web_name", ""),
        "task_id": task.get("task_id", ""),
    }


def load_and_split_dataset(
    csv_path: str,
    web_names: Optional[Sequence[str]] = None,
    train_ratio: float = 0.7,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Complete pipeline: load CSV and split into train/val.
    
    Args:
        csv_path: Path to dataset_unified.csv
        train_ratio: Fraction for training
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_examples, val_examples) formatted for GEPA
    """
    tasks = load_tasks_from_csv(csv_path, web_names)
    train_tasks, val_tasks = create_train_val_split(tasks, train_ratio, random_seed)
    
    # Convert to GEPA format
    train_examples = [prepare_example_for_gepa(t) for t in train_tasks]
    val_examples = [prepare_example_for_gepa(t) for t in val_tasks]
    
    logging.info(f"Prepared {len(train_examples)} train and {len(val_examples)} val examples for GEPA")
    
    return train_examples, val_examples


# ============================= TEST CODE =============================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and split dataset_unified.csv for GEPA evaluation.")
    parser.add_argument("--web_names", type=str, default=None, help="Comma-separated names of the web to test, e.g. 'CUSTOM,GAIA'.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of tasks to use for training.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible splits.")
    return parser.parse_args()

if __name__ == "__main__":
    # Quick test
    csv_path = "dataset_unified.csv"
    args = parse_args()
    # Convert comma-separated string to list if provided
    if args.web_names:
        args.web_names = [w.strip() for w in args.web_names.split(",")]

    train, val = load_and_split_dataset(csv_path, web_names=args.web_names, train_ratio=args.train_ratio, random_seed=args.random_seed)
    print(f"Train: {len(train)}, Val: {len(val)}")
    if train:
        print(f"Sample train example: {train[0]}")

