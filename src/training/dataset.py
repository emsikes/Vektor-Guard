import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.environ["HF_TOKEN"]


# Match id2label ordering in training_config.yaml
LABEL2ID = {
    "clean": 0,
    "instruction_override": 1,
    "indirect_injection": 2,
    "jailbreak": 3,
    "tool_call_hijacking": 4
}

def load_split(split_name: str, splits_dir: str = "data/splits") -> list[dict]:
    path = Path(splits_dir) / f"{split_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)     # Return a list of dicts matching unified normalize_example schema defined in src/data/loaders.py
    
def build_tokenizer(model_name: str = "answerdotai/ModernBERT-large"):
    # Instantiate once and pass around so we don't re-download on every call
    return AutoTokenizer.from_pretrained(model_name)

def resolve_label(label) -> int:
    """
    Convert string or integer label to integer class index.
    Phase 2 splits use binary integers (0/1).
    Phase 3 synthetic data uses string category names.
    """
    if isinstance(label, int):
        return label
    return LABEL2ID[label]

def compute_class_weights(examples: list[dict]) -> list[float]:
    """
    Compute inverse frequency class weights for weighted sampling.
    Rare classes get higher weights so the sampler draws them more frequently,
    correcting for class imbalance in the merged Phase 2 + synthetic dataset.
    Returns a weight value per example (not per class) for use with WeightedRandomSampler.
    """
    from collections import Counter

    labels = [resolve_label(ex["label"]) for ex in examples]
    class_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(LABEL2ID)

    # Inverse frequency: classes with fewer examples get higher weight
    class_weights = {
        cls: n_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }

    # Map per-example weights, each example gets its class weight
    sample_weights = [class_weights[label] for label in labels]

    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        label_name = [k for k, v in LABEL2ID.items() if v == cls][0]
        print(f"    {label_name}: {count} examples, weight={class_weights[cls]:.4f}")

    return sample_weights

def tokenize_split(examples: list[dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Extract texts and labels into separate lists for the tokenizer which expects a flat 
    list of strings, not a list dicts.
    """
    texts = [ex["text"] for ex in examples]
    labels = [resolve_label(ex["label"]) for ex in examples]

    tokenized = tokenizer(
        texts,
        truncation=True,            # Truncate to max_length to avoid OOM for long inputs
        padding="max_length",       # Pad all sequences to same length for consistent batch shapes
        max_length=max_length,      # 512 for baseline, will move to 2048 in a later phase
        return_tensors=None         # Return a plain Pythin list so Dataset.from_dict can consume them
    )

    tokenized["labels"] = labels    # Add labels after tokenization as a plain list
    return Dataset.from_dict(tokenized)
