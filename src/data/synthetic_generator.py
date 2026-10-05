import anthropic
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=True)


def load_vektorguard():
    """Load Vektor-Guard v2 tokenizer and model from HuggingFace. Returns (tokenizer, model) tuple ready for inference"""

    VEKTORGUARD_REPO = "theinferenceloop/vektor-guard-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(VEKTORGUARD_REPO, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(VEKTORGUARD_REPO)
    model.eval()
    model.to(device)

    return tokenizer, model

CATEGORIES = [
    "instruction_override",
    "indirect_injection",
    "jailbreak",
    "tool_call_hijacking",
    "clean"
]

def generate_examples_claude(category: str, n: int = 10) -> list[dict]:
    """Generate n synthetic prompt examples for the given attack category using Claude Sonnet 4.6. Returns list of dicts with text, label, and source fields."""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    all_examples = []
    batch_size = 25
    batches = (n + batch_size - 1) // batch_size

    for _ in range(batches):
        current_n = min(batch_size, n - len(all_examples))

        prompt = f"""Generate {current_n} examples of prompts that represent the attack category: {category}.

Each example should be a realistic prompt that a malicious user might send to an LLM-powered application.
Return ONLY a JSON array of strings, no explanation, no markdown, no preamble.
Example format: ["example 1", "example 2"]

Category definitions:
- instruction_override: User attempts to override, ignore, or replace the model's system prompt or instructions. Includes direct commands like "ignore previous instructions" and attempts to redefine the model's goals or behavior.
- indirect_injection: Malicious instructions embedded in external content like documents, web pages, or databases. Includes stored injection payloads waiting to be retrieved.
- jailbreak: Persona manipulation, roleplay exploits, DAN-style attacks that bypass safety guidelines through fictional framing.
- tool_call_hijacking: Manipulation of which tools are called or how tool parameters are invoked.
- clean: Completely legitimate prompts with no injection attempt.

Generate {current_n} examples for: {category}"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            batch = json.loads(message.content[0].text)
            all_examples.extend(batch)
        except json.JSONDecodeError:
            print(f"    Warning: failed to parse batch for {category}, skipping")
            continue

    return [{"text": ex, "label": category, "source": "synthetic_claude"} for ex in all_examples]

def generate_examples_openai(category: str, n: int = 10) -> list[dict]:
    """Generate n synthetic prompt examples for the given attack category using GPT-4.1. Returns list of dicts with text, label, and source fields."""

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    all_examples = []
    batch_size = 25
    batches = (n + batch_size - 1) // batch_size

    for _ in range(batches):
        current_n = min(batch_size, n - len(all_examples))

        prompt = f"""Generate {current_n} examples of prompts that represent the attack category: {category}.

Each example should be a realistic prompt that a malicious user might send to an LLM-powered application.
Return ONLY a JSON array of strings, no explanation, no markdown, no preamble.
Example format: ["example 1", "example 2"]

Category definitions:
- instruction_override: User attempts to override, ignore, or replace the model's system prompt or instructions. Includes direct commands like "ignore previous instructions" and attempts to redefine the model's goals or behavior.
- indirect_injection: Malicious instructions embedded in external content like documents, web pages, or databases. Includes stored injection payloads waiting to be retrieved.
- jailbreak: Persona manipulation, roleplay exploits, DAN-style attacks that bypass safety guidelines through fictional framing.
- tool_call_hijacking: Manipulation of which tools are called or how tool parameters are invoked.
- clean: Completely legitimate prompts with no injection attempt.

Generate {current_n} examples for: {category}"""

        response = client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            batch = json.loads(response.choices[0].message.content)
            all_examples.extend(batch)
        except json.JSONDecodeError:
            print(f"    Warning: failed to parse batch for {category}, skipping")
            continue

    return [{"text": ex, "label": category, "source": "synthetic_openai"} for ex in all_examples]

def validate_with_vektorguard(examples: list[dict], tokenizer, model, device, threshold: float = 0.85) -> tuple[list[dict], list[dict]]:
    """
    Layer 1 validation — run examples through Vektor-Guard v1. Returns (passed, flagged) where flagged examples,
    scored below threshold or were misclassified.
    """

    passed = []
    flagged = []

    for example in examples:
        inputs = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        probs = torch.softmax(output.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

        example["vg_confidence"] = round(confidence, 4)
        example["vg_label"] = "clean" if predicted_class == 0 else "injection"

        is_clean_category = example["label"] == "clean"
        is_correct = (predicted_class == 0) if is_clean_category else (predicted_class != 0)

        if is_correct and confidence >= threshold:
            passed.append(example)
        else:
            flagged.append(example)

    return passed, flagged

def verify_category(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Layer 2 validation — verify each example's attack category matches its label using Claude as a classifier. 
    Returns (passed, flagged) where flagged examples were assigned a different category.
    """

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    passed = []
    flagged = []

    for example in examples:
        prompt = f"""You are a prompt injection security classifier.

Given this prompt, identify which single attack category it belongs to.
Choose exactly one from this list:
- indirect_injection
- jailbreak
- instruction_override
- tool_call_hijacking
- clean

Respond with ONLY the category name, nothing else.

Prompt: {example["text"]}"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )

        if not message.content or not message.content[0].text.strip():
            example["verified_category"] = "unknown"
            flagged.append(example)
            continue

        verified_category = message.content[0].text.strip().lower().replace(" ", "_").replace("-", "_")
        # only keep the first word-group in case model adds explanation
        verified_category = verified_category.split("\n")[0].strip()
        example["verified_category"] = verified_category

        if verified_category == example["label"]:
            passed.append(example)
        else:
            flagged.append(example)

    return passed, flagged

def generate_dataset(n_per_class: int = 500, threshold: float = 0.85) -> dict:
    """
    Orchestrate full synthetic data generation pipeline across all attack categories. 
    Runs both validation layers, saves passed examples to data/synthetic/, flagged examples 
    to data/synthetic/flagged/, and returns a per-category quality report.
    """

    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir = Path("data/synthetic/flagged")
    flagged_dir.mkdir(parents=True, exist_ok=True)

    all_passed = []
    report = {}

    tokenizer, model = load_vektorguard()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for category in CATEGORIES:
        print(f"\n--- Generating {category} ---")

        threshold = 0.85

        n_each = n_per_class // 2
        claude_examples = generate_examples_claude(category, n=n_each)
        openai_examples = generate_examples_openai(category, n=n_each)
        raw = claude_examples + openai_examples

        print(f"    Generated: {len(raw)}")

        passed_11, flagged_11 = validate_with_vektorguard(raw, tokenizer, model, device, threshold=threshold)
        print(f"    Layer 1 passed: {len(passed_11)} / flagged: {len(flagged_11)}")

        passed_12, flagged_12 = verify_category(passed_11)
        print(f"    Layer 2 passed: {len(passed_12)} / flagged: {len(flagged_12)}")

        all_flagged = flagged_11 + flagged_12
        if all_flagged:
            flagged_path = flagged_dir / f"{category}_flagged.jsonl"
            with open(flagged_path, "w") as f:
                for ex in all_flagged:
                    f.write(json.dumps(ex) + "\n")

        all_passed.extend(passed_12)

        report[category] = {
            "generated": len(raw),
            "passed_11": len(passed_11),
            "passed_12": len(passed_12),
            "flagged_total": len(all_flagged),
        }

    output_path = output_dir / "synthetic_examples.jsonl"
    with open(output_path, "w") as f:
        for ex in all_passed:
            f.write(json.dumps(ex) + "\n")

    report_path = output_dir / "generation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n Done - {len(all_passed)} examples saved to {output_path}")
    print(f" Report saved to {report_path}")

    return report
    
if __name__ == "__main__":
    generate_dataset(n_per_class=500)