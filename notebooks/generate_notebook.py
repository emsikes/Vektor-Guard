import nbformat

nb = nbformat.v4.new_notebook()
cells = []

def code(source):
    cells.append(nbformat.v4.new_code_cell(source))

def markdown(source):
    cells.append(nbformat.v4.new_markdown_cell(source))

# ── Cell 1: Install dependencies FIRST (before any imports) ───────────────────
markdown("## 1. Install dependencies")
code("""\
!pip install "numpy>=2.0" transformers datasets accelerate>=1.1.0 evaluate huggingface_hub>=1.0.0 wandb python-dotenv -q
""")

# ── Cell 2: GPU check ─────────────────────────────────────────────────────────
markdown("## 2. Verify GPU")
code("""\
import torch
print(torch.cuda.get_device_name(0))
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
assert 'A100' in torch.cuda.get_device_name(0), 'WARNING: Not an A100 - change runtime to acquire for training'
""")

# ── Cell 3: Mount Google Drive ────────────────────────────────────────────────
markdown("## 3. Mount Google Drive (checkpoint persistence)")
code("""\
from google.colab import drive
drive.mount('/content/drive')

import os
CHECKPOINT_DIR = '/content/drive/MyDrive/vektor-guard/checkpoints-v2'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f'Checkpoint dir: {CHECKPOINT_DIR}')
""")

# ── Cell 4: Clone repo ────────────────────────────────────────────────────────
markdown("## 4. Clone repo")
code("""\
import os
if not os.path.exists('/content/vektor'):
    !git clone https://github.com/emsikes/vektor.git /content/vektor
%cd /content/vektor
""")

# ── Cell 5: Auth ──────────────────────────────────────────────────────────────
markdown("## 5. Authenticate HuggingFace and WandB")
code("""\
from huggingface_hub import login as hf_login
from google.colab import userdata
import wandb

# Secrets stored in Colab Secrets (left sidebar → key icon)
hf_login(token=userdata.get('HF_TOKEN'))
wandb.login(key=userdata.get('WANDB_API_KEY'))
""")

# ── Cell 6: Upload data splits and synthetic data ─────────────────────────────
markdown("## 6. Upload data splits and synthetic data")
code("""\
import os
os.makedirs('data/splits', exist_ok=True)
os.makedirs('data/synthetic', exist_ok=True)
from google.colab import files

print('Upload train.json, val.json, test.json and synthetic_examples.jsonl when prompted')
uploaded = files.upload()
for fname, data in uploaded.items():
    if fname == 'synthetic_examples.jsonl':
        path = f'data/synthetic/{fname}'
    else:
        path = f'data/splits/{fname}'
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Saved {path}')
""")

# ── Cell 7: Merge Phase 2 splits with Phase 3 synthetic data ──────────────────
markdown("## 7. Merge Phase 2 splits with Phase 3 synthetic data")
code("""\
import json, random

# Load Phase 2 binary training data
with open('data/splits/train.json') as f:
    phase2_train = json.load(f)

# Load Phase 2 val and test sets
with open('data/splits/val.json') as f:
    phase2_val = json.load(f)

with open('data/splits/test.json') as f:
    phase2_test = json.load(f)

# Load Phase 3 synthetic multi-class data
with open('data/synthetic/synthetic_examples.jsonl') as f:
    synthetic = [json.loads(line) for line in f]

# Shuffle synthetic data with fixed seed before splitting
random.seed(42)
random.shuffle(synthetic)

# Carve 15% of synthetic data for val, 5% for test, rest for train
n_synthetic = len(synthetic)
n_val = int(n_synthetic * 0.15)
n_test = int(n_synthetic * 0.05)

synthetic_val = synthetic[:n_val]
synthetic_test = synthetic[n_val:n_val + n_test]
synthetic_train = synthetic[n_val + n_test:]

# Map Phase 2 binary labels to Phase 3 taxonomy
PHASE2_LABEL_MAP = {0: "clean", 1: "instruction_override"}

def map_phase2(examples):
    return [{"text": ex["text"], "label": PHASE2_LABEL_MAP[ex["label"]], 
             "source": ex.get("source", "phase2")} 
            for ex in examples if isinstance(ex["label"], int)]

mapped_train = map_phase2(phase2_train)
mapped_val = map_phase2(phase2_val)
mapped_test = map_phase2(phase2_test)

# Combine and shuffle each split
combined_train = mapped_train + synthetic_train
combined_val = mapped_val + synthetic_val
combined_test = mapped_test + synthetic_test

random.shuffle(combined_train)
random.shuffle(combined_val)
random.shuffle(combined_test)

# Overwrite all three splits so build_trainer() picks them up
with open('data/splits/train.json', 'w') as f:
    json.dump(combined_train, f)

with open('data/splits/val.json', 'w') as f:
    json.dump(combined_val, f)

with open('data/splits/test.json', 'w') as f:
    json.dump(combined_test, f)

# Save phase3 reference copies
with open('data/splits/train_phase3.json', 'w') as f:
    json.dump(combined_train, f)

print(f"Phase 2 train: {len(mapped_train)} | Synthetic train: {len(synthetic_train)} | Combined: {len(combined_train)}")
print(f"Phase 2 val: {len(mapped_val)} | Synthetic val: {len(synthetic_val)} | Combined val: {len(combined_val)}")
print(f"Phase 2 test: {len(mapped_test)} | Synthetic test: {len(synthetic_test)} | Combined test: {len(combined_test)}")

# Show label distribution in val set
from collections import Counter
val_labels = Counter(ex["label"] for ex in combined_val)
print("\nVal set label distribution:")
for label, count in sorted(val_labels.items()):
    print(f"  {label}: {count}")
""")

# ── Cell 8: Train ─────────────────────────────────────────────────────────────
markdown("## 8. Train")
code("""\
import sys
sys.path.insert(0, '/content/vektor')

from src.training.trainer import build_trainer

# Point output_dir to Drive so checkpoints survive session expiry
trainer = build_trainer()
trainer.args.output_dir = CHECKPOINT_DIR

trainer.train()
""")

# ── Cell 9: Evaluate on test set ──────────────────────────────────────────────
markdown("## 9. Evaluate on test set")
code("""\
from src.training.dataset import load_split, build_tokenizer, tokenize_split
from src.training.metrics import compute_metrics, check_targets
import numpy as np

config_model = 'answerdotai/ModernBERT-large'
tokenizer = build_tokenizer(config_model)
test_dataset = tokenize_split(load_split('test'), tokenizer, max_length=2048)

# Run inference on test set using best checkpoint
predictions = trainer.predict(test_dataset)
metrics = compute_metrics((predictions.predictions, predictions.label_ids))

print(metrics)
check_targets(metrics)
""")

# ── Cell 10: Push to HuggingFace Hub ──────────────────────────────────────────
markdown("## 10. Push best model to HuggingFace Hub")
code("""\
trainer.model.push_to_hub('theinferenceloop/vektor-guard-v2')
tokenizer.push_to_hub('theinferenceloop/vektor-guard-v2')
print('Model pushed to https://huggingface.co/theinferenceloop/vektor-guard-v2')
""")

nb.cells = cells

with open('notebooks/multi_class_train_colab.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Notebook written to notebooks/multi_class_train_colab.ipynb')