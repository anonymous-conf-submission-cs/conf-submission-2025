# PerfAnnotator-mini: Usage Instructions

The model can be downloaded from: [Link](https://huggingface.co/annon-123/PerfAnnotator-mini)

Follow the steps below to load the model and run inference locally or directly from Hugging Face.

---

## 1. Requirements

- Python 3.8+  
- [PyTorch](https://pytorch.org/) 1.8+  
- [Transformers](https://github.com/huggingface/transformers) 4.20+

Install dependencies via pip:

```bash
pip install torch transformers
```

---

## 2. Model directory layout

```
./
├── config.json
├── model.safetensors      
├── tokenizer.json
├── merges.txt
├── special_tokens_map.json
├── tokenizer_config.json
├── vocab.json          
```

---

## 3. Loading the Model

### A. From Hugging Face Hub

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo_id = "annon-123/PerfAnnotator-mini"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model     = AutoModelForSequenceClassification.from_pretrained(repo_id)
model.eval()
```

### B. Locally from Disk

If you have cloned or downloaded the repo:

```bash
git clone https://huggingface.co/annon-123/PerfAnnotator-mini
cd PerfAnnotator-mini
```

Or if you have a `.zip` archive:

```bash
unzip PerfAnnotator-mini.zip -d PerfAnnotator-mini
cd PerfAnnotator-mini
```

Then load:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

local_path = "./PerfAnnotator-mini"  # or your folder

tokenizer = AutoTokenizer.from_pretrained(local_path)
model     = AutoModelForSequenceClassification.from_pretrained(local_path)
model.eval()
```

---

## 4. Running Inference

Use the loaded model to classify a commit message + diff text:

```python
# Example input: commit message + diff context
text = (
    "Fix unnecessary memcpy → mempcpy\n"
    "@@ -10,7 +10,7 @@ ..."
)

# Tokenize and prepare inputs
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# Perform a forward pass
with torch.no_grad():
    logits = model(**inputs).logits

prediction_id = logits.argmax(dim=-1).item()
print(f"Predicted label ID: {prediction_id}")
```

> **Label Mapping:**  
> `0 = non-performance-improving`  
> `1 = performance-improving`

---
