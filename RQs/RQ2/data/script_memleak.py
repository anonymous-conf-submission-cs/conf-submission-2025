import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

PATCH_DIR = "commit_patches" # Directory containing commit patches

model_path = ""  
# Specify the model name here (Please see PerfAnnotator-mini directory on how to download the model)
# or refer to RQ1 on how to load other models
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # force slow tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)

def classify_commit(patch_content):
    inputs = tokenizer(patch_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

def analyze_patches(patch_dir):
    patches = [f for f in os.listdir(patch_dir) if f.endswith('.patch')]

    results = {}

    for patch_file in patches:
        patch_path = os.path.join(patch_dir, patch_file)

        with open(patch_path, "r", encoding="utf-8", errors="ignore") as file:
            patch_content = file.read()

        predicted_class = classify_commit(patch_content)
        results[patch_file] = predicted_class
        result_text = "Yes" if predicted_class == 1 else "No"
        print(f"{patch_file}: Memory Leak - {result_text}")

    leak_count = sum(1 for r in results.values() if r == 1)
    total_patches = len(patches)
    accuracy = (leak_count / total_patches) * 100 if total_patches else 0

    print(f"\nTotal patches: {total_patches}")
    print(f"Patches indicating Memory Leak: {leak_count}")
    print(f"Percentage of memory leaks: {accuracy:.2f}%")

if __name__ == "__main__":
    analyze_patches(PATCH_DIR)