import os
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

JSONL_FILE = "ground_truth_dataset.jsonl"
model_name = "" # Specify the model name here
# please see PerfAnnotator-mini direcotry on how to download the model
#model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
    attn_implementation="eager"  # ensures compatibility and resolves warnings
)

def classify_patch_with_phi3(patch_content, max_new_tokens=3):
    prompt = (
        "<|system|>"
        "You are an expert software developer. Given a patch/diff, determine "
        "if it fixes or relates to a performance or resource improvement. "
        "Only respond with 'Non-Perf' or 'Perf', nothing else."
        "</s><|user|>" + patch_content + "</s><|assistant|>"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False  # fix caching error
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
    )

    return 1 if response.strip() == "Perf" else 0

def analyze_jsonl_file(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    total = len(lines)
    y_true = []
    y_pred = []

    for idx, line in enumerate(lines):
        data = json.loads(line)
        commit_message = data["commit_message"]
        code_diff = data["code_diff"]
        ground_truth = data["target"]

        combined_content = commit_message + " " + code_diff
        prediction = classify_patch_with_phi3(combined_content)

        result_text = "Perf" if prediction == 1 else "Non-perf"
        correctness = "Correct" if prediction == ground_truth else "Incorrect"

        print(f"Sample {idx+1}: Prediction: {result_text}, Ground Truth: {'Perf' if ground_truth else 'Non-perf'} - {correctness}")

        y_true.append(ground_truth)
        y_pred.append(prediction)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = (sum(p == t for p, t in zip(y_pred, y_true)) / total) * 100 if total else 0

    print("\n--- Analysis Summary ---")
    print(f"Total samples: {total}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    analyze_jsonl_file(JSONL_FILE)