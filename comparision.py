import json
import os
import sys

# Path to pre-saved prediction results
RESULTS_FILE = "eval_results.json"

def run_evaluation():
    print("\n" + "="*60)
    print("      OSTEOSCAN AI - MODEL EVALUATION & BENCHMARKING      ")
    print("="*60)

    if not os.path.exists(RESULTS_FILE):
        print(f"\n[!] Error: Benchmark file '{RESULTS_FILE}' not found in current directory.")
        print("Please ensure 'eval_results.json' is located inside the 'backend/' folder.")
        sys.exit(1)

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    cm = data["confusion_matrix"]
    benchmarks = data["clinical_benchmarks"]

    tp = cm["true_positives"]
    tn = cm["true_negatives"]
    fp = cm["false_positives"]
    fn = cm["false_negatives"]
    total = data["total_samples"]

    # Calculate metrics
    accuracy = ((tp + tn) / total) * 100
    error_rate = 100.0 - accuracy
    sensitivity = (tp / (tp + fn)) * 100  # Recall (Ability to catch true cases)
    specificity = (tn / (tn + fp)) * 100  # Ability to clear normal cases

    print(f"\n[+] Dataset Evaluated   : {data['dataset_name']}")
    print(f"[+] Holdout Test Scans : {total} images")
    print(f"[+] Model Backbone     : DenseNet-121 (Fine-tuned)")
    print("-" * 60)
    print("                   CONFUSION MATRIX                    ")
    print("-" * 60)
    print(f" True Positives  (TP) : {tp:4d}  | False Positives (FP) : {fp:4d}")
    print(f" False Negatives (FN) : {fn:4d}  | True Negatives  (TN) : {tn:4d}")
    print("-" * 60)
    print("               MODEL PERFORMANCE METRICS               ")
    print("-" * 60)
    print(f" Accuracy             : {accuracy:6.2f}%")
    print(f" Error Rate           : {error_rate:6.2f}%")
    print(f" Sensitivity (Recall) : {sensitivity:6.2f}%")
    print(f" Specificity          : {specificity:6.2f}%")
    print("-" * 60)
    print("        COMPARISON AGAINST RADIOLOGIST BENCHMARKS       ")
    print("-" * 60)
    print(f" Metric        | DenseNet-121 | Avg Radiologist | Delta")
    print(f" --------------|--------------|-----------------|-------")
    print(f" Error Rate    | {error_rate:10.2f}%   | {benchmarks['error_rate']:13.2f}%   | {benchmarks['error_rate'] - error_rate:+6.2f}%")
    print(f" Accuracy      | {accuracy:10.2f}%   | {benchmarks['accuracy']:13.2f}%   | {accuracy - benchmarks['accuracy']:+6.2f}%")
    print(f" Sensitivity   | {sensitivity:10.2f}%   | {benchmarks['sensitivity']:13.2f}%   | {sensitivity - benchmarks['sensitivity']:+6.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()