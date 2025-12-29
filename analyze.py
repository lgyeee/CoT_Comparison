import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Set the path to your jsonl or json file
PATH = "outputs/qwen3-32b/qwen3-32b_general_cot_all_20251210-180828.jsonl"
model="qwen3-32b"
dataset="AIME2025"

# Load data from file
data = []
if PATH.endswith('.jsonl'):
    with open(PATH, 'r', encoding='utf-8') as f:
        for part in f.read().split('\n\n'):
            part = part.strip()
            if part:
                try:
                    record = json.loads(part)
                    if isinstance(record, dict):
                        data.append(record)
                except:
                    pass
elif PATH.endswith('.json'):
    with open(PATH, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
        if isinstance(loaded_data, list):
            data = [r for r in loaded_data if isinstance(r, dict)]
        elif isinstance(loaded_data, dict):
            data = [loaded_data]

# ========================================================================
# 1. Calculate accuracy 
# ========================================================================
# 1.1 Filter records where full_response is not empty
valid_records = []
for record in data:
    # Check if record is a dictionary
    if not isinstance(record, dict):
        continue
    full_response = record.get('full_response', '')
    if full_response and full_response.strip():
        valid_records.append(record)

# 1.2 Count correct answers
correct_count = 0
total_count = len(valid_records)

for record in valid_records:
    if record.get('correct', False):
        correct_count += 1

# 1.3 Calculate accuracy
if total_count > 0:
    accuracy = correct_count / total_count
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%)")


# ========================================================================
# 2. Statistics for reasoning length distribution
# ========================================================================
# 2.1 Response Tokens Distribution for records with reasoning
reasoning_records = []
for record in data:
    if record.get('reasoning', ''):
        reasoning_records.append(record)

# 2.2 Separate records into 4 groups
no_full_response_correct = []    # 浅红
no_full_response_incorrect = []   # 深红
has_full_response_correct = []    # 浅蓝
has_full_response_incorrect = []  # 深蓝

for record in reasoning_records:
    response_tokens = record.get('response_tokens', 0)
    full_response = record.get('full_response', '')
    is_correct = record.get('correct', False)
    
    has_full = full_response and full_response.strip()
    if has_full:
        if is_correct:
            has_full_response_correct.append(response_tokens)
        else:
            has_full_response_incorrect.append(response_tokens)
    else:
        if is_correct:
            no_full_response_correct.append(response_tokens)
        else:
            no_full_response_incorrect.append(response_tokens)

# 2.3 Create histogram with four colors
all_groups = [no_full_response_correct, no_full_response_incorrect, 
              has_full_response_correct, has_full_response_incorrect]
if any(all_groups):
    plt.figure(figsize=(10, 6))
    bins = list(range(0, 24001, 1000))
    
    # Plot stacked histogram - all groups stacked vertically
    data_to_plot = []
    colors_to_plot = []
    labels_to_plot = []
    
    if no_full_response_incorrect:
        data_to_plot.append(no_full_response_incorrect)
        colors_to_plot.append('darkred')
        labels_to_plot.append('No full_response (Incorrect)')
    if no_full_response_correct:
        data_to_plot.append(no_full_response_correct)
        colors_to_plot.append('lightcoral')
        labels_to_plot.append('No full_response (Correct)')
    if has_full_response_incorrect:
        data_to_plot.append(has_full_response_incorrect)
        colors_to_plot.append('darkblue')
        labels_to_plot.append('Has full_response (Incorrect)')
    if has_full_response_correct:
        data_to_plot.append(has_full_response_correct)
        colors_to_plot.append('lightblue')
        labels_to_plot.append('Has full_response (Correct)')
    
    if data_to_plot:
        plt.hist(data_to_plot, bins=bins, range=(0, 24000), stacked=True, 
                 label=labels_to_plot, color=colors_to_plot, edgecolor='black', alpha=0.8)
    
    plt.xlabel('Response Tokens', fontsize=12)
    plt.ylabel('Frequency (count)', fontsize=12)
    plt.title(f'Response Tokens Distribution\nModel: {model} | Dataset: {dataset}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 2.4 Save the plot
    output_path = PATH.replace('.jsonl', '').replace('.json', '') + '_reasoning_tokens_dist.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # 2.5 Print statistics
    all_tokens = sum(all_groups, [])
    print(f"\nReasoning Tokens Statistics:")
    print(f"  Total records with reasoning: {len(reasoning_records)}")
    print(f"  No full_response - Correct: {len(no_full_response_correct)}, Incorrect: {len(no_full_response_incorrect)}")
    print(f"  Has full_response - Correct: {len(has_full_response_correct)}, Incorrect: {len(has_full_response_incorrect)}")
    if all_tokens:
        print(f"  Min tokens: {min(all_tokens)}")
        print(f"  Max tokens: {max(all_tokens)}")
        print(f"  Mean tokens: {sum(all_tokens)/len(all_tokens):.2f}")

