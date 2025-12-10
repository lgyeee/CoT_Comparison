# CoT Comparison

Chain-of-Thought (CoT) comparison framework for evaluating reasoning models.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Set your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
# 或
export OPENROUTER_KEY="your-api-key-here"
```

Or add it to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
echo 'export OPENROUTER_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Basic Usage

```bash
python reasoning_loop_resp.py \
    --model qwen3-8b \
    --mode general_cot \
    --dataset AIME2025 \
    --reasoning_effort high \
    --include_reasoning \
    --limit 30
```

### Using Bash Script

```bash
export OPENROUTER_KEY="your-api-key-here"
export MODEL="qwen3-8b"
bash bash_script.sh
```

## Arguments

- `--model`: Model identifier (e.g., `qwen3-8b`, `gpt-oss-120b`)
- `--mode`: CoT mode (`general_cot` or `gold_cot`)
- `--dataset`: Dataset name (e.g., `AIME2025`)
- `--category`: Optional category filter (`math`, `physics`, `chemistry`)
- `--reasoning_effort`: Reasoning effort level (`low`, `medium`, `high`)
- `--include_reasoning`: Include reasoning in response
- `--limit`: Maximum number of samples to process

## Output

Results are saved in `outputs/{model_name}/`:
- `{model}_{mode}_{category}_{timestamp}.json`: Main results
- `{model}_{mode}_{category}_{timestamp}.jsonl`: Line-delimited results
- `{model}_{mode}_{category}_{timestamp}_payloads.jsonl`: Full request/response payloads

Logs are saved in `logs/` directory.

## Security

⚠️ **Important**: Never commit API keys to the repository. Always use environment variables.

