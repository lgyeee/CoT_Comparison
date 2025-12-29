import sys
import os
import json
import time
import logging
import datetime
from pathlib import Path
from collections import Counter

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from utils import (
    MODEL_MAP,
    DATASET_MAP,
    extract_gold_cot,
    extract_boxed_answer,
    verify_answer,
    load_models_from_map,
    get_model_config
)

MAX_TOKENS_BUFFER = 100  # Buffer to leave some tokens for safety
EMPTY_RESPONSE_RETRIES = 0  # Retry when model returns empty content
RATE_LIMIT_DELAY = 0.6  # Delay between retries

def _estimate_messages_tokens(messages, model_id="gpt-4"):
    """Estimate token count for messages using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        return None
    
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        role_tokens = len(encoding.encode(role))
        content_tokens = len(encoding.encode(content))
        total_tokens += role_tokens + content_tokens + 4
    
    return total_tokens + 10

# ---------- logging: write to file and print to console ----------
os.makedirs("logs", exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
_LOG_PATH = f"logs/run-{_ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

_openrouter_client = None


def _get_openrouter_client():
    """Lazy init OpenRouter client via OpenAI SDK."""
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY (or OPENROUTER_KEY) not set")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    return _openrouter_client

def make_messages(mode, question, gold_cot=None):
    if mode == "general_cot":
        return [
            {"role": "user", "content": "Please think step by step.\n\n"
             "After reasoning, output the final answer in the format: \\boxed{answer}."
             "Make sure the boxed answer is the very last part of your response.\n\n"
             f"Question: {question}\n\n"}
        ]
    elif mode == "gold_cot":
        return [
            {
                "role": "system",
                "content": (
                    "You already generate a correct and complete worked solution."
                    "You must output ONLY the final numeric or symbolic answer, using the steps you’ve generate."
                    "and the OUTPUT MUST be written inside \\boxed{...}. "
                )
            },
            {"role": "user", "content": "Problem:\n" + question},
            {"role": "assistant", "content": gold_cot},
            {
                "role": "user",
                "content": (
                    "Now output ONLY the final answer from the solution, "
                    "in the format \\boxed{...}. Nothing else."
                )
            }
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def call_model(
    model_id,
    messages,
    temperature=0.0,
    timeout=60,
    max_tokens=None,
    reasoning=None,
    include_reasoning=None
):
    client = _get_openrouter_client()

    extra_params = {
        "provider": {
            "order": ["DeepInfra", "Fireworks"],
            "allow_fallbacks": True
        }
    }
    if include_reasoning is not None:
        extra_params["include_reasoning"] = include_reasoning
    if reasoning is not None:
        extra_params["reasoning"] = reasoning

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_params
        )
        return completion.model_dump()
    except Exception as e:
        logging.error(f"Error calling model: {e}")
        raise

def _extract_text_and_tokens(resp_json):
    """Extract text, tokens, reasoning_text and reasoning_details from response."""
    if not isinstance(resp_json, dict) or not resp_json.get("choices"):
        usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
        return "", usage.get("total_tokens"), None, None, usage.get("prompt_tokens")
    
    msg = resp_json["choices"][0].get("message", {}) or {}
    text = (msg.get("content") or "").strip()
    reasoning_details = msg.get("reasoning_details")
    
    # Extract raw reasoning text from reasoning_details (only reasoning.text type)
    reasoning_text = None
    if reasoning_details:
        for detail in reasoning_details:
            if isinstance(detail, dict) and detail.get("type") == "reasoning.text":
                reasoning_text = detail.get("text")
                break
    
    usage = resp_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    return text, usage.get("total_tokens"), reasoning_text, reasoning_details, prompt_tokens

def _format_json(obj):
    """Format JSON output, similar to jq's formatting."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


def _write_jsonl_line(file_path, record):
    """Write a single record to JSONL file (append mode)."""
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(_format_json(record) + "\n\n")
    except Exception as e:
        logging.error(f"Failed to write to {file_path}: {e}")

def run_mode(
    dataset_key,
    mode,
    category=None,          # None: run all dataset, "math"/"physics"/"chemistry": specific category
    model=None,
    reasoning_cfg=None,
    include_reasoning=None,
    limit=None
):
    # ========================================================================
    # 1. Load Configuration & Model 
    # ========================================================================
    # 1.1 Load dataset configuration
    ds_cfg = DATASET_MAP[dataset_key]
    
    # 1.2 Load dataset
    ds = load_dataset(ds_cfg["args"][0], split=ds_cfg["args"][1])
    
    # 1.3 Filter dataset by category if specified
    if category is not None and "category" in ds.column_names:
        ds = ds.filter(lambda ex: ex.get("category", "") == category)
    
    # 1.3.1 Filter MATH500 dataset by level 1-3 if dataset is MATH500
    if dataset_key == "MATH500":
        level_key = ds_cfg.get("level_key")
        if level_key and level_key in ds.column_names:
            ds = ds.filter(lambda ex: 1 <= ex.get(level_key, 0) <= 3)
            logging.info(f"Filtering MATH500 dataset by level: 1 <= {level_key} <= 3")
            logging.info(f"Filtered dataset size: {len(ds)}")
    
    # 1.4 Get model configuration
    model_key, model_id = model, MODEL_MAP[model]
    
    # 1.4.1 Load model configs from OpenRouter API
    load_models_from_map()
    
    # 1.5 Setup output directories
    # Add dataset subdirectory: outputs/{model}/{dataset}/
    out_dir = Path("outputs") / model_key / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_category = category if (category and "category" in ds.column_names) else "all"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main_json_path = out_dir / f"{model_key}_{mode}_{file_category}_{timestamp}.json"
    main_jsonl_path = out_dir / f"{model_key}_{mode}_{file_category}_{timestamp}.jsonl"
    payload_path = out_dir / f"{model_key}_{mode}_{file_category}_{timestamp}_payloads.jsonl"
    
    question_key = ds_cfg.get("question_key", "question")
    answer_key = ds_cfg.get("answer_key", "answer")
    cot_key = ds_cfg.get("cot_key")
    id_key = ds_cfg.get("id_key", "id")
    level_key = ds_cfg.get("level_key")  # Get level_key for MATH500
    default_category = None if "category" in ds.column_names else "all"
    
    results = []
    processed = 0
    
    logging.info(f"===== Running {model_key} on {dataset_key} with mode {mode}, category={category} =====")
    logging.info(f"Dataset columns: {ds.column_names}")
    logging.info(f"Using id_key: {id_key}")
    
    if "category" in ds.column_names:
        logging.info(f"Category counts: {Counter(ds['category'])}")
    
    # ========================================================================
    # 2. Process Questions Block
    # ========================================================================
    for sample in tqdm(ds, desc=f"Processing: {model_key}", dynamic_ncols=True):
        gold_cot = extract_gold_cot(sample.get(cot_key)) if (mode == "gold_cot" and cot_key) else None
        question = (sample.get(question_key) or "").strip()
        answer_gold = (sample.get(answer_key) or "").strip()
        id = sample.get(id_key)
        messages = make_messages(mode, question, gold_cot)
        
        # Calculate max_tokens
        model_config = get_model_config(model_id)
        context_length = model_config.context_length if model_config and hasattr(model_config, 'context_length') else (model_config.get("context_length") if model_config and isinstance(model_config, dict) else None)
        if context_length:
            prompt_tokens_estimated = _estimate_messages_tokens(messages, model_id)
            max_tokens = max(1, context_length - prompt_tokens_estimated - MAX_TOKENS_BUFFER) if prompt_tokens_estimated else int(context_length * 0.9)
        else:
            max_tokens = 20000
            prompt_tokens_estimated = None
        max_tokens = int(max_tokens)
        
        # Call model API with retry for empty responses
        resp_json = None
        full_response = ""
        last_err = None
        
        for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
            try:
                resp_json = call_model(
                    model_id=model_id,
                    messages=messages,
                    timeout=180,
                    max_tokens=max_tokens,
                    reasoning=reasoning_cfg,
                    include_reasoning=include_reasoning
                )
                full_response, _, _, _, _ = _extract_text_and_tokens(resp_json)
                if full_response and full_response.strip():
                    break
                last_err = "empty_response"
            except Exception as e:
                last_err = str(e)
                resp_json = None
            
            if attempt < EMPTY_RESPONSE_RETRIES:
                time.sleep(RATE_LIMIT_DELAY)
        
        if not full_response:
            # Extract reasoning_text from resp_json if available
            reasoning_text = None
            predicted = None
            correct = False
            if resp_json:
                _, _, reasoning_text, _, _ = _extract_text_and_tokens(resp_json)
                if reasoning_text and reasoning_text.strip():
                    predicted = extract_boxed_answer(reasoning_text) or [reasoning_text.strip()]
                    correct = verify_answer(reasoning_text, answer_gold)
            
            # Save payload if resp_json exists
            if resp_json:
                payload_record = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "question": question,
                    "id": id,
                    "request": {
                        "model": model_id,
                        "messages": messages,
                        "reasoning": reasoning_cfg,
                        "include_reasoning": include_reasoning,
                        "max_tokens": max_tokens,
                        "prompt_tokens_estimated": prompt_tokens_estimated
                    },
                    "response": resp_json
                }
                _write_jsonl_line(payload_path, payload_record)
            
            record = {
                "question": question,
                "full_response": None,
                "predicted": predicted,
                "answer_gold": answer_gold,
                "correct": correct,
                "tokens": None,
                "id": id,
                "error": last_err if last_err else "unknown_error",
                "category": sample.get("category", default_category),
                "gold_cot": gold_cot,
                "max_tokens": max_tokens,
                "prompt_tokens": prompt_tokens_estimated,
                "reasoning": reasoning_text,
                "response": resp_json
            }
            # Add level field if level_key exists (e.g., for MATH500)
            if level_key and level_key in sample:
                record["level"] = sample.get(level_key)
            results.append(record)
            _write_jsonl_line(main_jsonl_path, record)
            processed += 1
            if limit and processed >= limit:
                break
            continue
        
        # ========================================================================
        # 3. Log File Recording Block
        # ========================================================================
        # 3.1 Save full payload (request + response) to payload file
        try:
            payload_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "question": question,
                "id": id,
                "request": {
                    "model": model_id,
                    "messages": messages,
                    "reasoning": reasoning_cfg,
                    "include_reasoning": include_reasoning,
                    "max_tokens": max_tokens,
                    "prompt_tokens_estimated": prompt_tokens_estimated
                },
                "response": resp_json
            }
            _write_jsonl_line(payload_path, payload_record)
        except Exception as e:
            logging.error(f"Error saving payload for sample {id}: {e}")
        
        # ========================================================================
        # 4. Result Recording Block
        # ========================================================================
        try:
            # 4.1 Extract text and tokens from response
            full_response, response_tokens, reasoning_text, _, prompt_tokens = _extract_text_and_tokens(resp_json)
            predicted = extract_boxed_answer(full_response) or [full_response.strip()]
            correct = verify_answer(full_response, answer_gold)
            
            # 4.2 Create result record
            record = {
                "id": id,
                "question": question,
                "full_response": full_response,
                "predicted": predicted,
                "answer_gold": answer_gold,
                "correct": correct,
                "response_tokens": response_tokens,
                "reasoning": reasoning_text,
                "category": sample.get("category", default_category),
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "gold_cot": gold_cot if mode == "gold_cot" else None
            }
            # Add level field if level_key exists (e.g., for MATH500)
            if level_key and level_key in sample:
                record["level"] = sample.get(level_key)
            results.append(record)
            _write_jsonl_line(main_jsonl_path, record)
        except Exception as e:
            logging.error(f"Error processing sample {id}: {e}")
            # Create error record
            record = {
                "id": id,
                "question": question,
                "full_response": None,
                "predicted": None,
                "answer_gold": answer_gold,
                "correct": False,
                "response_tokens": None,
                "reasoning": None,
                "category": sample.get("category", default_category),
                "prompt_tokens": None,
                "max_tokens": max_tokens,
                "gold_cot": gold_cot if mode == "gold_cot" else None,
                "error": str(e)
            }
            # Add level field if level_key exists (e.g., for MATH500)
            if level_key and level_key in sample:
                record["level"] = sample.get(level_key)
            results.append(record)
            _write_jsonl_line(main_jsonl_path, record)
        
        processed += 1
        if limit and processed >= limit:
            break
    
    # ========================================================================
    # 5. Final Result Saving Block
    # ========================================================================
    logging.info(f"Processing completed. Total results: {len(results)}")
    logging.info(f"Output directory: {out_dir}")

    if results:
        try:
            logging.info(f"Saving JSON file to {main_json_path}")
            with open(main_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=False)
            logging.info(f"Successfully saved {len(results)} records to {main_json_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON file: {e}")
    else:
        logging.warning("No results to save. Results list is empty.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["general_cot","gold_cot"], required=True)
    parser.add_argument("--dataset", choices=list(DATASET_MAP.keys()), required=True)
    parser.add_argument("--category", type=str, default=None,
                        help="Category to filter: 'math', 'physics', 'chemistry'. Default: None (all categories)")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), required=True)
    parser.add_argument("--reasoning_effort", type=str, required=True)
    parser.add_argument("--include_reasoning", action="store_true")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Process at most N samples (default=1)")
    args = parser.parse_args()
    
    reasoning_cfg = {"effort": args.reasoning_effort}
    
    run_mode(
        dataset_key=args.dataset,
        category=args.category,
        mode=args.mode,
        model=args.model,
        reasoning_cfg=reasoning_cfg,
        include_reasoning=args.include_reasoning,
        limit=args.limit
    )
