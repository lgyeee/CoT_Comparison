# utils.py
import re
import os
from openrouter import OpenRouter

MODEL_MAP = {
    #"qwen3-8b": "qwen/qwen3-8b", 
    #"mistral-7b-instruct": "mistralai/ministral-8b",
    # OpenRouter 上可用的 gpt-oss 系列
    "gpt-oss-20b":  "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    #"deepseek-r1": "deepseek/deepseek-r1-0528",
    "qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    "deepseek-r1-distill-qwen-32b": "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-qwen-7b": "deepseek/deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-1.5b": "deepseek/deepseek-r1-distill-qwen-1.5b",
    "qwen3-vl-30b-a3b-thinking": "qwen/qwen3-vl-30b-a3b-thinking",
    "deepseek-r1-distill-llama-8b": "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "qwen3-8b": "qwen/qwen3-8b",
    "qwen3-32b": "qwen/qwen3-32b"
}

# Store model configs from OpenRouter API
_MODEL_CONFIGS = {}

def load_models_from_map(api_key=None):
    """Load model information from OpenRouter API for all models in MODEL_MAP."""
    api_key = api_key or os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    
    with OpenRouter(api_key=api_key) as open_router:
        res = open_router.models.list()
        models_in_map = set(MODEL_MAP.values())
        
        for model_obj in res.data:
            model_id = model_obj.id if hasattr(model_obj, 'id') else (model_obj.get("id") if isinstance(model_obj, dict) else None)
            if model_id and model_id in models_in_map:
                _MODEL_CONFIGS[model_id] = model_obj

def get_model_config(model_id):
    """Get model config from stored configs."""
    return _MODEL_CONFIGS.get(model_id)

DATASET_MAP = {
    "thinkbench": { 
        "args": ("zhiyuan218/Think-Bench", "train"),
        "id_key": "index",
        "question_key": "question",
        "cot_key": "key_annotation_steps",
        "answer_key": "answer",
        "category_key": "category" 
    },
    "gpqa":{
        "args": ("fingertap/GPQA-Diamond", "test"),
        "question_key": "question",
        "answer_key": "answer"
    },
    "AIME2025": {
        "args": ("yentinglin/aime_2025", "train"),
        "id_key": "id",
        "question_key": "problem",
        "answer_key": "answer"
    },
    "MATH500": {
        "args": ("HuggingFaceH4/MATH-500", "test"),
        "id_key": "unique_id",
        "question_key": "problem",
        "answer_key": "answer",
        "category_key": "subject",
        "level_key": "level"
    }
}

def extract_gold_cot(cot_field):
    if isinstance(cot_field, dict):
        solutions = cot_field.get("solution1").get("logical_conclusion", [])
        steps = " ".join(solutions[:-1])  # exclude last answer line
        return steps
    return cot_field or ""

import math
from math_verify import parse, verify as latex_verify, LatexExtractionConfig

# ---- 穩健版 boxed 擷取：支援 \boxed{...} / \\boxed{...} 並處理巢狀大括號 ----
def _extract_boxed_segments(text: str):
    """
    回傳所有 boxed 片段（只取大括號內容），支援 \boxed{...} 或 \\boxed{...}，可處理巢狀花括號。
    例如: r"... \\boxed{(0, 9) \\cup (9, 36)} ..." -> ["(0, 9) \\cup (9, 36)"]
    """
    if not text:
        return []
    # 找到 "boxed{" 的起點（允許前面有 1 或 2 個 backslash）
    # 例如 \boxed{ 或 \\boxed{
    pattern = re.compile(r"(?:\\{1,2})boxed\{")
    matches = []
    i = 0
    while True:
        m = pattern.search(text, i)
        if not m:
            break
        # m.end() 在 '{' 之後的位置
        start = m.end()  # 指向第一個 '{' 之後
        # 從 start-1 (即 '{' 本身) 開始做括號配對
        l = start - 1
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            # text[l] == '{', text[j-1] == '}', 取中間內容
            matches.append(text[l+1:j-1])
            i = j
        else:
            # 沒配對成功就退出
            break
    return matches

# ---- 取「模型輸出中的答案」：盡量回傳 latex 片段（最後一個 boxed），否則退化為最後一個數字 ----
def extract_boxed_answer(pred_text: str):
    """
    盡量抽出最後一個 \\boxed{...} 的內容（原樣 LaTeX 字串）。
    若沒有 boxed，退而求其次：回傳文本中最後一個數字（字串）。
    若仍沒有，回傳 None。
    """
    try:
        boxed = _extract_boxed_segments(pred_text)
        if boxed:
            # 回傳最後一個盒內原文 (LaTeX 片段)
            return [boxed[-1].strip()]  # 與你現有介面相容：回傳 list[str]
        # 沒有 boxed -> 抓最後一個數字
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", pred_text or "")
        if nums:
            return [nums[-1].lstrip('+')]
        return None
    except Exception as e:
        print("Error in extract_boxed_answer:", e)
        return None

# ---- gold 答案維持原樣（若你資料集 gold 已是 LaTeX，可直接用）----
def extract_gold_answer(gold_text: str):
    """
    直接回傳整段 gold（作為待比對的 latex 片段）。
    若需要也可在這裡加清理；目前保留原樣。
    """
    return [gold_text]

# ---- LaTeX-aware 驗證：盡量用 math_verify.parse/verify，比不到再退化為字串比較 ----
def verify_answer(pred_text: str, gold_text: str) -> bool:
    """
    允許 pred_text 是完整模型輸出；內部會先抽出 (最後一個) \\boxed{...}。
    優先用 LaTeX verify（math_verify）做等價判斷；失敗時降級為簡化字串比較。
    """
    if pred_text is None or gold_text is None:
        return False

    # 先抽出模型的「候選答案」（LaTeX）
    pred_tokens = extract_boxed_answer(pred_text)  # list[str] or None
    if not pred_tokens:
        return False
    pred = pred_tokens[-1].strip()
    ref  = gold_text.strip()

    # 特例 1：基數表記 (e.g. (1011)_{2}) 與對方純字串相等
    BASE_N_RE = re.compile(r"^\(?([0-9A-Za-z]+)\)?_\{(\d+)\}$")
    m = BASE_N_RE.match(pred)
    if m and m.group(1) == ref:
        return True
    m = BASE_N_RE.match(ref)
    if m and m.group(1) == pred:
        return True

    # 特例 2：巨大指數防呆（避免爆計算）→ 直接比較去空白字串
    EXP_RE = re.compile(r"\^\{(\d+)\}")
    MAX_SAFE_EXP = 10_000
    try:
        exps = [int(e) for e in EXP_RE.findall(pred)]
        if exps and max(exps) > MAX_SAFE_EXP:
            return pred.replace(" ", "") == ref.replace(" ", "")
    except Exception:
        pass

    # 嘗試用 LaTeX 結構驗證
    wrap = lambda s: f"\\({s}\\)"
    cfg = LatexExtractionConfig()
    try:
        g_node = parse(wrap(ref), extraction_config=[cfg])
        p_node = parse(wrap(pred), extraction_config=[cfg])
        ok = latex_verify(g_node, p_node, float_rounding=2)
        if ok:
            return True
    except Exception as e:
        # 解析/比對出錯就降級
        print(f"[verify_answer] LaTeX verify failed: {e}")

    # 最後降級：寬鬆字面比較（移除空白、把連續空白壓成無）
    def _norm(s: str):
        s = s.strip()
        s = re.sub(r"\s+", "", s)
        return s
    return _norm(pred) == _norm(ref)
