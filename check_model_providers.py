from openai import OpenAI
import os
import json
import requests

# 初始化 client
# 從環境變數讀取 API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("請設定環境變數 OPENROUTER_API_KEY 或 OPENROUTER_KEY")
model_id = "qwen/qwen3-8b"

def get_raw_provider_info(target_model):
    print(f"正在透過 Raw API 查詢 {target_model} ...\n")
    
    # 方法1: 查詢模型列表
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        for model in data:
            if model["id"] == target_model:
                print(f"=== 找到模型: {model['name']} ===\n")
                
                # 打印所有可用的 keys
                print("=== 模型所有欄位 ===")
                print("Keys:", list(model.keys()))
                
                # 檢查所有可能包含 provider 信息的字段
                print("\n=== 檢查 Provider 相關欄位 ===")
                
                # 1. top_provider
                if "top_provider" in model:
                    print("\n[top_provider]")
                    print(json.dumps(model.get("top_provider"), indent=2, ensure_ascii=False))
                
                # 2. endpoints
                if "endpoints" in model:
                    print("\n[endpoints] - 可能包含多個 provider")
                    print(json.dumps(model.get("endpoints"), indent=2, ensure_ascii=False))
                
                # 3. providers (如果有)
                if "providers" in model:
                    print("\n[providers]")
                    print(json.dumps(model.get("providers"), indent=2, ensure_ascii=False))
                
                # 4. 檢查所有嵌套的字典，看是否有 provider 相關信息
                print("\n=== 深度搜尋 Provider 相關資訊 ===")
                def find_provider_info(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key
                            if 'provider' in key.lower() or 'endpoint' in key.lower():
                                print(f"\n找到: {current_path}")
                                print(json.dumps(value, indent=2, ensure_ascii=False))
                            elif isinstance(value, (dict, list)):
                                find_provider_info(value, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            find_provider_info(item, f"{path}[{i}]")
                
                find_provider_info(model)
                
                # 打印完整模型資訊
                print("\n=== 完整模型資訊 (JSON) ===")
                print(json.dumps(model, indent=2, ensure_ascii=False, default=str))
                
                return model
        print(f"未找到 {target_model}")
    else:
        print(f"API 請求失敗: {response.status_code}")
        print(response.text)
    
    # 方法2: 嘗試查詢特定模型的詳細信息（如果有其他端點）
    print("\n\n=== 關於 Provider 資訊 ===")
    print("注意: OpenRouter API 不會直接暴露所有 provider 名稱")
    print("因為 OpenRouter 是自動路由服務，會自動選擇最佳 provider")
    print("\n如果您需要指定 provider，可以在 API 調用時使用 extra_body 參數：")
    print('extra_body={"provider": {"order": ["DeepInfra"], "allow_fallbacks": True}}')
    print("\n常見的 provider 包括：")
    print("  - DeepInfra")
    print("  - Together AI")
    print("  - Anyscale")
    print("  - Groq")
    print("  - 等等...")
    print("\n建議：查看 OpenRouter 官網 (https://openrouter.ai/models) 獲取完整 provider 列表")

# 執行
raw_model_data = get_raw_provider_info(model_id)