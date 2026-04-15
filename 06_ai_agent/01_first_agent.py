"""
AI Agent 02: 带工具的 Agent（国内可用版）
=========================================
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3"

def ask_ollama(prompt, system=""):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    return response.json()["message"]["content"]

def web_search(query):
    """
    用 SerpAPI 搜索（需要 API Key，免费额度够用）
    或者用 Bing 搜索
    """
    # 方案1：使用 DuckDuckGo HTML（不需要 API）
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://html.duckduckgo.com/html/?q={query}"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            text = response.text
            # 提取搜索结果摘要
            import re
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', text)
            if snippets:
                return snippets[0].strip()
    except:
        pass
    
    return "搜索失败"

def agent(question):
    judge_system = """回复"需要搜索"或"直接回答"，不要其他内容。"""
    judge = ask_ollama(f"问题：{question}", judge_system)
    print(f"🤔 判断: {judge.strip()}")
    
    if "需要搜索" in judge:
        search_result = web_search(question)
        print(f"🔍 搜索结果: {search_result[:150]}...")
        
        answer_system = """你是一个助手。根据搜索结果回答用户问题。"""
        answer = ask_ollama(f"搜索结果：{search_result}\n\n问题：{question}", answer_system)
    else:
        answer = ask_ollama(question)
    
    return answer

# ===== 测试 =====
print("=" * 50)
print("AI Agent 02: 带工具的 Agent")
print("=" * 50)

print("\n问题1: 什么是深度学习？")
answer = agent("什么是深度学习？")
print(f"回答: {answer}")

print("\n问题2: 2024年奥运会在哪里？")
answer = agent("2024年奥运会在哪里？")
print(f"回答: {answer}")

print("\n" + "=" * 50)
print("✅ 完成！")
print("=" * 50)