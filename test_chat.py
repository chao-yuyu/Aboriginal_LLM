#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import uuid

class ChatTester:
    def __init__(self, base_url="http://localhost:8000", use_history=False):
        self.base_url = base_url
        self.client_uuid = str(uuid.uuid4())
        self.use_history = use_history
        self.init_client()
    
    def init_client(self):
        """初始化客戶端"""
        url = f"{self.base_url}/init_llm_json"
        data = {"uuid": self.client_uuid}
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            mode = "有歷史記錄" if self.use_history else "無歷史記錄"
            print(f"✅ 客戶端初始化成功，UUID: {self.client_uuid}")
            print(f"📚 對話模式: {mode}")
        else:
            print(f"❌ 初始化失敗: {response.text}")
    
    def clear_queue(self):
        """清空回應隊列"""
        url = f"{self.base_url}/clear_llm_queue"
        params = {"uuid": self.client_uuid}
        
        response = requests.post(url, params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"🧹 已清空 {result.get('message', '0')} 個舊回應")
        else:
            print(f"⚠️  清空隊列失敗: {response.text}")
    
    def chat(self, query):
        """發送聊天訊息"""
        print(f"\n🗣️  您: {query}")
        
        # 先清空舊的回應
        self.clear_queue()
        
        # 根據是否使用歷史記錄選擇不同的端點
        endpoint = "chat_with_llm_json_history" if self.use_history else "chat_with_llm_json"
        url = f"{self.base_url}/{endpoint}"
        data = {"uuid": self.client_uuid, "query": query}
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("✅ 訊息發送成功，正在處理...")
            return self.get_response()
        else:
            print(f"❌ 發送失敗: {response.text}")
            return None
    
    def get_response(self):
        """獲取回應"""
        url = f"{self.base_url}/res_llm_queue"
        params = {"uuid": self.client_uuid}
        
        response = requests.post(url, params=params)
        if response.status_code == 200:
            result = response.json()
            if "responses" in result:
                resp = result["responses"]
                print(f"🤖 回應類型: {resp.get('type', 'unknown')}")
                
                if resp.get("type") == "clothing_recommendation":
                    print(f"👕 推薦訊息: {resp.get('message', '')}")
                    print(f"🆔 服飾ID: {resp.get('unique_ids', [])}")
                    
                    recommendations = resp.get("recommendations", [])
                    for rec in recommendations:
                        tribe = rec.get("tribe")
                        clothing = rec.get("clothing", {})
                        print(f"\n🏛️  {tribe}:")
                        for category, items in clothing.items():
                            print(f"  📋 {category}:")
                            for item in items:
                                print(f"    • {item['name']} (ID: {item['id']})")
                                print(f"      {item['description']}")
                
                elif resp.get("type") == "text":
                    text_content = resp.get("display_text", {}).get("text", "")
                    print(f"💬 回應: {text_content}")
                
                return resp
            else:
                print("⚠️  沒有收到回應")
                return None
        else:
            print(f"❌ 獲取回應失敗: {response.text}")
            return None

def main():
    print("🎭 原住民服飾推薦系統測試工具")
    print("=" * 50)
    
    # 讓用戶選擇對話模式
    print("\n📚 請選擇對話模式:")
    print("1. 無歷史記錄模式 (每次對話獨立)")
    print("2. 有歷史記錄模式 (記住之前的對話)")
    
    while True:
        try:
            mode_choice = input("\n請輸入 1 或 2: ").strip()
            if mode_choice == "1":
                use_history = False
                print("✅ 已選擇: 無歷史記錄模式")
                break
            elif mode_choice == "2":
                use_history = True
                print("✅ 已選擇: 有歷史記錄模式")
                break
            else:
                print("❌ 請輸入 1 或 2")
        except KeyboardInterrupt:
            print("\n👋 再見！")
            return
    
    tester = ChatTester(use_history=use_history)
    
    # 預設測試案例
    test_cases = [
        "我想看泰雅族的衣服",
        "排灣族的褲子有哪些？",
        "魯凱族的傳統服飾",
        "我想要泰雅族和排灣族的全套服裝",
        "原住民服裝有什麼推薦的？"
    ]
    
    print("\n🧪 預設測試案例:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case}")
    
    print("\n請選擇:")
    print("• 輸入數字 1-5 執行預設測試")
    print("• 直接輸入您想測試的文字")
    if use_history:
        print("• 輸入 'history' 查看對話歷史")
        print("• 輸入 'clear' 清空對話歷史")
    print("• 輸入 'quit' 結束")
    
    while True:
        try:
            user_input = input("\n➤ ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 再見！")
                break
            elif user_input.lower() == 'history' and use_history:
                show_history(tester)
                continue
            elif user_input.lower() == 'clear' and use_history:
                clear_history(tester)
                continue
            
            if user_input.isdigit():
                num = int(user_input)
                if 1 <= num <= len(test_cases):
                    query = test_cases[num - 1]
                    tester.chat(query)
                else:
                    print("❌ 請輸入 1-5 的數字")
            elif user_input:
                tester.chat(user_input)
            else:
                print("⚠️  請輸入內容")
                
        except KeyboardInterrupt:
            print("\n👋 再見！")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")

def show_history(tester):
    """顯示對話歷史"""
    try:
        response = requests.get(f"{tester.base_url}/get_conversation_history", 
                              params={"uuid": tester.client_uuid})
        if response.status_code == 200:
            data = response.json()
            print(f"\n📚 對話歷史 (共 {data['conversation_count']} 條訊息):")
            print("-" * 50)
            for i, msg in enumerate(data['conversation_history'], 1):
                role = "🗣️ 您" if msg['role'] == 'user' else "🤖 助手"
                print(f"{i}. {role}: {msg['content'][:100]}...")
        else:
            print("❌ 無法獲取對話歷史")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

def clear_history(tester):
    """清空對話歷史"""
    try:
        response = requests.post(f"{tester.base_url}/clear_conversation_history",
                               params={"uuid": tester.client_uuid})
        if response.status_code == 200:
            data = response.json()
            print(f"🧹 {data['message']}")
        else:
            print("❌ 無法清空對話歷史")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    main() 