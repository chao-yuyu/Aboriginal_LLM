#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速測試腳本 - 用於快速測試特定查詢
使用方法: python quick_test.py "你的查詢"
"""

import sys
import requests
import json
import time

API_BASE = "http://localhost:8000"
TEST_UUID = "quick_test"

def quick_test(query):
    """快速測試單個查詢"""
    print(f"🧪 測試查詢: {query}")
    print("=" * 50)
    
    try:
        # 初始化客戶端
        init_response = requests.post(f"{API_BASE}/init_llm?uuid={TEST_UUID}")
        if init_response.status_code != 200:
            print(f"❌ 初始化失敗: {init_response.status_code}")
            return
        
        # 發送查詢
        params = {'uuid': TEST_UUID, 'query': query}
        chat_response = requests.post(f"{API_BASE}/chat_with_llm", params=params, timeout=30)
        
        if chat_response.status_code != 200:
            print(f"❌ 聊天失敗: {chat_response.status_code}")
            return
        
        print("✅ 聊天請求完成，正在處理...")
        
        # 等待並獲取回應
        time.sleep(2)
        queue_response = requests.post(f"{API_BASE}/res_llm_queue?uuid={TEST_UUID}", timeout=10)
        
        if queue_response.status_code != 200:
            print(f"❌ 獲取回應失敗: {queue_response.status_code}")
            return
        
        result = queue_response.json()
        
        if 'responses' not in result:
            print("❌ 沒有收到回應")
            return
        
        response_data = result['responses']
        response_type = response_data.get('type', '未知')
        
        print(f"\n📤 回應類型: {response_type}")
        
        if response_type == 'clothing_recommendation':
            print("🎯 ✅ 觸發了服飾推薦！")
            recommendations = response_data.get('recommendations', [])
            
            for rec in recommendations:
                tribe = rec['tribe']
                clothing = rec['clothing']
                shirt = clothing['衣服'][0]
                pants = clothing['褲子'][0]
                
                print(f"\n👥 族群: {tribe}")
                print(f"👕 上衣: {shirt['name']} (ID: {shirt['id']})")
                print(f"👖 褲子: {pants['name']} (ID: {pants['id']})")
                
        elif response_type in ['text', 'audio']:
            print("💬 LLM正常回應（未觸發推薦）")
            if 'display_text' in response_data:
                text = response_data['display_text']['text']
                print(f"\n🤖 LLM說: {text}")
        else:
            print(f"❓ 未知回應類型")
            print(json.dumps(response_data, ensure_ascii=False, indent=2))
            
    except requests.exceptions.Timeout:
        print("⏰ 請求超時")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} \"你的查詢\"")
        print("\n範例:")
        print(f"  python {sys.argv[0]} \"我想了解泰雅族的傳統衣服\"")
        print(f"  python {sys.argv[0]} \"今天天氣如何？\"")
        sys.exit(1)
    
    query = sys.argv[1]
    quick_test(query) 