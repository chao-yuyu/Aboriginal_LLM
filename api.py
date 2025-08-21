from fastapi import FastAPI, Query, Request, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json
import re
import requests
from uuid import uuid4
from queue import Queue
from datetime import datetime
import os
import base64
import numpy as np
from pydub import AudioSegment
from math import log, ceil
from io import BytesIO

api_key = APIKeyHeader(name="Authorization")

def verify_api_key(key: str = Depends(api_key)):
    if key != "N84ahY]PZP*EEQ":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

app = FastAPI(dependencies=[Depends(verify_api_key)])


# 添加請求模型
class ChatRequest(BaseModel):
    uuid: str
    query: str

class InitRequest(BaseModel):
    uuid: str

# Global dictionary to store queues and conversation history for each client
client_queues = {}
client_conversations = {}  # 新增：儲存每個客戶端的對話歷史

gpt_sovit_api = "http://127.0.0.1:9880/tts"

# 原住民族群服飾資料庫
INDIGENOUS_CLOTHING_DB = {
    "泰雅族": {
        "衣服": [
            {"id": "atayal_shirt_001", "name": "泰雅族傳統上衣", "type": "上衣", "description": "紅色幾何圖紋傳統上衣"}
        ],
        "褲子": [
            {"id": "atayal_pants_001", "name": "泰雅族傳統褲子", "type": "褲子", "description": "黑色配紅邊傳統褲子"}
        ]
    },
    "排灣族": {
        "衣服": [
            {"id": "paiwan_shirt_001", "name": "排灣族頭目服", "type": "上衣", "description": "華麗刺繡頭目專用上衣"}
        ],
        "褲子": [
            {"id": "paiwan_pants_001", "name": "排灣族傳統褲子", "type": "褲子", "description": "深藍色刺繡傳統褲子"}
        ]
    },
    "魯凱族": {
        "衣服": [
            {"id": "rukai_shirt_001", "name": "魯凱族百合花上衣", "type": "上衣", "description": "百合花圖案傳統上衣"}
        ],
        "褲子": [
            {"id": "rukai_pants_001", "name": "魯凱族傳統褲子", "type": "褲子", "description": "深色配亮色邊飾褲子"}
        ]
    }
}

# Function calling tools definition - OpenAI compatible
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_indigenous_clothing",
            "description": "推薦原住民傳統服飾。當用戶詢問或提到泰雅族、排灣族、魯凱族的服裝、衣服、褲子、傳統服飾時使用此功能。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tribes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["泰雅族", "排灣族", "魯凱族"]
                        },
                        "description": "要推薦服飾的原住民族群，可以是一個或多個"
                    },
                    "clothing_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["衣服", "褲子"]
                        },
                        "description": "要推薦的服飾類型，可以是衣服、褲子或兩者都要"
                    }
                },
                "required": ["tribes", "clothing_types"]
            }
        }
    }
]

def execute_recommend_indigenous_clothing(tribes: list, clothing_types: list) -> dict:
    """
    執行推薦原住民服飾的函數
    """
    recommendations = []
    total_items = []
    
    for tribe in tribes:
        if tribe in INDIGENOUS_CLOTHING_DB:
            tribe_rec = {
                "tribe": tribe,
                "clothing": {}
            }
            
            for clothing_type in clothing_types:
                if clothing_type in INDIGENOUS_CLOTHING_DB[tribe]:
                    tribe_rec["clothing"][clothing_type] = INDIGENOUS_CLOTHING_DB[tribe][clothing_type]
                    total_items.extend(INDIGENOUS_CLOTHING_DB[tribe][clothing_type])
            
            if tribe_rec["clothing"]:  # 只添加有服飾資料的族群
                recommendations.append(tribe_rec)
    
    # 提取所有unique IDs
    unique_ids = [item["id"] for item in total_items]
    
    return {
        "success": True,
        "recommendations": recommendations,
        "unique_ids": unique_ids,
        "total_items": len(total_items),
        "message": f"為您推薦 {', '.join(tribes)} 的 {', '.join(clothing_types)}"
    }

# 關鍵字檢測作為備用方案
TRIBE_KEYWORDS = ["泰雅族", "排灣族", "魯凱族"]
CLOTHING_KEYWORDS = ["衣服", "上衣", "褲子", "服飾", "穿著", "傳統服裝", "族服", "服裝", "衣著"]

def detect_clothing_recommendation_need(text: str) -> dict:
    """
    檢測文本中是否包含原住民族群和衣服相關關鍵字
    返回檢測結果和推薦的衣服 - 作為 function calling 的備用方案
    """
    detected_tribes = []
    has_clothing_keyword = False
    
    # 檢測族群關鍵字
    for tribe in TRIBE_KEYWORDS:
        if tribe in text:
            detected_tribes.append(tribe)
    
    # 檢測衣服相關關鍵字
    for keyword in CLOTHING_KEYWORDS:
        if keyword in text:
            has_clothing_keyword = True
            break
    
    # 如果同時包含族群和衣服關鍵字，返回推薦
    if detected_tribes and has_clothing_keyword:
        # 確定服飾類型
        clothing_types = []
        if any(word in text for word in ["衣服", "上衣", "服裝"]):
            clothing_types.append("衣服")
        if any(word in text for word in ["褲子"]):
            clothing_types.append("褲子")
        
        # 如果沒有明確指定，推薦全套
        if not clothing_types:
            clothing_types = ["衣服", "褲子"]
        
        return {
            "should_recommend": True,
            "detected_tribes": detected_tribes,
            "clothing_types": clothing_types
        }
    
    return {"should_recommend": False}

@app.post("/init_llm")
async def init_llm(uuid: str = Query(..., description="UUID of the client")):
    if uuid not in client_queues:
        client_queues[uuid] = Queue()
        client_conversations[uuid] = []  # 初始化對話歷史
    return {"message": "Queue initialized for client", "uuid": uuid}

@app.post("/init_llm_json")
async def init_llm_json(request: InitRequest):
    """使用 JSON body 初始化 LLM - 更容易處理中文"""
    if request.uuid not in client_queues:
        client_queues[request.uuid] = Queue()
        client_conversations[request.uuid] = []  # 初始化對話歷史
    return {"message": "Queue initialized for client", "uuid": request.uuid}

@app.post("/chat_with_llm")
async def chat_with_llm(uuid: str = Query(..., description="UUID of the client"), query: str = Query(..., description="Query text")):
    # 原有的 query parameter 版本（無歷史記錄）
    return await _process_chat(uuid, query, use_history=False)

@app.post("/chat_with_llm_json")
async def chat_with_llm_json(request: ChatRequest):
    """使用 JSON body 聊天 - 更容易處理中文（無歷史記錄）"""
    return await _process_chat(request.uuid, request.query, use_history=False)

@app.post("/chat_with_llm_json_history")
async def chat_with_llm_json_history(request: ChatRequest):
    """使用 JSON body 聊天 - 支援對話歷史記錄"""
    return await _process_chat(request.uuid, request.query, use_history=True)

async def _process_chat(uuid: str, query: str, use_history: bool = False):
    """處理聊天的核心邏輯"""
    if uuid not in client_queues:
        return {"error": "Client UUID not initialized. Please call /init_llm first."}

    def remove_think_tags(text: str) -> str:
        # 保留原始內容，不做任何處理
        return text.replace("<think>", "").replace("</think>", "").strip()
    
    def is_only_punctuation(text: str) -> bool:
        # 定義標點符號
        punctuation = """!()-[]{};:'"\\,<>./?@#$%^&*_~。，！？；：「」（）、"""
        return all(char in punctuation or char.isspace() for char in text)

    def make_chunks(audio_segment, chunk_length):
        """
        Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
        long.
        if chunk_length is 50 then you'll get a list of 50 millisecond long audio
        segments back (except the last one, which can be shorter)
        """
        number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
        return [audio_segment[i * chunk_length:(i + 1) * chunk_length]
                for i in range(int(number_of_chunks))]

    def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
        """
        Calculate the normalized volume (RMS) for each chunk of the audio.

        Parameters:
            audio (AudioSegment): The audio segment to process.
            chunk_length_ms (int): The length of each audio chunk in milliseconds.

        Returns:
            list: Normalized volumes for each chunk.
        """
        chunks = make_chunks(audio, chunk_length_ms)
        volumes = [chunk.rms for chunk in chunks]
        max_volume = max(volumes)
        if max_volume == 0:
            raise ValueError("Audio is empty or all zero.")
        return [volume / max_volume for volume in volumes]

    def text_to_speech(text: str):
        try:
            # TTS API parameters
            data = {
                "text": text,
                "text_lang": "zh",
                "ref_audio_path": "C:\\Users\\victo\\Desktop\\AI\\Open-LLM-VTuber\\2ld2_tts.wav",
                "prompt_lang": "zh",
                "prompt_text": "大家好!我是極深空計畫的惡倫蒂兒，各位訊號員們，你們應該都沒忘記我們 nexus 虛實之側演唱會的時間吧!",
                "text_split_method": "cut0",
                "batch_size": "1",
                "media_type": "wav",
                "streaming_mode": "false"
            }

            # Send GET request to the TTS API with shorter timeout
            response = requests.get(gpt_sovit_api, params=data, timeout=5)
            
            if response.status_code == 200:
                # Get the audio data
                audio_data = response.content
                
                # Create audio segment from response content
                audio = AudioSegment.from_file(BytesIO(audio_data))
                
                # Calculate volumes before converting to base64
                volumes = _get_volume_by_chunks(audio, 20)
                
                # Export to WAV format in memory
                wav_buffer = BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                audio_bytes = wav_buffer.read()
                
                # Convert to base64
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

                return {
                    'type': 'audio',
                    'audio': base64_audio,
                    'volumes': volumes,
                    'slice_length': 20,
                    'display_text': {
                        'text': text,
                        'name': 'Shizuku',
                        'avatar': 'shizuku.png'
                    },
                    'actions': {},
                    'forwarded': False
                }
            else:
                print(f"TTS API returned status code: {response.status_code}")
                return create_text_only_response(text)
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            # Return text-only response when TTS fails
            return create_text_only_response(text)

    def create_text_only_response(text: str):
        """Create a text-only response when TTS is not available"""
        return {
            'type': 'text',
            'display_text': {
                'text': text,
                'name': 'Shizuku',
                'avatar': 'shizuku.png'
            },
            'actions': {},
            'forwarded': False
        }

    # 系統prompt - 讓LLM扮演服裝店店員並了解function calling
    system_prompt = """
/no_think
你是一位專業的服裝店店員，特別專精於原住民傳統服飾。你的特點是：

1. **專業知識**：對泰雅族、排灣族、魯凱族的傳統服飾有深入了解
2. **熱情服務**：總是以親切、熱情的態度為客人服務
3. **文化尊重**：對原住民文化抱持尊重和欣賞的態度
4. **推薦能力**：能根據客人的需求推薦最適合的傳統服飾

重要：當客人詢問或提到以下內容時，你**必須**使用 recommend_indigenous_clothing 函數：
- 泰雅族、排灣族、魯凱族的服裝
- 傳統服飾、族服、原住民服裝
- 衣服、上衣、褲子等服飾類型
- 想要了解或購買原住民服飾

使用函數時的參數說明：
- tribes: 必須根據用戶明確提到的族群填入，例如["泰雅族"]、["排灣族"]或["泰雅族", "排灣族"]
- clothing_types: 必須根據用戶需求精確填入["衣服"]、["褲子"]或["衣服", "褲子"]

範例：
- 用戶說「我想看泰雅族的衣服」→ 調用函數：tribes=["泰雅族"], clothing_types=["衣服"]
- 用戶說「排灣族的褲子」→ 調用函數：tribes=["排灣族"], clothing_types=["褲子"]
- 用戶說「魯凱族傳統服飾」→ 調用函數：tribes=["魯凱族"], clothing_types=["衣服", "褲子"]

請用溫暖、專業的語調與客人對話，並**務必**在適當時機調用推薦函數。"""
    
    url = "http://localhost:11434/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }

    # 構建訊息列表
    messages = [{"role": "system", "content": system_prompt}]
    
    if use_history and uuid in client_conversations:
        # 添加歷史對話（最多保留最近10輪對話）
        messages.extend(client_conversations[uuid][-20:])  # 保留最近20條訊息（10輪對話）
    
    # 添加當前用戶訊息
    messages.append({"role": "user", "content": query})

    # 構建包含function calling的payload
    payload = {
        "model": "qwen3:8b",
        "messages": messages,
        "temperature": 0.7,
        "tools": FUNCTION_TOOLS,
        "tool_choice": "auto",
        "stream": True
    }

    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as r:
            accumulated_text = ""
            function_calls = []
            
            for line in r.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        chunk = line[len(b"data: "):]
                        if chunk.strip() and chunk.strip() != b"[DONE]":
                            try:
                                chunk_str = chunk.decode("utf-8").strip()
                                if chunk_str:
                                    chunk_json = json.loads(chunk_str)
                                    
                                    if "choices" in chunk_json and chunk_json["choices"]:
                                        choice = chunk_json["choices"][0]
                                        
                                        # 處理一般文字回應
                                        if "delta" in choice and "content" in choice["delta"]:
                                            content = choice["delta"].get("content", "")
                                            content = remove_think_tags(re.sub(r"\[.*?\]", "", content))

                                            if content.strip():
                                                accumulated_text += content

                                                if any(char in accumulated_text for char in '.!?。！？'):
                                                    # 確認累積的文字不是只有標點符號
                                                    if not is_only_punctuation(accumulated_text):
                                                        # Convert text to speech and get the audio data
                                                        audio_data = text_to_speech(accumulated_text)
                                                        if audio_data:
                                                            client_queues[uuid].put(audio_data)
                                                    accumulated_text = ""
                                        
                                        # 處理function calling
                                        if "delta" in choice and "tool_calls" in choice["delta"]:
                                            tool_calls = choice["delta"]["tool_calls"]
                                            for tool_call in tool_calls:
                                                if tool_call.get("function"):
                                                    function_calls.append(tool_call)
                                        
                                        # 處理完整的tool_calls (非streaming模式)
                                        if "message" in choice and choice["message"].get("tool_calls"):
                                            function_calls.extend(choice["message"]["tool_calls"])
                            
                            except json.JSONDecodeError as je:
                                print(f"JSON decode error: {str(je)} - Chunk: {chunk[:100]}...")
                                continue
                            except UnicodeDecodeError as ue:
                                print(f"Unicode decode error: {str(ue)}")
                                continue
                    except Exception as e:
                        print(f"Error processing line: {str(e)}")
                        continue

            # # 處理完整的LLM回應（如果有文字回應）
            # if accumulated_text.strip():
            #     audio_data = text_to_speech(accumulated_text)
            #     if audio_data:
            #         client_queues[uuid].put(audio_data)

            # 處理function calls
            if function_calls:
                print(f"DEBUG: Found {len(function_calls)} function calls")
                for tool_call in function_calls:
                    print(f"DEBUG: Tool call: {tool_call}")
                    if tool_call.get("function", {}).get("name") == "recommend_indigenous_clothing":
                        try:
                            # 解析function arguments
                            args_str = tool_call["function"].get("arguments", "{}")
                            print(f"DEBUG: Raw arguments: {args_str}")
                            
                            if isinstance(args_str, str):
                                args = json.loads(args_str)
                            else:
                                args = args_str
                            
                            print(f"DEBUG: Parsed arguments: {args}")
                            
                            # 執行function
                            result = execute_recommend_indigenous_clothing(
                                tribes=args.get("tribes", []),
                                clothing_types=args.get("clothing_types", [])
                            )
                            
                            print(f"DEBUG: Function execution result: {result}")
                            
                            # 將結果添加到隊列
                            recommendation_data = {
                                'type': 'clothing_recommendation',
                                'function_result': result['recommendations'],
                            }
                            client_queues[uuid].put(recommendation_data)
                            
                            # # 更新accumulated_text以包含function結果
                            # function_response = f"\n\n{result['message']}，包含以下服飾ID：{', '.join(result['unique_ids'])}"
                            # accumulated_text += function_response
                            
                        except Exception as e:
                            print(f"Function execution error: {str(e)}")
                            error_response = "抱歉，推薦服飾時發生錯誤，請稍後再試。"
                            accumulated_text += error_response
            else:
                print("DEBUG: No function calls detected")
                print(f"DEBUG: Accumulated text: {accumulated_text[:200]}...")
                
                # 備用機制：如果沒有檢測到 function calling，使用關鍵字檢測
                clothing_detection = detect_clothing_recommendation_need(query)
                if clothing_detection["should_recommend"]:
                    print("DEBUG: Fallback detection triggered")
                    print(f"DEBUG: Detected tribes: {clothing_detection['detected_tribes']}")
                    print(f"DEBUG: Detected clothing types: {clothing_detection['clothing_types']}")
                    
                    # 執行推薦函數
                    result = execute_recommend_indigenous_clothing(
                        tribes=clothing_detection["detected_tribes"],
                        clothing_types=clothing_detection["clothing_types"]
                    )
                    
                    # 將結果添加到隊列
                    recommendation_data = {
                        'type': 'clothing_recommendation',
                        'function_result': result,
                        'recommendations': result["recommendations"],
                        'unique_ids': result["unique_ids"],
                        'message': result["message"],
                        'display_text': {
                            'text': f"為您推薦以下傳統服飾 (共{result['total_items']}件)",
                            'name': 'Shizuku',
                            'avatar': 'shizuku.png'
                        },
                        'actions': {
                            'type': 'show_clothing_recommendations',
                            'data': result["recommendations"],
                            'unique_ids': result["unique_ids"]
                        },
                        'forwarded': False
                    }
                    client_queues[uuid].put(recommendation_data)
                    
                    # 更新accumulated_text
                    function_response = f"\n\n{result['message']}，包含以下服飾ID：{', '.join(result['unique_ids'])}"
                    accumulated_text += function_response



    except Exception as e:
        print(f"Error in chat_with_llm: {str(e)}")
        error_response = "抱歉，處理您的請求時發生錯誤，請稍後再試。"
        audio_data = text_to_speech(error_response)
        if audio_data:
            client_queues[uuid].put(audio_data)

    # 如果使用歷史記錄，保存當前對話
    if use_history and uuid in client_conversations:
        # 保存用戶訊息
        client_conversations[uuid].append({"role": "user", "content": query})
        
        # 保存助手回應（如果有）
        if accumulated_text.strip():
            client_conversations[uuid].append({"role": "assistant", "content": accumulated_text})
        
        # 限制歷史記錄長度（保留最近30條訊息）
        if len(client_conversations[uuid]) > 30:
            client_conversations[uuid] = client_conversations[uuid][-30:]

    return {"message": "finished llm chat"}

@app.post("/res_llm_queue")
async def res_llm_queue(uuid: str = Query(..., description="UUID of the client")):
    if uuid not in client_queues:
        return {"error": "Client UUID not initialized. Please call /init_llm first."}

    # 獲取最新的回應，清空隊列中的舊回應
    if not client_queues[uuid].empty():
        return {"responses": client_queues[uuid].get()}
    else:
        return {"error": "No responses available"}

@app.post("/clear_llm_queue") 
async def clear_llm_queue(uuid: str = Query(..., description="UUID of the client")):
    """清空指定客戶端的回應隊列"""
    if uuid not in client_queues:
        return {"error": "Client UUID not initialized. Please call /init_llm first."}
    
    cleared_count = 0
    while not client_queues[uuid].empty():
        client_queues[uuid].get()
        cleared_count += 1
    
    return {"message": f"Cleared {cleared_count} responses from queue", "uuid": uuid}

@app.get("/get_conversation_history")
async def get_conversation_history(uuid: str = Query(..., description="UUID of the client")):
    """獲取指定客戶端的對話歷史"""
    if uuid not in client_conversations:
        return {"error": "Client UUID not found or no conversation history"}
    
    return {
        "uuid": uuid,
        "conversation_count": len(client_conversations[uuid]),
        "conversation_history": client_conversations[uuid]
    }

@app.post("/clear_conversation_history")
async def clear_conversation_history(uuid: str = Query(..., description="UUID of the client")):
    """清空指定客戶端的對話歷史"""
    if uuid not in client_conversations:
        return {"error": "Client UUID not found"}
    
    cleared_count = len(client_conversations[uuid])
    client_conversations[uuid] = []
    
    return {"message": f"Cleared {cleared_count} conversation messages", "uuid": uuid}

@app.get("/conversation_stats")
async def conversation_stats():
    """獲取所有客戶端的對話統計"""
    stats = {}
    total_messages = 0
    
    for uuid, history in client_conversations.items():
        message_count = len(history)
        stats[uuid] = {
            "message_count": message_count,
            "last_activity": "有歷史記錄" if message_count > 0 else "無歷史記錄"
        }
        total_messages += message_count
    
    return {
        "total_clients": len(client_conversations),
        "total_messages": total_messages,
        "clients": stats
    }

@app.get("/get_clothing_recommendations")
async def get_clothing_recommendations(tribe: str = Query(..., description="原住民族群名稱")):
    """
    根據族群名稱獲取服飾推薦
    """
    if tribe not in INDIGENOUS_CLOTHING_DB:
        return {"error": f"找不到 {tribe} 的服飾資料"}
    
    return {
        "tribe": tribe,
        "clothing": INDIGENOUS_CLOTHING_DB[tribe],
        "message": f"為您推薦 {tribe} 傳統服飾"
    }

@app.get("/get_clothing_by_id")
async def get_clothing_by_id(clothing_id: str = Query(..., description="服飾的唯一ID")):
    """
    根據唯一ID獲取特定服飾詳細資訊
    """
    for tribe, categories in INDIGENOUS_CLOTHING_DB.items():
        for category, items in categories.items():
            for item in items:
                if item["id"] == clothing_id:
                    return {
                        "id": item["id"],
                        "name": item["name"],
                        "type": item["type"],
                        "description": item["description"],
                        "tribe": tribe,
                        "category": category
                    }
    
    return {"error": f"找不到ID為 {clothing_id} 的服飾"}

@app.get("/get_all_tribes_clothing")
async def get_all_tribes_clothing():
    """
    獲取所有族群的服飾資料
    """
    return {
        "tribes": list(INDIGENOUS_CLOTHING_DB.keys()),
        "clothing_database": INDIGENOUS_CLOTHING_DB,
        "total_items": sum(len(categories["衣服"]) + len(categories["褲子"]) 
                          for categories in INDIGENOUS_CLOTHING_DB.values())
    }

# 新增測試function calling的端點
@app.post("/test_function_calling")
async def test_function_calling(query: str = Query(..., description="測試查詢")):
    """
    測試function calling功能的端點
    """
    result = execute_recommend_indigenous_clothing(
        tribes=["泰雅族", "排灣族"], 
        clothing_types=["衣服", "褲子"]
    )
    return result