from fastapi import FastAPI, Query
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

app = FastAPI()

# Global dictionary to store queues for each client
client_queues = {}

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

# 關鍵字檢測
TRIBE_KEYWORDS = ["泰雅族", "排灣族", "魯凱族"]
CLOTHING_KEYWORDS = ["衣服", "上衣", "褲子", "服飾", "穿著", "傳統服裝", "族服", "服裝", "衣著"]

def detect_clothing_recommendation_need(text: str) -> dict:
    """
    檢測文本中是否包含原住民族群和衣服相關關鍵字
    返回檢測結果和推薦的衣服
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
        recommendations = []
        for tribe in detected_tribes:
            if tribe in INDIGENOUS_CLOTHING_DB:
                tribe_clothing = INDIGENOUS_CLOTHING_DB[tribe]
                recommendations.append({
                    "tribe": tribe,
                    "clothing": {
                        "衣服": tribe_clothing["衣服"],
                        "褲子": tribe_clothing["褲子"]
                    }
                })
        
        return {
            "should_recommend": True,
            "detected_tribes": detected_tribes,
            "recommendations": recommendations
        }
    
    return {"should_recommend": False}

# Function calling tools definition
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_indigenous_clothing",
            "description": "當使用者提到原住民族群（泰雅族、排灣族、魯凱族）和衣服相關內容時，推薦對應的傳統服飾",
            "parameters": {
                "type": "object",
                "properties": {
                    "tribes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "檢測到的原住民族群名稱"
                    },
                    "clothing_type": {
                        "type": "string",
                        "enum": ["衣服", "褲子", "全套"],
                        "description": "推薦的服飾類型"
                    }
                },
                "required": ["tribes", "clothing_type"]
            }
        }
    }
]

@app.post("/init_llm")
async def init_llm(uuid: str = Query(..., description="UUID of the client")):
    if uuid not in client_queues:
        client_queues[uuid] = Queue()
    return {"message": "Queue initialized for client", "uuid": uuid}

@app.post("/chat_with_llm")
async def chat_with_llm(uuid: str = Query(..., description="UUID of the client"), query: str = Query(..., description="Query text")):
    if uuid not in client_queues:
        return {"error": "Client UUID not initialized. Please call /init_llm first."}

    def remove_think_tags(text: str) -> str:
        # 保留原始內容，不做任何處理
        return text

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

    # 檢測是否需要推薦原住民服飾
    clothing_detection = detect_clothing_recommendation_need(query)
    
    # 如果檢測到需要推薦服飾，直接添加推薦到隊列
    if clothing_detection["should_recommend"]:
        recommendation_data = {
            'type': 'clothing_recommendation',
            'recommendations': clothing_detection["recommendations"],
            'message': f"根據您提到的{', '.join(clothing_detection['detected_tribes'])}，我為您推薦以下傳統服飾：",
            'display_text': {
                'text': f"為您推薦{', '.join(clothing_detection['detected_tribes'])}傳統服飾",
                'name': 'Shizuku',
                'avatar': 'shizuku.png'
            },
            'actions': {
                'type': 'show_clothing_recommendations',
                'data': clothing_detection["recommendations"]
            },
            'forwarded': False
        }
        client_queues[uuid].put(recommendation_data)
    
    # 系統prompt - 讓LLM扮演服裝店店員
    system_prompt = """你是一位專業的服裝店店員，特別專精於原住民傳統服飾。你的特點是：

1. **專業知識**：對泰雅族、排灣族、魯凱族的傳統服飾有深入了解
2. **熱情服務**：總是以親切、熱情的態度為客人服務
3. **文化尊重**：對原住民文化抱持尊重和欣賞的態度
4. **推薦能力**：能根據客人的需求推薦最適合的傳統服飾

當客人詢問原住民族群的服飾時，你會：
- 熱情地介紹各族群服飾的特色和文化意義
- 主動推薦相關的傳統服飾
- 分享服飾背後的文化故事
- 以專業且親切的語氣回應

請用溫暖、專業的語調與客人對話，就像一位真正的服裝店店員一樣。"""
    
    url = "http://localhost:11434/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0.7,
        "stream": True
    }
    
    # 移除 function calling 配置，因為 Ollama 可能不支持
    # if clothing_detection["should_recommend"]:
    #     payload["tools"] = FUNCTION_TOOLS
    #     payload["tool_choice"] = "auto"

    with requests.post(url, headers=headers, json=payload, stream=True) as r:
        accumulated_text = ""
        
        for line in r.iter_lines():
            if line and line.startswith(b"data: "):
                try:
                    chunk = line[len(b"data: "):]
                    if chunk.strip() and chunk.strip() != b"[DONE]":  # 確保 chunk 不是空的且不是結束標記
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                            if chunk_str:  # 確保解碼後的字符串不為空
                                chunk_json = json.loads(chunk_str)
                                
                                # 檢查是否有正常的文字回應
                                if "choices" in chunk_json and chunk_json["choices"]:
                                    choice = chunk_json["choices"][0]
                                    
                                    # 處理文字回應
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"].get("content", "")
                                        # 直接累積所有LLM輸出，不做任何處理或截斷
                                        if content:
                                            accumulated_text += content
                        except json.JSONDecodeError as je:
                            print(f"JSON decode error: {str(je)} - Chunk: {chunk[:100]}...")
                            continue
                        except UnicodeDecodeError as ue:
                            print(f"Unicode decode error: {str(ue)}")
                            continue
                except Exception as e:
                    print(f"Error processing line: {str(e)}")
                    continue

        # 處理完整的LLM回應
        if accumulated_text.strip():
            audio_data = text_to_speech(accumulated_text)
            if audio_data:
                client_queues[uuid].put(audio_data)

    return {"message": "finished llm chat"}

@app.post("/res_llm_queue")
async def res_llm_queue(uuid: str = Query(..., description="UUID of the client")):
    if uuid not in client_queues:
        return {"error": "Client UUID not initialized. Please call /init_llm first."}

    # responses = []
    # while not client_queues[uuid].empty():
    #     responses.append(client_queues[uuid].get())

    return {"responses": client_queues[uuid].get()}

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



