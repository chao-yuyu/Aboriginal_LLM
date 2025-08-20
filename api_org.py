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

            # Send GET request to the TTS API
            response = requests.get(gpt_sovit_api, params=data, timeout=120)
            
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
            return None
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None

    url = "http://localhost:11434/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen3:8b",
        "messages": [
            {"role": "user", "content": query},
        ],
        "temperature": 0.0,
        "stream": True
    }

    with requests.post(url, headers=headers, json=payload, stream=True) as r:
        accumulated_text = ""
        for line in r.iter_lines():
            if line and line.startswith(b"data: "):
                try:
                    chunk = line[len(b"data: "):]
                    if chunk.strip():  # 確保 chunk 不是空的
                        try:
                            chunk_json = json.loads(chunk.decode("utf-8"))
                            content = chunk_json["choices"][0]["delta"].get("content", "")
                            cleaned_text = remove_think_tags(re.sub(r"\[.*?\]", "", content))
                            
                            if cleaned_text.strip():
                                accumulated_text += cleaned_text
                                
                                # When we have a complete sentence or significant chunk, process it
                                if any(char in accumulated_text for char in '.!?。！？'):
                                    # 確認累積的文字不是只有標點符號
                                    if not is_only_punctuation(accumulated_text):
                                        # Convert text to speech and get the audio data
                                        audio_data = text_to_speech(accumulated_text)
                                        if audio_data:
                                            client_queues[uuid].put(audio_data)
                                    accumulated_text = ""  # Reset accumulated text
                        except json.JSONDecodeError as je:
                            print(f"Invalid JSON in chunk: {str(je)}")
                            continue
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue

        # Process any remaining text
        if accumulated_text.strip() and not is_only_punctuation(accumulated_text):
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



