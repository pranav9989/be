#!/usr/bin/env python3
"""Test script for AssemblyAI WebSocket connection"""

import os
import websocket
import json
import base64
from urllib.parse import urlencode
import time

def test_assemblyai_connection():
    """Test basic WebSocket connection to AssemblyAI"""

    api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not found")
        return

    print("Testing AssemblyAI WebSocket connection...")

    # Try different parameter combinations
    test_configs = [
        {
            "name": "Basic params",
            "params": {"sample_rate": 16000, "encoding": "pcm_s16le"}
        },
        {
            "name": "With base model",
            "params": {"sample_rate": 16000, "encoding": "pcm_s16le", "model": "base"}
        },
        {
            "name": "With large-v2 model",
            "params": {"sample_rate": 16000, "encoding": "pcm_s16le", "model": "large-v2"}
        }
    ]

    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"   Params: {config['params']}")

        try:
            params = config['params']
            ws_url = f"wss://api.assemblyai.com/v2/realtime/ws?{urlencode(params)}"
            headers = {"Authorization": api_key}

            ws = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_message=lambda ws, msg: print(f"   Message: {msg[:100]}..."),
                on_error=lambda ws, err: print(f"   Error: {err}"),
                on_close=lambda ws, code, msg: print(f"   Closed: {code} - {msg}"),
                on_open=lambda ws: print("   Connected successfully!")
            )

            # Try to connect with timeout
            ws.run_forever(ping_interval=10, ping_timeout=5)

        except Exception as e:
            print(f"   Exception: {e}")

        # Small delay between tests
        time.sleep(1)

if __name__ == "__main__":
    test_assemblyai_connection()