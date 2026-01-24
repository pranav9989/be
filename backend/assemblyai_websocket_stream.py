import websocket
import json
import threading
import time
import os
from urllib.parse import urlencode


class AssemblyAIWebSocketStreamer:
    """
    Direct WebSocket connection to AssemblyAI Universal Streaming API
    (NO SDK, NO Socket.IO)
    """

    def __init__(self, on_partial, on_final, on_error=None):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error

        self.ws = None
        self.ws_thread = None
        self.is_active = False

        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY not set")

    # ---------------------------
    # START STREAM
    # ---------------------------
    def start(self):
        # Match the official docs - v3 API with query parameters
        params = {
            "sample_rate": 16000,
            "format_turns": True  # Request formatted final transcripts
        }

        ws_url = (
            "wss://streaming.assemblyai.com/v3/ws?"
            + urlencode(params)
        )

        headers = {
            "Authorization": self.api_key
        }

        self.ws = websocket.WebSocketApp(
            ws_url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"ping_interval": 5, "ping_timeout": 3}
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait until socket is active
        timeout = 5
        start = time.time()
        while not self.is_active and time.time() - start < timeout:
            time.sleep(0.05)

        if not self.is_active:
            raise RuntimeError("AssemblyAI WebSocket failed to connect")

        print("✅ AssemblyAI realtime streaming started")

    # ---------------------------
    # SEND AUDIO (PCM 16-bit)
    # ---------------------------
    def send_audio(self, pcm_bytes: bytes):
        if not self.is_active:
            return

        if not pcm_bytes:
            return

        try:
            # Send raw binary audio data (matches official docs)
            self.ws.send(pcm_bytes, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print("❌ Failed to send audio:", e)

    # ---------------------------
    # STOP STREAM
    # ---------------------------
    def stop(self):
        if not self.is_active:
            return

        try:
            # Send termination message (matches official docs)
            terminate_message = {"type": "Terminate"}
            self.ws.send(json.dumps(terminate_message))
            time.sleep(0.2)
            self.ws.close()
        except Exception:
            pass

        self.is_active = False
        print("🛑 AssemblyAI streaming stopped")

    # ---------------------------
    # SOCKET EVENTS
    # ---------------------------
    def _on_open(self, ws):
        self.is_active = True
        print("🔗 Connected to AssemblyAI")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == "Begin":
                session_id = data.get('id')
                expires_at = data.get('expires_at')
                print(f"Session began: ID={session_id}")
            elif msg_type == "Turn":
                transcript = data.get('transcript', '')
                formatted = data.get('turn_is_formatted', False)

                if not transcript.strip():
                    return

                if formatted:
                    # This is a final, formatted transcript
                    self.on_final(transcript)
                else:
                    # This is a partial transcript
                    self.on_partial(transcript)
            elif msg_type == "Termination":
                audio_duration = data.get('audio_duration_seconds', 0)
                session_duration = data.get('session_duration_seconds', 0)
                print(f"Session Terminated: Audio={audio_duration}s, Session={session_duration}s")

        except Exception as e:
            print("❌ Message handling error:", e)

    def _on_error(self, ws, error):
        self.is_active = False
        print("❌ AssemblyAI WebSocket error:", error)
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, code, reason):
        self.is_active = False
        print(f"🔌 AssemblyAI WebSocket closed ({code}): {reason}")
