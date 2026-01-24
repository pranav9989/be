import websocket
import json
import threading
import time
import os
import random
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

def warmup_assemblyai():
    """
    Warm up AssemblyAI using your specific interview audio file.
    This ensures consistent warmup with real speech patterns.
    """
    print("🔥 Warming up AssemblyAI with your interview audio file...")
    
    def noop(*args, **kwargs):
        pass

    try:
        # Path to your specific interview audio file
        audio_file = "D:\\skin disease\\BE_PROJECT\\uploads\\interview_1_20260124_171606.wav"
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            print("📂 Available files in uploads folder:")
            uploads_dir = "D:\\skin disease\\BE_PROJECT\\uploads\\"
            if os.path.exists(uploads_dir):
                files = os.listdir(uploads_dir)
                wav_files = [f for f in files if f.lower().endswith('.wav')]
                for f in wav_files[:10]:  # Show first 10 WAV files
                    print(f"  - {f}")
                if wav_files:
                    # Use the most recent WAV file
                    latest_file = max(wav_files, 
                                     key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f)))
                    audio_file = os.path.join(uploads_dir, latest_file)
                    print(f"🔄 Using latest file instead: {latest_file}")
                else:
                    print("🔄 Creating synthetic warmup audio...")
                    return warmup_assemblyai()
            else:
                print("🔄 Creating synthetic warmup audio...")
                return warmup_assemblyai()
        
        # Read the audio file
        import wave
        with wave.open(audio_file, 'rb') as wf:
            # Get audio parameters
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames_count = wf.getnframes()
            duration = frames_count / sample_rate
            
            # Verify format (must be 16kHz, mono, 16-bit for AssemblyAI)
            if sample_rate != 16000:
                print(f"⚠️ Audio file is {sample_rate}Hz (needs 16000Hz)")
                # We'll still try to use it, but warn the user
                
            if channels != 1:
                print(f"⚠️ Audio file has {channels} channels (needs mono)")
                # We'll still try to use it
            
            frames = wf.readframes(frames_count)
        
        print(f"📊 Using warmup file: {audio_file}")
        print(f"   Size: {os.path.getsize(audio_file)} bytes")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Channels: {channels}")
        print(f"   Sample width: {sample_width} bytes")
        
        # Start AssemblyAI
        streamer = AssemblyAIWebSocketStreamer(
            on_partial=noop,
            on_final=noop,
            on_error=noop
        )
        streamer.start()
        
        # Send the audio in chunks (mimicking real-time streaming speed)
        CHUNK_SIZE = 3200  # 200ms chunks at 16kHz, 16-bit mono
        total_bytes = len(frames)
        bytes_sent = 0
        
        print(f"📤 Streaming {total_bytes} bytes in {CHUNK_SIZE}-byte chunks...")
        
        for i in range(0, total_bytes, CHUNK_SIZE):
            chunk = frames[i:i + CHUNK_SIZE]
            if not chunk:
                break
                
            streamer.send_audio(chunk)
            bytes_sent += len(chunk)
            
            # Calculate progress
            progress = (bytes_sent / total_bytes) * 100
            if i % (CHUNK_SIZE * 10) == 0:  # Log every 10 chunks
                print(f"   Progress: {progress:.1f}% ({bytes_sent}/{total_bytes} bytes)")
            
            # Add small delay to mimic real-time streaming
            # At 16kHz, 16-bit mono: 16000 * 2 = 32000 bytes per second
            # CHUNK_SIZE / 32000 = seconds per chunk
            time.sleep(CHUNK_SIZE / (sample_rate * sample_width))
        
        print(f"✅ Sent {bytes_sent} bytes to AssemblyAI")
        
        # Wait a bit for processing
        time.sleep(1.0)
        streamer.stop()
        
        print("✅ AssemblyAI successfully warmed up with real interview audio")
        print(f"   Used file: {os.path.basename(audio_file)}")
        
    except Exception as e:
        print(f"❌ Warmup from file failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback
        print("🔄 Trying fallback warmup...")
        try:
            warmup_assemblyai()
        except Exception as e2:
            print(f"❌ Fallback warmup also failed: {e2}")