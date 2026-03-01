import websocket
import json
import threading
import time
import os
import numpy as np
from urllib.parse import urlencode


class AssemblyAIWebSocketStreamer:
    """
    Direct WebSocket connection to AssemblyAI Universal Streaming API
    (NO SDK, NO Socket.IO)
    """

    def __init__(self, on_partial, on_final, on_error=None, on_ready=None):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.on_ready = on_ready  # 🔥 CRITICAL: Add this callback

        self.ws = None
        self.ws_thread = None
        self.is_active = False
        self.is_ready = False  # 🔥 Track when ready to receive audio
        self.early_audio_buffer = []  # 🔥 Buffer for early audio chunks
        self.buffer_lock = threading.Lock()  # 🔥 Thread-safe buffer access

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

        print(f"🔗 Connecting to AssemblyAI at: {ws_url}")
        
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

        # Wait until socket is active (with longer timeout for cold start)
        timeout = 10  # Increased timeout for first-time cold start
        start = time.time()
        while not self.is_active and time.time() - start < timeout:
            time.sleep(0.1)

        if not self.is_active:
            raise RuntimeError(f"AssemblyAI WebSocket failed to connect within {timeout} seconds")

        print("✅ AssemblyAI realtime streaming started")
        print("   Waiting for session to begin...")

    # ---------------------------
    # SEND AUDIO (PCM 16-bit) WITH BUFFERING
    # ---------------------------
    def send_audio(self, pcm_bytes: bytes):
        if not pcm_bytes or len(pcm_bytes) == 0:
            return

        # Auto-reconnect if thread died mysteriously
        if not self.is_active and hasattr(self, 'ws_thread') and self.ws_thread and not self.ws_thread.is_alive():
            if not getattr(self, '_is_reconnecting', False):
                self._is_reconnecting = True
                print("⚠️ AssemblyAI connection dropped. Auto-reconnecting...")
                def _reconnect():
                    try:
                        self.start()
                    except Exception as e:
                        print(f"Reconnect failed: {e}")
                    finally:
                        self._is_reconnecting = False
                import threading
                threading.Thread(target=_reconnect, daemon=True).start()

        # If not active yet, buffer the audio
        if not self.is_active:
            with self.buffer_lock:
                self.early_audio_buffer.append(pcm_bytes)
            # Log periodically to avoid spam
            if len(self.early_audio_buffer) % 50 == 1:
                print(f"📦 Buffered audio (connection not active yet - {len(self.early_audio_buffer)} chunks)")
            return
        
        # If not ready yet, buffer the audio
        if not self.is_ready:
            with self.buffer_lock:
                self.early_audio_buffer.append(pcm_bytes)
            # Log periodically to avoid spam
            if len(self.early_audio_buffer) % 50 == 1:
                print(f"📦 Buffered audio (session not ready yet - {len(self.early_audio_buffer)} chunks)")
            return

        # Session is ready - send audio directly
        try:
            self.ws.send(pcm_bytes, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(f"❌ Failed to send audio: {e}")
            # If send fails, buffer it for later retry
            with self.buffer_lock:
                self.early_audio_buffer.append(pcm_bytes)

    # ---------------------------
    # FLUSH BUFFERED AUDIO
    # ---------------------------
    def flush_buffer(self):
        """Send all buffered audio chunks"""
        with self.buffer_lock:
            if not self.early_audio_buffer:
                return
            
            buffer_count = len(self.early_audio_buffer)
            print(f"📤 Flushing {buffer_count} buffered audio chunks...")
            
            # Calculate total buffered audio duration
            total_bytes = sum(len(chunk) for chunk in self.early_audio_buffer)
            duration_ms = (total_bytes / (16000 * 2)) * 1000  # 16kHz, 16-bit mono
            
            for i, chunk in enumerate(self.early_audio_buffer):
                try:
                    self.ws.send(chunk, websocket.ABNF.OPCODE_BINARY)
                    if i % 5 == 0:  # Log every 5 chunks
                        print(f"   Sending buffered chunk {i+1}/{buffer_count}")
                except Exception as e:
                    print(f"❌ Failed to send buffered chunk {i+1}: {e}")
            
            self.early_audio_buffer.clear()
            print(f"✅ Buffer flushed ({duration_ms:.0f}ms of audio)")

    # ---------------------------
    # PRIME THE SYSTEM WITH SILENCE
    # ---------------------------
    def prime_system(self):
        """Send 300ms of silence to calibrate speech recognition"""
        if not self.is_active or not self.is_ready:
            print("⚠️ Cannot prime system: not active or not ready")
            return
        
        print("🔊 Priming speech recognition with 300ms silence...")
        
        # Generate 300ms of silence (16kHz, 16-bit, mono)
        silence_samples = int(16000 * 0.3)  # 300ms at 16kHz
        silence = np.zeros(silence_samples, dtype=np.int16).tobytes()
        
        # Send priming silence
        try:
            self.ws.send(silence, websocket.ABNF.OPCODE_BINARY)
            print("✅ System primed - ready for speech")
        except Exception as e:
            print(f"❌ Failed to prime system: {e}")

    # ---------------------------
    # STOP STREAM
    # ---------------------------
    def stop(self):
        if not self.is_active:
            return

        try:
            # Clear buffer first
            with self.buffer_lock:
                buffer_count = len(self.early_audio_buffer)
                if buffer_count > 0:
                    print(f"⚠️ Discarding {buffer_count} buffered audio chunks on stop")
                    self.early_audio_buffer.clear()
            
            # Send termination message (matches official docs)
            terminate_message = {"type": "Terminate"}
            self.ws.send(json.dumps(terminate_message))
            time.sleep(0.2)
            self.ws.close()
        except Exception as e:
            print(f"⚠️ Error during stop: {e}")
        finally:
            self.is_active = False
            self.is_ready = False
            print("🛑 AssemblyAI streaming stopped")

    # ---------------------------
    # GET BUFFER STATUS
    # ---------------------------
    def get_buffer_status(self):
        """Get information about buffered audio"""
        with self.buffer_lock:
            buffer_count = len(self.early_audio_buffer)
            total_bytes = sum(len(chunk) for chunk in self.early_audio_buffer)
            duration_ms = (total_bytes / (16000 * 2)) * 1000 if total_bytes > 0 else 0
            return {
                'count': buffer_count,
                'total_bytes': total_bytes,
                'duration_ms': duration_ms,
                'is_active': self.is_active,
                'is_ready': self.is_ready
            }

    # ---------------------------
    # SOCKET EVENTS
    # ---------------------------
    def _on_open(self, ws):
        self.is_active = True
        print("🔗 WebSocket connected to AssemblyAI")
        print("   Waiting for session initialization...")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == "Begin":
                session_id = data.get('id')
                expires_at = data.get('expires_at')
                print(f"🎯 Session began: ID={session_id}")
                
                # 🔥 CRITICAL: NOW we're ready to receive audio!
                self.is_ready = True
                print("✅ AssemblyAI ready for audio input")
                
                # Send priming silence to calibrate
                self.prime_system()
                
                # 🔥 FLUSH ANY BUFFERED AUDIO
                self.flush_buffer()
                
                # 🔥 CALL ON_READY CALLBACK (THIS IS THE RIGHT PLACE)
                if self.on_ready:
                    self.on_ready()
                
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
                print(f"🏁 Session Terminated: Audio={audio_duration:.1f}s, Session={session_duration:.1f}s")
                self.is_ready = False

        except Exception as e:
            print(f"❌ Message handling error: {e}")

    def _on_error(self, ws, error):
        self.is_active = False
        self.is_ready = False
        print(f"❌ AssemblyAI WebSocket error: {error}")
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, code, reason):
        self.is_active = False
        self.is_ready = False
        print(f"🔌 AssemblyAI WebSocket closed (code={code}): {reason}")


def warmup_assemblyai():
    """
    Simple warmup - verify API key and create a short test connection
    This prevents cold-start delays on first real interview
    """
    print("🔥 Warming up AssemblyAI connection...")
    
    def noop_partial(text):
        pass
    
    def noop_final(text):
        pass
    
    def noop_error(error):
        print(f"Warmup error: {error}")
    
    def on_warmup_ready():
        print("✅ AssemblyAI warmup connection ready")
    
    try:
        # Quick test connection
        test_streamer = AssemblyAIWebSocketStreamer(
            on_partial=noop_partial,
            on_final=noop_final,
            on_error=noop_error,
            on_ready=on_warmup_ready
        )
        
        test_streamer.start()
        
        # Wait for connection to establish and session to begin
        timeout = 5
        start = time.time()
        while not test_streamer.is_ready and time.time() - start < timeout:
            time.sleep(0.1)
        
        if test_streamer.is_ready:
            # Send minimal test audio (100ms of silence)
            silence_samples = int(16000 * 0.1)  # 100ms
            silence = np.zeros(silence_samples, dtype=np.int16).tobytes()
            test_streamer.send_audio(silence)
            
            # Wait a bit for processing
            time.sleep(0.5)
        
        # Cleanly stop
        test_streamer.stop()
        
        print("✅ AssemblyAI warmup complete")
        return True
        
    except Exception as e:
        print(f"⚠️ AssemblyAI warmup failed: {e}")
        return False


def test_assemblyai_connection():
    """
    Test function to verify AssemblyAI connection works
    """
    print("🧪 Testing AssemblyAI connection...")
    
    transcript_received = False
    
    def on_partial(text):
        print(f"Partial: {text}")
    
    def on_final(text):
        nonlocal transcript_received
        print(f"Final: {text}")
        transcript_received = True
    
    def on_error(error):
        print(f"Error: {error}")
    
    def on_ready():
        print("✅ AssemblyAI ready!")
    
    try:
        streamer = AssemblyAIWebSocketStreamer(
            on_partial=on_partial,
            on_final=on_final,
            on_error=on_error,
            on_ready=on_ready
        )
        
        streamer.start()
        
        # Wait for readiness
        timeout = 8
        start = time.time()
        while not streamer.is_ready and time.time() - start < timeout:
            time.sleep(0.1)
        
        if not streamer.is_ready:
            print("❌ AssemblyAI did not become ready in time")
            streamer.stop()
            return False
        
        # Send test audio (500ms of silence + test tone)
        print("🎵 Sending test audio...")
        
        # 500ms silence
        silence = np.zeros(int(16000 * 0.5), dtype=np.int16).tobytes()
        streamer.send_audio(silence)
        
        # 200ms test tone (440Hz sine wave)
        t = np.linspace(0, 0.2, int(16000 * 0.2))
        test_tone = (np.sin(2 * np.pi * 440 * t) * 3000).astype(np.int16).tobytes()
        streamer.send_audio(test_tone)
        
        # Wait for processing
        time.sleep(1.0)
        
        # Check buffer status
        buffer_status = streamer.get_buffer_status()
        print(f"📊 Buffer status: {buffer_status}")
        
        streamer.stop()
        
        if transcript_received:
            print("✅ AssemblyAI test PASSED - transcription working")
            return True
        else:
            print("⚠️ AssemblyAI test: No transcription received (may be normal for silence)")
            return True
            
    except Exception as e:
        print(f"❌ AssemblyAI test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("=" * 50)
    print("AssemblyAI WebSocket Streamer Test")
    print("=" * 50)
    
    # First warm up the connection
    warmup_assemblyai()
    
    # Wait a moment
    time.sleep(1)
    
    # Run a full test
    test_assemblyai_connection()