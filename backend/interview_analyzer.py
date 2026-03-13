import librosa
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import spacy
from faster_whisper import WhisperModel
import re
import time
import threading
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import concurrent.futures
from functools import partial

# Load models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: monotonic timestamp in seconds
def now_ts():
    return time.monotonic()

class RunningStatistics:
    """
    RESEARCH-GRADE SPEECH METRICS
    CORRECT LOGIC:
    - speaking_time = actual voice activity
    - silence_time = pauses BETWEEN speech segments (thinking time)
    - forced_silence_time = 15s waits AFTER answer complete
    """
    def __init__(self, pause_threshold=0.3, long_pause_threshold=5.0, ignore_long_pause_over=20.0):
        self.lock = threading.RLock()
        
        # =====================================
        # RESEARCH-GRADE CLOCKS
        # =====================================
        self.session_start_time = now_ts()
        self.session_end_time = None
        
        # Speech event tracking
        self.current_speech_start = None
        self.last_speech_end = None
        self.first_voice_recorded = False
        self.question_end_time = None
        self.response_latencies = []
        
        # =====================================
        # CORRECT AGGREGATES
        # =====================================
        self.total_speaking_time = 0.0      # Actual voice activity
        self.total_silence_time = 0.0        # Pauses BETWEEN speech segments
        self.forced_silence_time = 0.0       # 15s waits AFTER answer complete
        self.total_session_duration = 0.0    # Total wall-clock time
        self.effective_duration = 0.0        # total - forced_silence
        
        self.total_words = 0
        
        # =====================================
        # PAUSE TRACKING
        # =====================================
        self.pause_threshold = float(pause_threshold)
        self.long_pause_threshold = float(long_pause_threshold)
        self.ignore_long_pause_over = float(ignore_long_pause_over)
        self.pause_durations = []
        self.long_pause_count = 0
        
        # =====================================
        # TRANSCRIPT & QA TRACKING
        # =====================================
        self.transcript_parts = []
        self.conversation = []
        self.full_transcript = []
        self.question_scores = []
        self.total_semantic_score = 0.0
        self.total_keyword_score = 0.0
        self.question_count = 0
        
        # Legacy fields (keep for compatibility)
        self.pitch_count = 0
        self.pitch_mean = 0.0
        self.pitch_m2 = 0.0
        self.pitch_min = float('inf')
        self.pitch_max = float('-inf')
        self.filler_counts = {}
        self.jitter_values = []
        self.shimmer_values = []
        self.hnr_values = []
        
        # Diagnostics
        self.event_log = []  # Keep last 20 events for debugging
    
    # =====================================
    # SPEECH EVENT HANDLERS (THREAD-SAFE)
    # =====================================
    def update_pitch(self, audio_chunk, sr=16000):
        """
        YIN pitch extraction + Welford streaming variance.
        This implements the DSP section described in the research paper.
        """
        try:
            # Convert raw audio to numpy float
            audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            if len(audio) < 512:
                return

            # Normalize
            audio = audio / 32768.0

            # YIN pitch estimation
            f0 = librosa.yin(
                audio,
                fmin=80,
                fmax=400,
                sr=sr
            )

            # Remove invalid pitch values
            f0 = f0[f0 > 0]

            if len(f0) == 0:
                return

            for pitch in f0:

                self.pitch_count += 1

                # Welford streaming mean + variance
                delta = pitch - self.pitch_mean
                self.pitch_mean += delta / self.pitch_count
                delta2 = pitch - self.pitch_mean
                self.pitch_m2 += delta * delta2

                # Track range
                self.pitch_min = min(self.pitch_min, pitch)
                self.pitch_max = max(self.pitch_max, pitch)

        except Exception as e:
            print(f"Pitch analysis error: {e}")

    def get_pitch_stability(self):
        """
        Calculate pitch stability using coefficient of variation.
        Matches the equation described in the research paper.
        """
        if self.pitch_count < 2:
            return 0

        variance = self.pitch_m2 / (self.pitch_count - 1)
        std = np.sqrt(variance)

        if self.pitch_mean == 0:
            return 0

        cv = std / self.pitch_mean

        # Convert to score (0-100)
        score = max(0, 100 - (cv * 100))

        return score
    
    def record_question_end(self, ts=None):
        """Called when interviewer finishes asking question"""
        with self.lock:
            ts = now_ts() if ts is None else ts
            self.question_end_time = ts
            self._log_event("question_end", ts)
            print(f"⏱️ Question end recorded at {ts:.1f}s")
    
    def record_speech_start(self, ts=None):
        """
        Called when user starts speaking
        Detects pause BEFORE starting new speech (thinking time)
        """
        with self.lock:
            ts = now_ts() if ts is None else ts
            
            # Detect pause before starting new speech
            if self.last_speech_end is not None:
                pause_duration = ts - self.last_speech_end
                if pause_duration > self.pause_threshold:
                    # Only count reasonable pauses (< ignore_long_pause_over)
                    if pause_duration < self.ignore_long_pause_over:
                        self.pause_durations.append(pause_duration)
                        self.total_silence_time += pause_duration
                        self._log_event("pause", pause_duration, ts)
                        print(f"⏸️ Silence segment: +{pause_duration:.2f}s (total silence: {self.total_silence_time:.1f}s)")
                        
                        if pause_duration > self.long_pause_threshold:
                            self.long_pause_count += 1
                            print(f"   → Long pause detected ({pause_duration:.1f}s)")
                    else:
                        self._log_event("ignored_pause", pause_duration, ts)
                        print(f"⏸️ Ignoring pause >{self.ignore_long_pause_over:.0f}s: {pause_duration:.1f}s (likely system wait)")
            
            # Start new speech segment if not already started
            if self.current_speech_start is None:
                self.current_speech_start = ts
                self._log_event("speech_start", ts)
                print(f"🎤 Speech START at {ts:.2f}s")
                
                # Record first voice for latency tracking
                if not self.first_voice_recorded and self.question_end_time is not None:
                    latency = ts - self.question_end_time
                    self.response_latencies.append(latency)
                    self.first_voice_recorded = True
                    self._log_event("response_latency", latency, ts)
                    print(f"⏱️ Response latency: {latency:.2f}s")
    
    def record_speech_end(self, ts=None):
        """Called when user stops speaking"""
        with self.lock:
            ts = now_ts() if ts is None else ts
            
            if self.current_speech_start is None:
                return
            
            # Calculate speech duration for this segment
            speech_duration = ts - self.current_speech_start
            
            # 🔥 SANITY CHECK: If speaking time is too long for word count, cap it
            expected_speaking_time = self.total_words * 0.3  # 300ms per word estimate
            if expected_speaking_time > 0 and speech_duration > expected_speaking_time * 2:
                print(f"⚠️ Unusual speaking time: {speech_duration:.1f}s for {self.total_words} words")
                speech_duration = min(speech_duration, expected_speaking_time * 1.5)
            
            self.total_speaking_time += speech_duration
            self.last_speech_end = ts
            self.current_speech_start = None
            self._log_event("speech_end", speech_duration, ts)
            print(f"🎤 Speech END: +{speech_duration:.2f}s (total speaking: {self.total_speaking_time:.1f}s)")
    
    def record_forced_silence(self, duration):
        """
        Called when silence watcher hits 15s timeout
        This 15s is NOT counted in speaking or silence time
        It's tracked separately as system waiting time
        """
        with self.lock:
            self.forced_silence_time += float(duration)
            self._log_event("forced_silence", float(duration), now_ts())
            print(f"⏰ Forced silence (system wait): +{duration:.1f}s (total forced: {self.forced_silence_time:.1f}s)")
    
    # =====================================
    # TRANSCRIPT & QA TRACKING
    # =====================================
    
    def update_transcript(self, new_text):
        if new_text:
            with self.lock:
                self.transcript_parts.append(new_text)
                self.total_words += len(new_text.split())
    
    def record_question(self, question):
        with self.lock:
            self.conversation.append({'role': 'interviewer', 'text': question})
            self.full_transcript.append(f"Interviewer: {question}")
    
    def record_qa_pair(self, question, answer, expected_answer, similarity_score, keyword_score):
        with self.lock:
            self.question_scores.append({
                'question': question,
                'answer': answer,
                'expected_answer': expected_answer,
                'similarity': similarity_score,
                'keyword_coverage': keyword_score
            })
            self.total_semantic_score += similarity_score
            self.total_keyword_score += keyword_score
            self.question_count += 1
            
            self.conversation.append({'role': 'user', 'text': answer})
            self.full_transcript.append(f"User: {answer}")
    
    # =====================================
    # FINAL METRICS COMPUTATION
    # =====================================
    
    def finalize_session_metrics(self):
        """Compute final, mathematically valid metrics"""
        with self.lock:
            self.session_end_time = now_ts()
            
            if self.session_start_time:
                self.total_session_duration = max(0.0, self.session_end_time - self.session_start_time)
            
            # If a speech segment is still open, close it now
            if self.current_speech_start is not None:
                seg = now_ts() - self.current_speech_start
                self.total_speaking_time += seg
                self.last_speech_end = now_ts()
                self.current_speech_start = None
                self._log_event("auto_close_final", seg)
                print(f"🎤 Auto-closed final speech segment: +{seg:.2f}s")
            
            # Effective duration = total - forced silence (system waits)
            self.effective_duration = max(0.0, self.total_session_duration - self.forced_silence_time)
            
            # Speaking ratio = speaking_time / effective_duration
            if self.effective_duration > 0:
                self.speaking_ratio = self.total_speaking_time / self.effective_duration
            else:
                self.speaking_ratio = 0.0
    
    def compute_research_metrics(self):
        """Return research-grade metrics dictionary"""
        with self.lock:
            self.finalize_session_metrics()
            
            # Words Per Minute (using speaking time only)
            if self.total_speaking_time > 5.0:
                speaking_minutes = self.total_speaking_time / 60
                wpm = self.total_words / speaking_minutes
            else:
                wpm = 0.0
            
            # Pause statistics
            avg_pause = float(np.mean(self.pause_durations)) if self.pause_durations else 0.0
            
            # Hesitation rate (pauses per minute of speaking)
            if speaking_minutes > 0:
                hesitation_rate = len(self.pause_durations) / speaking_minutes
            else:
                hesitation_rate = 0.0
            
            # Articulation rate (words per second of actual speaking)
            articulation_rate = self.total_words / max(0.1, self.total_speaking_time)
            
            # 🔥 CORRECT: Calculate silence_time = effective_duration - speaking_time
            silence_time = max(0.0, self.effective_duration - self.total_speaking_time)
            
            # Fluency score (0-1, higher is more fluent)
            if self.effective_duration > 0:
                fluency_score = self.total_speaking_time / self.effective_duration
            else:
                fluency_score = 0.0
            
            # Average response latency
            avg_response_latency = float(np.mean(self.response_latencies)) if self.response_latencies else 0.0
            
            # Average Q&A scores
            avg_semantic = self.total_semantic_score / self.question_count if self.question_count > 0 else 0.0
            avg_keyword = self.total_keyword_score / self.question_count if self.question_count > 0 else 0.0
            
            return {
                "session_duration": round(self.total_session_duration, 1),
                "effective_duration": round(self.effective_duration, 1),
                "speaking_time": round(self.total_speaking_time, 1),
                "silence_time": round(silence_time, 1),
                "forced_silence_time": round(self.forced_silence_time, 1),
                "speaking_ratio": round(self.speaking_ratio, 3),
                "wpm": round(wpm, 1),
                "total_words": self.total_words,
                
                # ✅ FIXED: These variables ARE defined in this function
                "avg_pause_duration": round(avg_pause, 2),
                "pause_count": len(self.pause_durations),
                "long_pause_count": self.long_pause_count,
                "hesitation_rate": round(hesitation_rate, 2),
                "articulation_rate": round(articulation_rate, 2),
                "fluency_score": round(fluency_score, 3),
                "avg_response_latency": round(avg_response_latency, 2),
                "avg_semantic_similarity": round(avg_semantic, 3),
                "avg_keyword_coverage": round(avg_keyword, 3),
                "questions_answered": self.question_count,
                "event_log": self.event_log[-20:],
                
                # ✅ ADD THESE - they ARE available from self
                "pitch_mean": round(self.pitch_mean, 2),
                "pitch_std": round(np.sqrt(self.pitch_m2 / (self.pitch_count - 1)) if self.pitch_count > 1 else 0, 2),
                "pitch_range": round(self.pitch_max - self.pitch_min if self.pitch_min != float('inf') else 0, 2),
                "pitch_stability": round(self.get_pitch_stability(), 2)
            }
    
    def _log_event(self, event_type, *args):
        """Internal method to log events for debugging"""
        self.event_log.append((event_type, now_ts(), args))
        if len(self.event_log) > 100:
            self.event_log = self.event_log[-100:]
    
    # =====================================
    # LEGACY METHODS (KEEP FOR COMPATIBILITY)
    # =====================================
    
    def get_avg_response_latency(self):
        return float(np.mean(self.response_latencies)) if self.response_latencies else 0.0
    
    def get_current_stats(self):
        return self.compute_research_metrics()
    
    def update_pitch_stats(self, pitch_values):
        for pitch in pitch_values:
            if np.isfinite(pitch):
                self.pitch_count += 1
                delta = pitch - self.pitch_mean
                self.pitch_mean += delta / self.pitch_count
                delta2 = pitch - self.pitch_mean
                self.pitch_m2 += delta * delta2
                self.pitch_min = min(self.pitch_min, pitch)
                self.pitch_max = max(self.pitch_max, pitch)
    
    def update_filler_stats(self, filler_counts):
        for filler, count in filler_counts.items():
            self.filler_counts[filler] = self.filler_counts.get(filler, 0) + count
    
    def update_voice_quality(self, jitter, shimmer, hnr):
        if np.isfinite(jitter):
            self.jitter_values.append(jitter)
        if np.isfinite(shimmer):
            self.shimmer_values.append(shimmer)
        if np.isfinite(hnr):
            self.hnr_values.append(hnr)


class VoiceActivityDetector:
    """
    Lightweight energy-based VAD suitable for streaming audio frames.
    - feed raw PCM float32 arrays (mono) at sample_rate (e.g. 16000)
    - callbacks: on_voice_start(), on_voice_end()
    - hangover logic to prevent fragmentation.
    """
    def __init__(self, sample_rate=16000, frame_ms=30, energy_threshold=0.01,
                 hangover_ms=200, min_speech_ms=120):
        self.sr = int(sample_rate)
        self.frame_ms = frame_ms
        self.frame_samples = max(1, int(self.sr * (frame_ms / 1000.0)))
        self.energy_threshold = float(energy_threshold)
        self.hangover_ms = int(hangover_ms)
        self.min_speech_ms = int(min_speech_ms)
        
        self._speech_state = False
        self._hangover_remaining = 0.0  # seconds
        self._since_speech_start = 0.0
        self._since_speech_end = 0.0
        
        # Callbacks (assign externally)
        self.on_voice_start = None
        self.on_voice_end = None
        
        print(f"🔊 VAD initialized: frame={frame_ms}ms, threshold={energy_threshold}, hangover={hangover_ms}ms")
    
    def _rms(self, frame):
        """Calculate RMS energy of frame (float32 [-1..1])"""
        if frame is None or len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    
    def process_frame(self, frame, timestamp=None):
        """
        Process one frame (numpy array, float32). Provide a timestamp if available.
        """
        ts = now_ts() if timestamp is None else timestamp
        
        if len(frame) == 0:
            return
            
        rms = self._rms(frame)
        frame_duration = len(frame) / self.sr
        
        # Voice detection
        is_voice = (rms >= self.energy_threshold)
        
        if is_voice:
            self._since_speech_start += frame_duration
            self._since_speech_end = 0.0
            
            # Enter speech state if not already and minimum speech seen
            if not self._speech_state:
                if self._since_speech_start * 1000.0 >= self.min_speech_ms:
                    self._speech_state = True
                    self._hangover_remaining = self.hangover_ms / 1000.0
                    if callable(self.on_voice_start):
                        self.on_voice_start(ts)
            else:
                # Already speaking; refresh hangover
                self._hangover_remaining = self.hangover_ms / 1000.0
        else:
            # Silent frame
            self._since_speech_start = 0.0
            
            if self._speech_state:
                # Decrement hangover
                self._hangover_remaining -= frame_duration
                if self._hangover_remaining <= 0:
                    # Commit speech end
                    self._speech_state = False
                    if callable(self.on_voice_end):
                        self.on_voice_end(ts)
            else:
                self._since_speech_end += frame_duration
    
    def reset(self):
        """Reset VAD state"""
        self._speech_state = False
        self._hangover_remaining = 0.0
        self._since_speech_start = 0.0
        self._since_speech_end = 0.0


# =====================================
# EXISTING FUNCTIONS (KEEP AS-IS)
# =====================================

def analyze_audio_chunk_fast(pcm_chunk, sample_rate, stats: RunningStatistics, prev_overlap=None):
    if stats is None or len(pcm_chunk) < 512:
        return
        
    # Convert int16 PCM → float32
    audio = pcm_chunk.astype(np.float32) / 32768.0
    
    # If we have previous overlap, prepend it for continuity
    if prev_overlap is not None and len(prev_overlap) > 0:
        audio = np.concatenate([prev_overlap, audio])
    
    try:
        # YIN pitch estimation
        f0 = librosa.yin(
            audio, 
            sr=sample_rate,
            fmin=65,
            fmax=400,
            frame_length=1024,
            hop_length=512
        )
        
        voiced_f0 = f0[np.isfinite(f0)]
        if len(voiced_f0) > 0:
            stats.update_pitch_stats(voiced_f0)
            
            # Voice Quality (Jitter/Shimmer)
            jitter = np.std(voiced_f0) / np.mean(voiced_f0) if np.mean(voiced_f0) > 0 else 0
            rms = np.sqrt(np.mean(audio**2))
            shimmer = np.std(audio) / (rms + 1e-6)
            
            stats.update_voice_quality(jitter, shimmer, 10.0)
        
        # Return last 100ms of audio for next chunk's overlap
        overlap_samples = int(sample_rate * 0.1)  # 100ms overlap
        return audio[-overlap_samples:] if len(audio) > overlap_samples else audio
        
    except Exception as e:
        print(f"DSP Error: {e}")
        return None


def detect_fillers_repetitions(text):
    fillers = ["um", "uh", "like", "you know", "i mean"]
    filler_count = 0
    repetitions = 0
    words = text.lower().split()
    
    for i, word in enumerate(words):
        if word in fillers:
            filler_count += 1
        if i > 0 and words[i] == words[i-1]:
            repetitions += 1
            
    return filler_count, repetitions


def calculate_semantic_similarity(answer, expected_answer):
    """
    Calculate TRUE semantic similarity between answer and expected answer.
    Returns raw cosine similarity (0.0 to 1.0) - NO ARTIFICIAL SCALING.
    """
    if not answer or not answer.strip():
        print(f"📊 Empty answer detected, similarity = 0.0")
        return 0.0
    
    if not expected_answer or not expected_answer.strip():
        print(f"📊 Empty expected answer, similarity = 0.0")
        return 0.0
    
    try:
        answer_emb = embedder.encode([answer], normalize_embeddings=True)[0]
        expected_emb = embedder.encode([expected_answer], normalize_embeddings=True)[0]
        
        similarity = cosine_similarity([answer_emb], [expected_emb])[0][0]
        similarity = max(0.0, float(similarity))
        
        print(f"📊 TRUE Semantic similarity: {similarity:.3f}")
        return similarity
        
    except Exception as e:
        print(f"❌ Semantic similarity error: {e}")
        return 0.0


def calculate_keyword_coverage(answer, question):
    """
    Calculate how many keywords from the question appear in the answer.
    Returns RAW coverage (0.0 to 1.0) - NO SCALING.
    """
    if not answer or not question:
        return 0.0
    
    import re
    
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                  'that', 'these', 'those', 'there', 'here', 'what', 'which', 'who',
                  'whom', 'whose', 'why', 'how', 'between', 'difference'}
    
    tech_keywords = {
        'dbms': ['database', 'sql', 'query', 'index', 'transaction', 'acid', 'normalization', 
                'join', 'primary key', 'foreign key', 'schema', 'table', 'bcnf', '3nf'],
        'os': ['process', 'thread', 'memory', 'deadlock', 'scheduling', 'virtual memory',
              'kernel', 'system call', 'context switch', 'semaphore', 'mutex', 'paging',
              'segmentation', 'fifo', 'lru', 'race condition', 'critical section'],
        'oops': ['class', 'object', 'inheritance', 'polymorphism', 'encapsulation', 
                'abstraction', 'interface', 'method', 'constructor', 'destructor',
                'virtual function', 'abstract class', 'multiple inheritance', 'composition']
    }
    
    question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    question_words = {w for w in question_words if w not in STOP_WORDS}
    
    for topic, keywords in tech_keywords.items():
        for kw in keywords:
            if kw in question.lower():
                if ' ' in kw:
                    question_words.add(kw)
                else:
                    question_words.add(kw)
    
    answer_lower = answer.lower()
    matches = 0
    matched_keywords = []
    
    for word in question_words:
        if ' ' in word:
            if word in answer_lower:
                matches += 1
                matched_keywords.append(word)
        else:
            if re.search(r'\b' + re.escape(word) + r'\b', answer_lower):
                matches += 1
                matched_keywords.append(word)
    
    if question_words:
        coverage = min(1.0, matches / len(question_words))
    else:
        coverage = 0.0
    
    print(f"🔑 RAW Keyword coverage: {matches}/{len(question_words)} = {coverage:.3f}")
    print(f"   Matched: {matched_keywords}")
    
    return coverage


def analyze_pitch_comprehensive(audio_path):
    """Comprehensive pitch analysis"""
    y, sr = librosa.load(audio_path)
    f0 = librosa.yin(y, sr=sr, fmin=65, fmax=300, frame_length=512, hop_length=128)
    f0_voiced = f0[np.isfinite(f0)]
    
    if len(f0_voiced) < 10:
        return {
            "pitch_mean": 0, "pitch_std": 0, "pitch_min": 0, "pitch_max": 0,
            "pitch_range": 0, "pitch_stability": 0, "pitch_score": 0,
            "pitch_feedback": "Insufficient voiced audio for pitch analysis"
        }
    
    pitch_mean = np.mean(f0_voiced)
    pitch_std = np.std(f0_voiced)
    pitch_min = np.min(f0_voiced)
    pitch_max = np.max(f0_voiced)
    pitch_range = pitch_max - pitch_min
    pitch_stability = pitch_std / pitch_mean if pitch_mean > 0 else 1.0
    
    stability_score = max(0, min(100, 100 * (1 - pitch_stability / 0.5)))
    
    if 50 < pitch_range < 300:
        range_score = 100
    elif 20 < pitch_range < 400:
        range_score = 70
    else:
        range_score = 30
    
    if 100 < pitch_mean < 250:
        mean_score = 100
    elif 80 < pitch_mean < 300:
        mean_score = 80
    else:
        mean_score = 40
    
    pitch_score = (stability_score * 0.5 + range_score * 0.3 + mean_score * 0.2)
    
    feedback_parts = []
    if pitch_stability > 0.4:
        feedback_parts.append("pitch varies too much")
    elif pitch_stability < 0.2:
        feedback_parts.append("pitch is very stable")
    
    pitch_feedback = "Pitch analysis: " + "; ".join(feedback_parts) if feedback_parts else "Pitch is well-modulated"
    
    return {
        "pitch_mean": float(pitch_mean),
        "pitch_std": float(pitch_std),
        "pitch_min": float(pitch_min),
        "pitch_max": float(pitch_max),
        "pitch_range": float(pitch_range),
        "pitch_stability": float(pitch_stability),
        "pitch_score": float(pitch_score),
        "pitch_feedback": pitch_feedback
    }


def analyze_voice_quality(audio_path):
    """Analyze voice quality metrics"""
    y, sr = librosa.load(audio_path)
    f0 = librosa.yin(y, sr=sr, fmin=65, fmax=300, frame_length=512, hop_length=128)
    f0_voiced = f0[np.isfinite(f0)]
    
    if len(f0_voiced) < 20:
        return {
            "jitter": 0, "shimmer": 0, "hnr": 0,
            "voice_quality_score": 0,
            "voice_quality_feedback": "Insufficient voiced audio for voice quality analysis"
        }
    
    if len(f0_voiced) > 1:
        f0_diffs = np.abs(np.diff(f0_voiced))
        jitter = np.mean(f0_diffs) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0
    else:
        jitter = 0
    
    frame_length = int(sr * 0.02)
    hop_length = int(sr * 0.01)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(rms) > 1:
        rms_diffs = np.abs(np.diff(rms))
        shimmer = np.mean(rms_diffs) / np.mean(rms) if np.mean(rms) > 0 else 0
    else:
        shimmer = 0
    
    def calculate_hnr(signal):
        if len(signal) < 100:
            return 0
        signal = signal - np.mean(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peak_idx = np.argmax(autocorr[:len(autocorr)//4])
        if peak_idx < len(autocorr) - 10:
            search_start = peak_idx + 5
            search_end = min(search_start + 50, len(autocorr))
            noise_floor = np.min(autocorr[search_start:search_end])
            hnr = 10 * np.log10(autocorr[peak_idx] / noise_floor) if noise_floor > 0 else 0
        else:
            hnr = 0
        return max(0, hnr)
    
    hnr_values = []
    segment_length = int(sr * 0.1)
    for i in range(0, len(y) - segment_length, segment_length // 2):
        segment = y[i:i + segment_length]
        hnr_val = calculate_hnr(segment)
        if hnr_val > 0:
            hnr_values.append(hnr_val)
    
    hnr = np.mean(hnr_values) if hnr_values else 0
    
    jitter_score = max(0, min(100, 100 * (1 - jitter / 0.02)))
    shimmer_score = max(0, min(100, 100 * (1 - shimmer / 0.1)))
    hnr_score = max(0, min(100, hnr * 6.67))
    voice_quality_score = (jitter_score * 0.3 + shimmer_score * 0.3 + hnr_score * 0.4)
    
    feedback_parts = []
    if jitter > 0.015:
        feedback_parts.append("pitch perturbations detected")
    if shimmer > 0.08:
        feedback_parts.append("amplitude variations detected")
    if hnr < 10:
        feedback_parts.append("signal has significant background noise")
    
    voice_quality_feedback = "Voice quality: " + "; ".join(feedback_parts) if feedback_parts else "Voice quality is clear and stable"
    
    return {
        "jitter": float(jitter),
        "shimmer": float(shimmer),
        "hnr": float(hnr),
        "jitter_score": float(jitter_score),
        "shimmer_score": float(shimmer_score),
        "hnr_score": float(hnr_score),
        "voice_quality_score": float(voice_quality_score),
        "voice_quality_feedback": voice_quality_feedback
    }


def get_voiced_segments(audio_path, chunk_duration=5.0, overlap=0.5):
    """Extract voiced segments from audio"""
    y, sr = librosa.load(audio_path, sr=8000)
    intervals = librosa.effects.split(y, top_db=20)
    voiced_segments = []
    for start_sample, end_sample in intervals:
        start_time = start_sample / sr
        end_time = end_sample / sr
        duration = end_time - start_time
        if duration >= 0.5:
            voiced_segments.append((start_time, end_time))
    return voiced_segments


def transcribe_voiced_segments(audio_path, model_name="medium.en"):
    """Transcribe only the voiced segments"""
    try:
        y_full, sr_full = librosa.load(audio_path, sr=16000)
        voiced_segments = get_voiced_segments(audio_path)
        if not voiced_segments:
            return "", []
        full_transcript = ""
        segment_details = []
        model = model_manager.get_model(model_name)
        for start_time, end_time in voiced_segments:
            start_sample = int(start_time * sr_full)
            end_sample = int(end_time * sr_full)
            segment_audio = y_full[start_sample:end_sample]
            if len(segment_audio) < sr_full:
                continue
            segments, info = model.transcribe(segment_audio, language="en", beam_size=3, vad_filter=False)
            segment_text = " ".join([seg.text for seg in segments]).strip()
            if segment_text:
                full_transcript += segment_text + " "
                segment_details.append({
                    'start': start_time,
                    'end': end_time,
                    'text': segment_text,
                    'duration': end_time - start_time
                })
        return full_transcript.strip(), segment_details
    except Exception as e:
        print(f"Error in voiced segment transcription: {e}")
        return speech_to_text(audio_path, model_name), []


def parallel_analyze_segment(segment_data, model_name="medium.en"):
    """Analyze a single segment in parallel"""
    start_time, end_time, segment_audio, sr_full = segment_data
    results = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'transcript': '',
        'word_count': 0,
        'filler_count': 0
    }
    try:
        model = model_manager.get_model(model_name)
        segments, _ = model.transcribe(segment_audio, language="en", beam_size=3, vad_filter=False)
        transcript = " ".join([seg.text for seg in segments]).strip()
        results['transcript'] = transcript
        results['word_count'] = len(transcript.split()) if transcript else 0
        if transcript:
            fillers = ["um", "uh", "like", "you know", "i mean"]
            filler_count = sum(transcript.lower().count(filler) for filler in fillers)
            results['filler_count'] = filler_count
    except Exception as e:
        print(f"Error analyzing segment {start_time}-{end_time}: {e}")
    return results


def parallel_pitch_analysis(audio_path):
    """Parallel pitch analysis function"""
    try:
        return analyze_pitch_comprehensive(audio_path)
    except Exception as e:
        print(f"Parallel pitch analysis failed: {e}")
        return {
            "pitch_mean": 0, "pitch_std": 0, "pitch_min": 0, "pitch_max": 0,
            "pitch_range": 0, "pitch_stability": 0, "pitch_score": 0,
            "pitch_feedback": "Analysis failed"
        }


def parallel_voice_quality_analysis(audio_path):
    """Parallel voice quality analysis function"""
    try:
        return analyze_voice_quality(audio_path)
    except Exception as e:
        print(f"Parallel voice quality analysis failed: {e}")
        return {
            "jitter": 0, "shimmer": 0, "hnr": 0,
            "voice_quality_score": 0,
            "voice_quality_feedback": "Analysis failed"
        }


def analyze_audio_chunk(audio_chunk, sample_rate, stats):
    """Analyze a single audio chunk"""
    chunk_duration = len(audio_chunk) / sample_rate
    intervals = librosa.effects.split(audio_chunk, top_db=20)
    speaking_time = sum((end - start) / sample_rate for start, end in intervals)
    
    pause_durations = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            current_end = intervals[i][1] / sample_rate
            next_start = intervals[i + 1][0] / sample_rate
            pause_duration = next_start - current_end
            if pause_duration > 0.1:
                pause_durations.append(pause_duration)
    
    if sample_rate != 8000:
        chunk_8k = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=8000)
        sr_8k = 8000
    else:
        chunk_8k = audio_chunk
        sr_8k = sample_rate
    
    try:
        f0 = librosa.yin(chunk_8k, sr=sr_8k, fmin=65, fmax=300, frame_length=512, hop_length=128)
        voiced_f0 = f0[np.isfinite(f0)]
        if len(voiced_f0) > 0:
            stats.update_pitch_stats(voiced_f0)
    except Exception as e:
        print(f"Pitch analysis failed for chunk: {e}")
    
    try:
        if len(voiced_f0) > 10:
            jitter = np.std(voiced_f0) / np.mean(voiced_f0) if np.mean(voiced_f0) > 0 else 0
            rms = librosa.feature.rms(y=chunk_8k, frame_length=256, hop_length=128)[0]
            shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
            hnr = 10
            stats.update_voice_quality(jitter, shimmer, hnr)
    except Exception as e:
        print(f"Voice quality analysis failed for chunk: {e}")
    
    return {
        'chunk_duration': chunk_duration,
        'speaking_time': speaking_time,
        'pause_count': len(pause_durations)
    }


def analyze_interview_response_optimized(audio_path, ideal_answer_text="", ideal_keywords=None, use_large_model=False):
    """Optimized interview response analysis"""
    if ideal_keywords is None:
        ideal_keywords = []
    
    model_name = "large-v3" if use_large_model else "medium.en"
    stats = RunningStatistics()
    
    y_full, sr_full = librosa.load(audio_path, sr=16000)
    voiced_segments = get_voiced_segments(audio_path)
    
    if not voiced_segments:
        return {
            "transcribed_text": "",
            "wpm": 0,
            "pause_ratio": 1.0,
            "filler_count": 0,
            "semantic_similarity": 0,
            "keyword_coverage": 0,
            "pitch_std": 0,
            "fluency_score": 0,
            "clarity_score": 0,
            "overall_score": 0,
            "performance_level": "No Speech Detected",
            "improvement_suggestions": ["Please speak louder and closer to the microphone"]
        }
    
    for start_time, end_time in voiced_segments:
        start_sample = int(start_time * sr_full)
        end_sample = int(end_time * sr_full)
        segment_audio = y_full[start_sample:end_sample]
        chunk_results = analyze_audio_chunk(segment_audio, sr_full, stats)
        
        if len(segment_audio) >= sr_full:
            try:
                model = model_manager.get_model(model_name)
                segments, _ = model.transcribe(segment_audio, language="en", beam_size=3, vad_filter=False)
                segment_text = " ".join([seg.text for seg in segments]).strip()
                if segment_text:
                    stats.update_transcript(segment_text)
                    filler_count, _ = detect_fillers_repetitions(segment_text)
                    stats.update_filler_stats({"total": filler_count})
            except Exception as e:
                print(f"Transcription failed for segment {start_time}-{end_time}: {e}")
    
    final_stats = stats.get_current_stats()
    transcript = final_stats['transcript']
    semantic_similarity = calculate_semantic_similarity(transcript, ideal_answer_text) if transcript else 0
    keyword_coverage = calculate_keyword_coverage(transcript, ideal_keywords) if transcript else 0
    
    fluency_results = {
        'wpm': final_stats['wpm'],
        'pause_ratio': final_stats['pause_ratio'],
        'filler_count': sum(stats.filler_counts.values()),
        'speaking_time': final_stats['speaking_time'],
        'total_duration': final_stats['total_duration']
    }
    
    pitch_results = {
        'pitch_mean': final_stats['pitch_mean'],
        'pitch_std': final_stats['pitch_std'],
        'pitch_min': final_stats['pitch_min'],
        'pitch_max': final_stats['pitch_max'],
        'pitch_range': final_stats['pitch_range']
    }
    
    voice_quality_results = {
        'jitter': final_stats['avg_jitter'],
        'shimmer': final_stats['avg_shimmer'],
        'hnr': final_stats['avg_hnr']
    }
    
    overall_score = compute_overall_score_independent({
        "semantic_similarity": semantic_similarity,
        "keyword_coverage": keyword_coverage,
        "pitch_mean": final_stats["pitch_mean"],
        "pitch_range": final_stats["pitch_range"],
        "speaking_time": final_stats["speaking_time"],
        "total_duration": final_stats["total_duration"]
    })
    
    feedback_data = generate_comprehensive_feedback({
        "overall_score": overall_score,
        "semantic_similarity": semantic_similarity,
        "keyword_coverage": keyword_coverage,
        "pitch_mean": final_stats["pitch_mean"],
        "pitch_range": final_stats["pitch_range"],
        "speaking_time": final_stats["speaking_time"],
        "total_duration": final_stats["total_duration"]
    })
    
    return {
        "transcribed_text": transcript,
        "total_words": final_stats['total_words'],
        "total_duration": final_stats['total_duration'],
        "speaking_time": final_stats['speaking_time'],
        "wpm": fluency_results['wpm'],
        "pause_ratio": fluency_results['pause_ratio'],
        "filler_count": fluency_results['filler_count'],
        "long_pause_count": stats.long_pause_count,
        "pitch_mean": final_stats['pitch_mean'],
        "pitch_std": final_stats['pitch_std'],
        "pitch_range": final_stats['pitch_range'],
        "pitch_min": final_stats['pitch_min'],
        "pitch_max": final_stats['pitch_max'],
        "avg_jitter": final_stats['avg_jitter'],
        "avg_shimmer": final_stats['avg_shimmer'],
        "avg_hnr": final_stats['avg_hnr'],
        "semantic_similarity": semantic_similarity,
        "keyword_coverage": keyword_coverage,
        **feedback_data,
        "processing_method": "optimized_chunked",
        "model_used": model_name,
        "voiced_segments_count": len(voiced_segments)
    }


def speech_to_text(audio_path, model_name="medium.en", use_vad=True, min_speech_duration=1000):
    """Placeholder - actual implementation not needed"""
    pass


def fluency_score(results):
    """Calculate fluency score"""
    wpm_norm = min(1.0, results['wpm'] / 150)
    fluency = 100 * (0.4 * wpm_norm + 
                     0.3 * (1 - results['pause_ratio']) + 
                     0.2 * (1 - min(1.0, results['filler_count'] / 50)) + 
                     0.1 * 0.8)
    return max(0, min(100, fluency))


def clarity_score(results, ideal_answer, ideal_keywords):
    """Calculate clarity score"""
    clarity = 100 * (0.5 * results['semantic_similarity'] + 
                     0.3 * results['keyword_coverage'] + 
                     0.2 * 0.9)
    return max(0, min(100, clarity))


def compute_overall_score_independent(results):
    """Research-safe overall score using ONLY independent metrics"""
    semantic_similarity = results.get("semantic_similarity", 0.0)
    keyword_coverage = results.get("keyword_coverage", 0.0)
    
    semantic_score = semantic_similarity * 100
    keyword_score = keyword_coverage * 100
    
    pitch_mean = results.get("pitch_mean", 0.0)
    pitch_range = results.get("pitch_range", 0.0)
    
    if pitch_mean > 0:
        pitch_stability = 1.0 - min(pitch_range / pitch_mean, 1.0)
    else:
        pitch_stability = 0.0
    
    pitch_score = pitch_stability * 100
    
    speaking_time = results.get("speaking_time", 0.0)
    total_duration = results.get("total_duration", 0.0)
    
    engagement = (speaking_time / total_duration) if total_duration > 0 else 0.0
    engagement_score = engagement * 100
    
    overall_score = (0.40 * semantic_score + 0.30 * keyword_score + 0.15 * pitch_score + 0.15 * engagement_score)
    
    return round(overall_score, 2)


def generate_comprehensive_feedback(results):
    """Generate comprehensive qualitative feedback"""
    overall_score = float(results.get("overall_score", 0.0))
    
    if overall_score >= 80:
        performance_level = "Excellent"
        performance_feedback = "Your response is clear, confident, and technically strong."
    elif overall_score >= 70:
        performance_level = "Good"
        performance_feedback = "Your response is solid, with minor areas for improvement."
    elif overall_score >= 60:
        performance_level = "Fair"
        performance_feedback = "Your response is understandable but needs refinement."
    else:
        performance_level = "Needs Improvement"
        performance_feedback = "Your response needs clearer structure, content focus, and delivery."
    
    detailed_feedback = []
    semantic_similarity = results.get("semantic_similarity", 0.0)
    keyword_coverage = results.get("keyword_coverage", 0.0)
    
    if semantic_similarity < 0.3:
        detailed_feedback.append("The answer does not closely address the question asked.")
    elif semantic_similarity < 0.6:
        detailed_feedback.append("The answer partially addresses the question but lacks depth.")
    
    if keyword_coverage < 0.4:
        detailed_feedback.append("Many expected technical terms are missing.")
    elif keyword_coverage < 0.7:
        detailed_feedback.append("Some important technical terms could be added.")
    
    pitch_mean = results.get("pitch_mean", 0.0)
    pitch_range = results.get("pitch_range", 0.0)
    
    if pitch_mean > 0:
        pitch_variability_ratio = pitch_range / pitch_mean
        if pitch_variability_ratio > 0.6:
            detailed_feedback.append("Pitch varies significantly, which may reduce clarity.")
        elif pitch_variability_ratio < 0.15:
            detailed_feedback.append("Pitch is very flat; adding variation can improve engagement.")
    
    speaking_time = results.get("speaking_time", 0.0)
    total_duration = results.get("total_duration", 0.0)
    
    if total_duration > 0:
        engagement_ratio = speaking_time / total_duration
        if engagement_ratio < 0.5:
            detailed_feedback.append("There are long silent gaps; try to maintain a steadier response.")
    
    improvement_suggestions = []
    
    if semantic_similarity < 0.5:
        improvement_suggestions.append("Focus more directly on answering the question.")
    
    if keyword_coverage < 0.6:
        improvement_suggestions.append("Include more relevant technical keywords.")
    
    if pitch_mean > 0 and pitch_range / pitch_mean > 0.5:
        improvement_suggestions.append("Work on maintaining a more consistent pitch.")
    
    if total_duration > 0 and speaking_time / total_duration < 0.6:
        improvement_suggestions.append("Reduce long pauses to improve flow.")
    
    improvement_suggestions = improvement_suggestions[:5]
    
    return {
        "overall_score": overall_score,
        "performance_level": performance_level,
        "performance_feedback": performance_feedback,
        "detailed_feedback": detailed_feedback,
        "improvement_suggestions": improvement_suggestions,
        "summary": f"Overall Performance: {performance_level} ({overall_score:.1f}/100)"
    }


def analyze_interview_response(audio_file_path, ideal_answer_text, ideal_keywords):
    """Comprehensive speech analysis"""
    transcribed_text = speech_to_text(audio_file_path)
    
    fluency_results = analyze_fluency_comprehensive(audio_file_path, transcribed_text)
    pitch_results = analyze_pitch_comprehensive(audio_file_path)
    voice_quality_results = analyze_voice_quality(audio_file_path)
    
    filler_count, repetitions = detect_fillers_repetitions(transcribed_text)
    semantic_similarity_score = calculate_semantic_similarity(transcribed_text, ideal_answer_text)
    keyword_coverage_score = calculate_keyword_coverage(transcribed_text, ideal_keywords)
    
    results = {
        "transcribed_text": transcribed_text,
        **fluency_results,
        **pitch_results,
        **voice_quality_results,
        "filler_count": filler_count,
        "repetitions": repetitions,
        "semantic_similarity": semantic_similarity_score,
        "keyword_coverage": keyword_coverage_score,
        "fluency_score": fluency_results.get('fluency_score', 0),
        "clarity_score": clarity_score({
            'semantic_similarity': semantic_similarity_score,
            'keyword_coverage': keyword_coverage_score,
            'filler_count': filler_count
        }, ideal_answer_text, ideal_keywords)
    }
    
    results["overall_score"] = compute_overall_score_independent(results)
    feedback_results = generate_comprehensive_feedback(results)
    results.update(feedback_results)
    
    return results


def finalize_interview(stats: RunningStatistics, user_answer: str, expected_answer: str) -> dict:
    """FINAL research-safe output"""
    metrics = stats.compute_research_metrics()  # ← Calls CLASS method now!
    
    analysis_valid = True
    if stats.total_speaking_time < 3 or stats.total_words < 5:
        analysis_valid = False
    
    return {
        "metrics": metrics,
        "semantic_similarity": metrics.get('avg_semantic_similarity', 0),
        "keyword_coverage": metrics.get('avg_keyword_coverage', 0),
        "overall_relevance": metrics.get('overall_relevance', 0),
        "questions_answered": metrics.get('questions_answered', 0),
        "analysis_valid": analysis_valid
    }


if __name__ == '__main__':
    print("Interview Analyzer module loaded and ready.")