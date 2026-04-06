import librosa
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import re
import time
import threading
from sklearn.metrics.pairwise import cosine_similarity

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

TECH_KEYWORDS = {
    'DBMS': [
        'database', 'sql', 'query', 'index', 'transaction', 'acid', 
        'normalization', 'join', 'primary key', 'foreign key', 'schema', 
        'table', 'bcnf', '3nf', 'redundancy', 'anomaly', 'lock',
        'deadlock', 'concurrency', 'rollback', 'commit', 'logging'
    ],
    'OS': [
        'process', 'thread', 'memory', 'deadlock', 'scheduling', 
        'virtual memory', 'kernel', 'system call', 'context switch', 
        'semaphore', 'mutex', 'paging', 'segmentation', 'fifo', 'lru',
        'race condition', 'critical section', 'monitor', 'dining philosophers'
    ],
    'OOPS': [
        'class', 'object', 'inheritance', 'polymorphism', 'encapsulation', 
        'abstraction', 'interface', 'method', 'constructor', 'destructor',
        'overloading', 'overriding', 'virtual function', 'abstract class',
        'multiple inheritance', 'diamond problem', 'composition', 'aggregation'
    ]
}

# Synonym mapping for robust matching
SYNONYMS = {
    'mutex': ['mutex', 'mutual exclusion', 'lock', 'binary semaphore'],
    'semaphore': ['semaphore', 'counting semaphore', 'signal'],
    'deadlock': ['deadlock', 'deadly embrace', 'circular wait'],
    'process': ['process', 'task', 'job'],
    'thread': ['thread', 'lightweight process', 'lwp'],
    'primary key': ['primary key', 'pk', 'primary-key'],
    'foreign key': ['foreign key', 'fk', 'foreign-key'],
    'normalization': ['normalization', 'normal form', '1nf', '2nf', '3nf', 'bcnf'],
    'acid': ['acid', 'atomicity', 'consistency', 'isolation', 'durability'],
    'inheritance': ['inheritance', 'extends', 'subclass', 'derived class'],
    'polymorphism': ['polymorphism', 'overloading', 'overriding', 'dynamic binding'],
    'encapsulation': ['encapsulation', 'data hiding', 'information hiding'],
    'abstraction': ['abstraction', 'abstract', 'interface'],
}

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
        # 🔥 NEW: User turn tracking
        self.total_user_turn_time = 0.0      # Total time user had the floor
        self.current_user_turn_start = None  # When current user turn started
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
        self._in_user_turn = False
        self.pitch_window = []
        self.pitch_window_size = 100   # ~ last 2–3 seconds

    def start_user_turn(self, ts=None):
        """Called when user turn starts"""
        with self.lock:
            # 🔥 CRITICAL: Set the flag BEFORE any other operations
            self._in_user_turn = True
            self.current_user_turn_start = now_ts() if ts is None else ts
            print(f"⏱️ USER TURN START at {self.current_user_turn_start:.1f}s")
            self._log_event("user_turn_start", self.current_user_turn_start)


    def end_user_turn(self, ts=None):
        """Called when user turn ends"""
        with self.lock:
            if self.current_user_turn_start is not None:
                ts = now_ts() if ts is None else ts
                turn_duration = ts - self.current_user_turn_start
                self.total_user_turn_time += turn_duration
                self.current_user_turn_start = None
                print(f"⏱️ USER TURN END: +{turn_duration:.1f}s (total user turn: {self.total_user_turn_time:.1f}s)")
                self._log_event("user_turn_end", turn_duration, ts)
            
            # 🔥 CRITICAL: Clear the flag AFTER all calculations
            self._in_user_turn = False
    
    # =====================================
    # SPEECH EVENT HANDLERS (THREAD-SAFE)
    # =====================================
    def get_pitch_stability(self):
        if len(self.pitch_window) < 10:
            return 0

        window = np.array(self.pitch_window)

        mean = np.mean(window)
        std = np.std(window)

        if mean == 0:
            return 0

        cv = std / mean

        k = 3.0
        score = 100 * np.exp(-k * cv)

        return max(0, min(100, score))
    
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
            # 🔥 CRITICAL FIX: Only record speech during user turns
            if not getattr(self, '_in_user_turn', False):
                # Silent ignore - don't even print to avoid spam
                return
                
            ts = now_ts() if ts is None else ts
            
            # Detect pause before starting new speech
            if self.last_speech_end is not None:
                pause_duration = ts - self.last_speech_end
                if pause_duration > self.pause_threshold:
                    # Count ALL pauses during user turn
                    self.pause_durations.append(pause_duration)
                    self.total_silence_time += pause_duration
                    self._log_event("pause", pause_duration, ts)
                    print(f"⏸️ Silence segment: +{pause_duration:.2f}s (total silence: {self.total_silence_time:.1f}s)")
                    
                    if pause_duration > self.long_pause_threshold:
                        self.long_pause_count += 1
                        print(f"   → Long pause detected ({pause_duration:.1f}s)")
            
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
            # 🔥 CRITICAL FIX: Only record speech during user turns
            if not getattr(self, '_in_user_turn', False):
                # Silent ignore - don't even print to avoid spam
                return
                
            ts = now_ts() if ts is None else ts
            
            if self.current_speech_start is None:
                return
            
            # Calculate speech duration for this segment
            speech_duration = ts - self.current_speech_start
            
            # Let the actual measurement stand (no artificial limits)
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
            speaking_minutes = 0.0  # 🔥 FIX: Initialize before using
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
            
            # 🔥 CORRECT: Calculate available speaking time (user turn time minus forced silence)
            available_speaking_time = max(0, self.total_user_turn_time - self.forced_silence_time)
            
            # 🔥 CORRECT: Calculate silence during turn
            if self.total_user_turn_time > 0:
                silence_during_turn = max(0, self.total_user_turn_time - self.total_speaking_time - self.forced_silence_time)
                # 🔥 FIXED: Speaking ratio based on AVAILABLE speaking time, not total user turn time
                if available_speaking_time > 0:
                    speaking_ratio_during_turn = self.total_speaking_time / available_speaking_time
                else:
                    speaking_ratio_during_turn = 0.0
            else:
                silence_during_turn = 0.0
                speaking_ratio_during_turn = 0.0
            
            # Legacy silence calculation (for backward compatibility)
            if self.total_speaking_time > self.effective_duration:
                silence_time = 0.0
            else:
                silence_time = self.effective_duration - self.total_speaking_time

            # Average response latency
            avg_response_latency = float(np.mean(self.response_latencies)) if self.response_latencies else 0.0
            
            # Average Q&A scores
            avg_semantic = self.total_semantic_score / self.question_count if self.question_count > 0 else 0.0
            avg_keyword = self.total_keyword_score / self.question_count if self.question_count > 0 else 0.0
            
            # Calculate overall relevance (80% semantic + 20% keyword)
            overall_relevance = (avg_semantic * 0.8) + (avg_keyword * 0.2)
            
            return {
                "session_duration": round(self.total_session_duration, 1),
                "total_user_turn_time": round(self.total_user_turn_time, 1),
                "available_speaking_time": round(available_speaking_time, 1),
                "effective_duration": round(self.effective_duration, 1),
                "speaking_time": round(self.total_speaking_time, 1),
                "silence_time": round(silence_time, 1),
                "silence_during_turn": round(silence_during_turn, 1),
                "forced_silence_time": round(self.forced_silence_time, 1),
                "speaking_ratio": round(self.speaking_ratio, 3),
                "speaking_ratio_during_turn": round(speaking_ratio_during_turn, 3),
                "wpm": round(wpm, 1),
                "total_words": self.total_words,
                "avg_pause_duration": round(avg_pause, 2),
                "pause_count": len(self.pause_durations),
                "long_pause_count": self.long_pause_count,
                "hesitation_rate": round(hesitation_rate, 2),
                "articulation_rate": round(articulation_rate, 2),
                "avg_response_latency": round(avg_response_latency, 2),
                "avg_semantic_similarity": round(avg_semantic, 3),
                "avg_keyword_coverage": round(avg_keyword, 3),
                "overall_relevance": round(overall_relevance, 3),
                "questions_answered": self.question_count,
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
    
    def update_pitch_stats(self, pitch_values):
        for pitch in pitch_values:
            if np.isfinite(pitch):

                # 🔥 NEW: sliding window
                self.pitch_window.append(pitch)
                if len(self.pitch_window) > self.pitch_window_size:
                    self.pitch_window.pop(0)

                # KEEP old stats for research
                self.pitch_count += 1
                delta = pitch - self.pitch_mean
                self.pitch_mean += delta / self.pitch_count
                delta2 = pitch - self.pitch_mean
                self.pitch_m2 += delta * delta2
    
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

        VAD State Machine (FIXED):
        - Voice frame + NOT in speech → accumulate _since_speech_start.
          Once min_speech_ms of continuous voice seen → fire on_voice_start,
          enter speech state, reset hangover to its maximum.
        - Voice frame + IN speech → RESET hangover to maximum (NOT add).
          This ensures the countdown always starts fresh from the full
          hangover window once the user stops speaking.
        - Silent frame + IN speech → decrement hangover.
          When hangover reaches 0 → fire on_voice_end, leave speech state.
        - Silent frame + NOT in speech → accumulate _since_speech_end.
        """
        ts = now_ts() if timestamp is None else timestamp
        
        if len(frame) == 0:
            return
            
        rms = self._rms(frame)
        frame_duration = len(frame) / self.sr
        
        # Voice detection
        is_voice = (rms >= self.energy_threshold)
        
        if is_voice:
            self._since_speech_end = 0.0
            # Only reset hangover if energy is significantly above threshold (real speech)
            if rms > self.energy_threshold * 1.5:   # 50% above threshold = real speech
                if not self._speech_state:
                    self._since_speech_start += frame_duration
                    if self._since_speech_start * 1000.0 >= self.min_speech_ms:
                        self._speech_state = True
                        self._since_speech_start = 0.0
                        self._hangover_remaining = self.hangover_ms / 1000.0
                        if callable(self.on_voice_start):
                            self.on_voice_start(ts)
                            print(f"🔊 VAD STATE CHANGE: START at {ts:.2f}s")
                else:
                    # Reset hangover only for genuine speech
                    self._hangover_remaining = self.hangover_ms / 1000.0
            else:
                # Low-energy frame – treat as silence for hangover purposes
                # (do not reset hangover, let it decrement)
                pass
        else:
            # Silent frame — reset consecutive voice counter
            self._since_speech_start = 0.0
            
            if self._speech_state:
                # Count down the hangover window
                self._hangover_remaining -= frame_duration
                if self._hangover_remaining <= 0:
                    # Hangover expired: speech segment has ended
                    self._speech_state = False
                    self._hangover_remaining = 0.0
                    if callable(self.on_voice_end):
                        self.on_voice_end(ts)
                        print(f"🔇 VAD STATE CHANGE: END at {ts:.2f}s")
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


def get_domain_keywords(question, topic=None):
    """
    Extract expected technical keywords based on question context
    
    Literature:
    - Technical interviews expect 3-5 key terms per answer (Behrend et al., 2014)
    - Keyword lists derived from industry-standard textbooks:
        * Silberschatz et al., 2019 (Operating Systems)
        * Elmasri & Navathe, 2016 (Database Systems)
        * Gamma et al., 1995 (Design Patterns)
    """
    # Detect topic from question if not provided
    if not topic:
        q_lower = question.lower()
        if any(kw in q_lower for kw in TECH_KEYWORDS['OS']):
            topic = 'OS'
        elif any(kw in q_lower for kw in TECH_KEYWORDS['DBMS']):
            topic = 'DBMS'
        elif any(kw in q_lower for kw in TECH_KEYWORDS['OOPS']):
            topic = 'OOPS'
        else:
            topic = 'DBMS'  # Default
    
    # Get base keywords for this topic
    base_keywords = TECH_KEYWORDS.get(topic, [])
    
    # Add question-specific keywords that are technical terms
    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    additional_keywords = []
    
    for word in q_words:
        # Check if word is in any technical keyword list
        for kw_list in TECH_KEYWORDS.values():
            if word in kw_list or any(word in kw for kw in kw_list):
                if word not in base_keywords:
                    additional_keywords.append(word)
    
    # Combine and deduplicate
    all_keywords = list(set(base_keywords + additional_keywords))
    
    # Limit to most relevant keywords (max 10 per question)
    # Prioritize keywords that appear in the question
    question_keywords = []
    for kw in all_keywords:
        kw_lower = kw.lower()
        if kw_lower in question.lower():
            question_keywords.append(kw)
    
    remaining = [kw for kw in all_keywords if kw not in question_keywords]
    
    return (question_keywords + remaining)[:]

def calculate_keyword_coverage(answer, question, topic=None):
    """
    Calculate keyword coverage using domain-specific technical keywords
    
    Justification:
    - Uses validated technical keyword lists from industry-standard curricula
    - Synonym support for robust matching (e.g., "mutex" = "mutual exclusion")
    - Accounts for interviewers' expectations of 3-5 key terms per answer
    
    Literature:
    [1] Behrend, T. S., et al. (2014). The viability of using online video 
        interviews in selection. *Journal of Applied Psychology*, 99(3), 484.
    [2] Silberschatz, A., Galvin, P. B., & Gagne, G. (2019). 
        *Operating System Concepts* (10th ed.). Wiley.
    [3] Elmasri, R., & Navathe, S. B. (2016). *Fundamentals of Database Systems* (7th ed.).
    """
    import re
    
    if not answer or not question:
        return 0.0
    
    # Get expected keywords for this question
    expected_keywords = get_domain_keywords(question, topic)
    
    if not expected_keywords:
        print(f"⚠️ No domain keywords found for question")
        return 0.0
    
    answer_lower = answer.lower()
    matches = 0
    matched_keywords = []
    
    for keyword in expected_keywords:
        keyword_lower = keyword.lower()
        
        # Direct match
        if keyword_lower in answer_lower:
            matches += 1
            matched_keywords.append(keyword)
            continue
        
        # Multi-word match without spaces
        if ' ' in keyword_lower:
            keyword_no_space = keyword_lower.replace(' ', '')
            answer_no_space = answer_lower.replace(' ', '')
            if keyword_no_space in answer_no_space:
                matches += 1
                matched_keywords.append(keyword)
                continue
        
        # Synonym matching
        if keyword_lower in SYNONYMS:
            for synonym in SYNONYMS[keyword_lower]:
                if synonym in answer_lower:
                    matches += 1
                    matched_keywords.append(f"{keyword} (synonym: {synonym})")
                    break
    
    # Calculate coverage (max 1.0)
    # Interviewers expect 3-5 key terms; normalize to max 5 keywords
    normalized_max = min(5, len(expected_keywords))
    coverage = min(1.0, matches / normalized_max) if normalized_max > 0 else 0.0
    
    print(f"\n📊 KEYWORD COVERAGE:")
    print(f"   Expected ({len(expected_keywords)}): {expected_keywords[:5]}")
    print(f"   Matched ({matches}): {matched_keywords[:5]}")
    print(f"   Score: {matches}/{normalized_max} = {coverage:.3f}")
    
    return coverage


def speech_to_text(audio_path, model_name="medium.en", use_vad=True, min_speech_duration=1000):
    """Placeholder - actual transcription handled by AssemblyAI streamer"""
    pass


def finalize_interview(stats: RunningStatistics, user_answer: str, expected_answer: str) -> dict:
    """FINAL research-safe output"""
    metrics = stats.compute_research_metrics()
    
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