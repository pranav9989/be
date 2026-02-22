import librosa
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import spacy
from faster_whisper import WhisperModel
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import concurrent.futures
from functools import partial

# Load models
# Only load models once when the module is imported
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Check if GPU is available and use it
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"  # Use int8 for CPU (faster)
print(f"Using device: {device} for Faster-Whisper models (compute_type: {compute_type})")

# Model manager for different use cases
class WhisperModelManager:
    def __init__(self):
        self.models = {}

    def get_model(self, model_name="medium.en"):
        """
        Get or create a Whisper model.
        model_name options:
        - "medium.en": Fast, good for live processing (3x faster than large-v3)
        - "large-v3": Most accurate, good for offline detailed analysis
        """
        if model_name not in self.models:
            print(f"Loading Whisper model: {model_name}")
            try:
                self.models[model_name] = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type
                )
                print(f"[OK] Loaded {model_name} model successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load {model_name}: {e}")
                # Fallback to medium.en if large-v3 fails
                if model_name == "large-v3":
                    print("Falling back to medium.en")
                    return self.get_model("medium.en")
                raise e

        return self.models[model_name]

# Global model manager
model_manager = WhisperModelManager()

# For backward compatibility, keep a reference to medium.en as default
whisper_model = model_manager.get_model("medium.en")


class RunningStatistics:
    """
    Maintains running statistics for incremental speech analysis.
    Supports Welford's online algorithm for mean/variance calculation.
    """

    def __init__(self):
        # Text/transcription statistics
        self.total_words = 0
        self.transcript_parts = []  # This stores ALL user speech parts

        # 🔥 Store full conversation
        self.conversation = []  # List of {"role": "interviewer" or "user", "text": text}
        self.full_transcript = []  # Formatted conversation for display

        # Time statistics
        self.total_duration = 0.0
        self.speaking_time = 0.0

        # Pause statistics
        self.pause_durations = []
        self.long_pause_count = 0

        # Pitch statistics (using Welford's algorithm)
        self.pitch_count = 0
        self.pitch_mean = 0.0
        self.pitch_m2 = 0.0  # For variance calculation
        self.pitch_min = float('inf')
        self.pitch_max = float('-inf')

        # Filler word statistics
        self.filler_counts = {}

        # Voice quality statistics
        self.jitter_values = []
        self.shimmer_values = []
        self.hnr_values = []

        # 🔥 Track per-question scores WITH expected answers
        self.question_scores = []  # List of dicts with question, answer, expected_answer, similarity, keyword_coverage
        self.total_semantic_score = 0.0
        self.total_keyword_score = 0.0
        self.question_count = 0

    # 🔥 Record interviewer question
    def record_question(self, question):
        """Record interviewer question"""
        self.conversation.append({
            'role': 'interviewer',
            'text': question
        })
        # Also add to formatted transcript
        self.full_transcript.append(f"Interviewer: {question}")
        print(f"📝 Recorded question: {question[:50]}...")

    # 🔥 MODIFIED: Update record_qa_pair to include expected_answer
    def record_qa_pair(self, question, answer, expected_answer, similarity_score, keyword_score):
        """Record a question-answer pair with its scores and the expected answer"""
        
        # FIX: Ensure question is recorded BEFORE answer
        # Check if this question is already in conversation
        question_exists = False
        for msg in self.conversation:
            if msg['role'] == 'interviewer' and msg['text'] == question:
                question_exists = True
                break
        
        if not question_exists:
            self.conversation.append({
                'role': 'interviewer',
                'text': question
            })
            self.full_transcript.append(f"Interviewer: {question}")
        
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
        
        # Record user answer
        self.conversation.append({
            'role': 'user',
            'text': answer
        })
        self.full_transcript.append(f"User: {answer}")
        
        print(f"📝 Recorded Q&A pair #{self.question_count}:")
        print(f"   Q: {question[:50]}...")
        print(f"   A: {answer[:50]}...")
        print(f"   Expected: {expected_answer[:50]}...")
        print(f"   Semantic: {similarity_score:.3f}, Keyword: {keyword_score:.3f}")

    def update_transcript(self, new_text):
        """Update transcription statistics incrementally."""
        if new_text:
            self.transcript_parts.append(new_text)
            words_in_segment = len(new_text.split())
            self.total_words += words_in_segment

    def update_time_stats(self, segment_duration, speaking_duration):
        """Update time-based statistics with validation."""
        self.total_duration += segment_duration
        
        # 🔥 CRITICAL: Speaking time cannot exceed total duration
        speaking_duration = min(speaking_duration, segment_duration)
        self.speaking_time += speaking_duration

    def update_pause_stats(self, pause_durations):
        """Update pause statistics with proper long pause detection (5s threshold)."""
        for pause in pause_durations:
            self.pause_durations.append(pause)
            # 🔥 FIXED: Long pause threshold = 5 seconds (not 1 second)
            if pause > 5.0:  # Changed from 1.0 to 5.0
                self.long_pause_count += 1
                print(f"⏸️ LONG PAUSE DETECTED: {pause:.1f}s (count: {self.long_pause_count})")

    def update_pitch_stats(self, pitch_values):
        """Update pitch statistics using Welford's online algorithm."""
        for pitch in pitch_values:
            if np.isfinite(pitch):  # Only include finite values
                self.pitch_count += 1
                delta = pitch - self.pitch_mean
                self.pitch_mean += delta / self.pitch_count
                delta2 = pitch - self.pitch_mean
                self.pitch_m2 += delta * delta2

                self.pitch_min = min(self.pitch_min, pitch)
                self.pitch_max = max(self.pitch_max, pitch)

    def update_filler_stats(self, filler_counts):
        """Update filler word statistics."""
        for filler, count in filler_counts.items():
            self.filler_counts[filler] = self.filler_counts.get(filler, 0) + count

    def update_voice_quality(self, jitter, shimmer, hnr):
        """Update voice quality metrics."""
        if np.isfinite(jitter):
            self.jitter_values.append(jitter)
        if np.isfinite(shimmer):
            self.shimmer_values.append(shimmer)
        if np.isfinite(hnr):
            self.hnr_values.append(hnr)

    # 🔥 MODIFIED: Update get_current_stats to include expected answers
    def get_current_stats(self):
        """Get current computed statistics including Q&A scores."""
        # Calculate derived statistics
        transcript = " ".join(self.transcript_parts)  # This is user speech only
        
        # Create formatted conversation
        conversation_text = "\n\n".join(self.full_transcript)

        # WPM calculation
        wpm = (self.total_words / self.speaking_time * 60) if self.speaking_time > 3 else 0

        # Pause ratio
        pause_ratio = (self.total_duration - self.speaking_time) / self.total_duration if self.total_duration > 0 else 0

        # Pitch statistics
        pitch_std = np.sqrt(self.pitch_m2 / self.pitch_count) if self.pitch_count > 1 else 0
        pitch_range = self.pitch_max - self.pitch_min if self.pitch_count > 0 else 0

        # Voice quality averages
        avg_jitter = np.mean(self.jitter_values) if self.jitter_values else 0
        avg_shimmer = np.mean(self.shimmer_values) if self.shimmer_values else 0
        avg_hnr = np.mean(self.hnr_values) if self.hnr_values else 0

        # Calculate average Q&A scores
        avg_semantic = self.total_semantic_score / self.question_count if self.question_count > 0 else 0
        avg_keyword = self.total_keyword_score / self.question_count if self.question_count > 0 else 0

        # 🔥 NEW: Reduce keyword weight to 20% of overall
        combined_score = (avg_semantic * 0.7) + (avg_keyword * 0.3)

        return {
            'transcript': transcript,
            'conversation': conversation_text,  # Full conversation
            'conversation_parts': self.conversation,  # Raw conversation parts
            'total_words': self.total_words,
            'total_duration': self.total_duration,
            'speaking_time': self.speaking_time,
            'wpm': wpm,
            'pause_ratio': pause_ratio,
            'pitch_mean': self.pitch_mean,
            'pitch_std': pitch_std,
            'pitch_min': self.pitch_min if self.pitch_count > 0 else 0,
            'pitch_max': self.pitch_max if self.pitch_count > 0 else 0,
            'pitch_range': pitch_range,
            'filler_counts': self.filler_counts,
            'avg_jitter': avg_jitter,
            'avg_shimmer': avg_shimmer,
            'avg_hnr': avg_hnr,
            # Q&A metrics
            'question_count': self.question_count,
            'avg_semantic_similarity': avg_semantic,
            'avg_keyword_coverage': avg_keyword,
            'combined_relevance_score': combined_score,
            'qa_pairs': self.question_scores  # Now includes expected_answer
        }


def analyze_audio_chunk_fast(pcm_chunk, sample_rate, stats: RunningStatistics):
    """
    Analyze audio chunk with proper VAD and pause detection.
    Long pause threshold = 5 seconds (tracked across chunks).
    """
    # Convert int16 PCM → float32 for librosa
    audio = pcm_chunk.astype(np.float32) / 32768.0
    duration = len(audio) / sample_rate
    
    # 🔥 FIXED: Use proper VAD with conservative parameters
    try:
        import librosa
        
        # Calculate energy of the chunk
        energy = np.sqrt(np.mean(audio**2))
        
        # Adaptive threshold based on chunk energy
        if energy < 0.005:  # Very quiet - probably silence
            speaking_time = 0
            stats.update_time_stats(duration, speaking_time)
            
            # 🔥 CRITICAL: For silence, we need to track potential long pauses
            # This will be handled by the silence_watcher in app.py
            return
        
        # Use librosa's VAD with better parameters for speech detection
        intervals = librosa.effects.split(
            audio, 
            top_db=25,  # 🔥 REDUCED from 30 to 25 (more sensitive to speech)
            frame_length=2048,
            hop_length=512
        )
        
        # Calculate actual speaking time
        speaking_time = 0
        pauses_in_chunk = []
        
        if len(intervals) > 0:
            # Sum all speech intervals
            for s, e in intervals:
                speaking_time += (e - s) / sample_rate
            
            # 🔥 CRITICAL: Speaking time CANNOT exceed total duration
            speaking_time = min(speaking_time, duration)
            
            # Calculate pauses BETWEEN intervals
            if len(intervals) > 1:
                for i in range(len(intervals) - 1):
                    pause_start = intervals[i][1] / sample_rate
                    pause_end = intervals[i+1][0] / sample_rate
                    pause_duration = pause_end - pause_start
                    
                    # Only count significant pauses (> 0.3s)
                    if pause_duration > 0.3:
                        pauses_in_chunk.append(pause_duration)
                        print(f"⏸️ Pause in chunk: {pause_duration:.2f}s")
        
        # Update stats
        stats.update_time_stats(duration, speaking_time)
        stats.update_pause_stats(pauses_in_chunk)
        
        
    except Exception as e:
        print(f"VAD failed: {e}")
        # Fallback: assume 30% speaking time if detection fails
        stats.update_time_stats(duration, duration * 0.3)

def detect_fillers_repetitions(text):
    fillers = ["um", "uh", "like", "you know", "i mean"]
    filler_count = 0
    repetitions = 0
    words = text.lower().split()
    
    for i, word in enumerate(words):
        if word in fillers:
            filler_count += 1
        # Simple repetition detection for immediate consecutive words
        if i > 0 and words[i] == words[i-1]:
            repetitions += 1
            
    return filler_count, repetitions

def calculate_semantic_similarity(answer, expected_answer):
    """
    Calculate TRUE semantic similarity between answer and expected answer.
    Returns raw cosine similarity (0.0 to 1.0) - NO ARTIFICIAL SCALING.
    """
    # 🔥 FIX: Handle empty answers
    if not answer or not answer.strip():
        print(f"📊 Empty answer detected, similarity = 0.0")
        return 0.0
    
    if not expected_answer or not expected_answer.strip():
        print(f"📊 Empty expected answer, similarity = 0.0")
        return 0.0
    
    try:
        # Encode both texts
        answer_emb = embedder.encode([answer], normalize_embeddings=True)[0]
        expected_emb = embedder.encode([expected_answer], normalize_embeddings=True)[0]
        
        # Calculate raw cosine similarity
        similarity = cosine_similarity([answer_emb], [expected_emb])[0][0]
        
        # 🔥 Ensure non-negative (cosine similarity can be slightly negative, but for text it should be >= 0)
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
    
    # 🔥 Stop words to ignore
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                  'that', 'these', 'those', 'there', 'here', 'what', 'which', 'who',
                  'whom', 'whose', 'why', 'how', 'between', 'difference'}
    
    # Technical keywords by topic
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
    
    # Extract words from question
    question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    
    # Remove stop words
    question_words = {w for w in question_words if w not in STOP_WORDS}
    
    # Add technical keywords if they appear in question
    for topic, keywords in tech_keywords.items():
        for kw in keywords:
            if kw in question.lower():
                if ' ' in kw:
                    question_words.add(kw)
                else:
                    question_words.add(kw)
    
    # Count matches in answer
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
    
    # Calculate RAW coverage (0.0 to 1.0) - NO SCALING
    if question_words:
        coverage = min(1.0, matches / len(question_words))
    else:
        coverage = 0.0
    
    print(f"🔑 RAW Keyword coverage: {matches}/{len(question_words)} = {coverage:.3f}")
    print(f"   Matched: {matched_keywords}")
    print(f"   Stop words filtered: {len(question_words)} keywords considered")
    
    return coverage  # 🔥 Return RAW value (0.0-1.0)

def analyze_pitch_comprehensive(audio_path):
    """
    Comprehensive pitch analysis including range, average, stability, and quality score.
    Returns: dict with pitch metrics
    """
    y, sr = librosa.load(audio_path)

    # Fundamental frequency (F0) tracking using YIN (much faster than pYIN)
    # Focus on speech-relevant frequency range for better performance
    f0 = librosa.yin(
        y, sr=sr,
        fmin=65,  # 65 Hz (male speech range)
        fmax=300, # 300 Hz (female speech range)
        frame_length=512,  # Smaller frames for speed
        hop_length=128     # Faster hop
    )

    # Filter out NaN/infinite values (YIN doesn't provide voiced_flag, so we filter directly)
    f0_voiced = f0[np.isfinite(f0)]

    if len(f0_voiced) < 10:  # Need minimum samples for analysis
        return {
            "pitch_mean": 0,
            "pitch_std": 0,
            "pitch_min": 0,
            "pitch_max": 0,
            "pitch_range": 0,
            "pitch_stability": 0,
            "pitch_score": 0,
            "pitch_feedback": "Insufficient voiced audio for pitch analysis"
        }

    # Calculate pitch metrics
    pitch_mean = np.mean(f0_voiced)
    pitch_std = np.std(f0_voiced)
    pitch_min = np.min(f0_voiced)
    pitch_max = np.max(f0_voiced)
    pitch_range = pitch_max - pitch_min

    # Pitch stability (coefficient of variation - lower is more stable)
    pitch_stability = pitch_std / pitch_mean if pitch_mean > 0 else 1.0

    # Pitch score (0-100) - based on stability and appropriate range
    # Ideal pitch range for speech is typically 85-180 Hz for males, 165-255 Hz for females
    # Good stability has low coefficient of variation (< 0.3)
    stability_score = max(0, min(100, 100 * (1 - pitch_stability / 0.5)))  # Lower stability = higher score

    # Range score - penalize too narrow or too wide ranges
    if 50 < pitch_range < 300:  # Good range
        range_score = 100
    elif 20 < pitch_range < 400:  # Acceptable range
        range_score = 70
    else:  # Too narrow or too wide
        range_score = 30

    # Average pitch score - prefer mid-range pitches
    if 100 < pitch_mean < 250:  # Good average pitch range
        mean_score = 100
    elif 80 < pitch_mean < 300:  # Acceptable range
        mean_score = 80
    else:  # Too low or too high
        mean_score = 40

    pitch_score = (stability_score * 0.5 + range_score * 0.3 + mean_score * 0.2)

    # Generate feedback
    feedback_parts = []
    if pitch_stability > 0.4:
        feedback_parts.append("pitch varies too much")
    elif pitch_stability < 0.2:
        feedback_parts.append("pitch is very stable")

    if pitch_range < 50:
        feedback_parts.append("pitch range is too narrow")
    elif pitch_range > 400:
        feedback_parts.append("pitch range is too wide")

    if pitch_mean < 85:
        feedback_parts.append("pitch is quite low")
    elif pitch_mean > 300:
        feedback_parts.append("pitch is quite high")

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
    """
    Analyze voice quality metrics: jitter, shimmer, and HNR (Harmonics-to-Noise Ratio).
    Returns: dict with voice quality metrics
    """
    y, sr = librosa.load(audio_path)

    # Get fundamental frequency using YIN (much faster than pYIN)
    f0 = librosa.yin(
        y, sr=sr,
        fmin=65,  # 65 Hz (male speech range)
        fmax=300, # 300 Hz (female speech range)
        frame_length=512,  # Smaller frames for speed
        hop_length=128     # Faster hop
    )

    # Filter to finite values only (YIN doesn't provide voiced_flag)
    f0_voiced = f0[np.isfinite(f0)]

    if len(f0_voiced) < 20:  # Need minimum samples
        return {
            "jitter": 0,
            "shimmer": 0,
            "hnr": 0,
            "voice_quality_score": 0,
            "voice_quality_feedback": "Insufficient voiced audio for voice quality analysis"
        }

    # Calculate Jitter (pitch perturbation)
    # Jitter is the cycle-to-cycle variation in fundamental frequency
    if len(f0_voiced) > 1:
        f0_diffs = np.abs(np.diff(f0_voiced))
        jitter = np.mean(f0_diffs) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0
    else:
        jitter = 0

    # Calculate Shimmer (amplitude perturbation)
    # For simplicity, we'll use RMS amplitude variations over short windows
    frame_length = int(sr * 0.02)  # 20ms frames
    hop_length = int(sr * 0.01)    # 10ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Use RMS variations as shimmer approximation
    if len(rms) > 1:
        rms_diffs = np.abs(np.diff(rms))
        shimmer = np.mean(rms_diffs) / np.mean(rms) if np.mean(rms) > 0 else 0
    else:
        shimmer = 0

    # Calculate Harmonics-to-Noise Ratio (HNR) using autocorrelation
    def calculate_hnr(signal):
        """Calculate HNR using autocorrelation method"""
        if len(signal) < 100:  # Need minimum signal length
            return 0

        # Remove DC component
        signal = signal - np.mean(signal)

        # Calculate autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags

        # Find peak and noise floor
        peak_idx = np.argmax(autocorr[:len(autocorr)//4])

        if peak_idx < len(autocorr) - 10:
            # Find minimum after peak (approximate noise floor)
            search_start = peak_idx + 5
            search_end = min(search_start + 50, len(autocorr))
            noise_floor = np.min(autocorr[search_start:search_end])

            # HNR in dB
            hnr = 10 * np.log10(autocorr[peak_idx] / noise_floor) if noise_floor > 0 else 0
        else:
            hnr = 0

        return max(0, hnr)  # Ensure non-negative

    # Calculate HNR for multiple segments
    hnr_values = []
    segment_length = int(sr * 0.1)  # 100ms segments

    for i in range(0, len(y) - segment_length, segment_length // 2):
        segment = y[i:i + segment_length]
        hnr_val = calculate_hnr(segment)
        if hnr_val > 0:  # Only include valid HNR values
            hnr_values.append(hnr_val)

    hnr = np.mean(hnr_values) if hnr_values else 0

    # Voice quality scores (0-100)
    # Jitter: lower is better (ideal < 0.01)
    jitter_score = max(0, min(100, 100 * (1 - jitter / 0.02)))

    # Shimmer: lower is better (ideal < 0.05)
    shimmer_score = max(0, min(100, 100 * (1 - shimmer / 0.1)))

    # HNR: higher is better (ideal > 15 dB)
    hnr_score = max(0, min(100, hnr * 6.67))  # 15 dB = 100 points

    # Overall voice quality score
    voice_quality_score = (jitter_score * 0.3 + shimmer_score * 0.3 + hnr_score * 0.4)

    # Generate feedback
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
    """
    Extract voiced segments from audio for efficient processing.

    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds (default 5s)
        overlap: Overlap between chunks in seconds (default 0.5s)

    Returns:
        List of (start_time, end_time) tuples for voiced segments
    """
    # Load audio at lower sample rate for efficiency (we'll upsample for ASR if needed)
    y, sr = librosa.load(audio_path, sr=8000)  # 8kHz is sufficient for voice detection

    # Detect voiced segments
    intervals = librosa.effects.split(y, top_db=20)

    # Convert sample indices to time
    voiced_segments = []
    for start_sample, end_sample in intervals:
        start_time = start_sample / sr
        end_time = end_sample / sr
        duration = end_time - start_time

        # Only include segments longer than 0.5 seconds
        if duration >= 0.5:
            voiced_segments.append((start_time, end_time))

    return voiced_segments


def transcribe_voiced_segments(audio_path, model_name="medium.en"):
    """
    Transcribe only the voiced segments of audio for efficiency.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use

    Returns:
        Tuple of (full_transcript, segment_details)
    """
    try:
        # Load full audio at 16kHz for ASR
        y_full, sr_full = librosa.load(audio_path, sr=16000)

        # Get voiced segments at lower sample rate
        voiced_segments = get_voiced_segments(audio_path)

        if not voiced_segments:
            return "", []

        # Transcribe each voiced segment
        full_transcript = ""
        segment_details = []

        model = model_manager.get_model(model_name)

        for start_time, end_time in voiced_segments:
            # Convert time to samples (16kHz)
            start_sample = int(start_time * sr_full)
            end_sample = int(end_time * sr_full)

            # Extract segment
            segment_audio = y_full[start_sample:end_sample]

            # Skip if segment is too short
            if len(segment_audio) < sr_full:  # Less than 1 second
                continue

            # Transcribe segment
            segments, info = model.transcribe(
                segment_audio,
                language="en",
                beam_size=3,  # Reduced for speed
                vad_filter=False  # Already filtered
            )

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
        # Fallback to full transcription
        return speech_to_text(audio_path, model_name), []


def parallel_analyze_segment(segment_data, model_name="medium.en"):
    """
    Analyze a single segment in parallel.
    Returns transcription and basic metrics for that segment.
    """
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
        # Transcribe segment
        model = model_manager.get_model(model_name)
        segments, _ = model.transcribe(
            segment_audio,
            language="en",
            beam_size=3,
            vad_filter=False
        )
        transcript = " ".join([seg.text for seg in segments]).strip()
        results['transcript'] = transcript
        results['word_count'] = len(transcript.split()) if transcript else 0

        # Basic filler detection
        if transcript:
            fillers = ["um", "uh", "like", "you know", "i mean"]
            filler_count = sum(transcript.lower().count(filler) for filler in fillers)
            results['filler_count'] = filler_count

    except Exception as e:
        print(f"Error analyzing segment {start_time}-{end_time}: {e}")

    return results


def parallel_pitch_analysis(audio_path):
    """Parallel pitch analysis function."""
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
    """Parallel voice quality analysis function."""
    try:
        return analyze_voice_quality(audio_path)
    except Exception as e:
        print(f"Parallel voice quality analysis failed: {e}")
        return {
            "jitter": 0, "shimmer": 0, "hnr": 0, "voice_quality_score": 0,
            "voice_quality_feedback": "Analysis failed"
        }


def analyze_audio_chunk(audio_chunk, sample_rate, stats):
    """
    Analyze a single audio chunk and update running statistics.

    Args:
        audio_chunk: Numpy array of audio samples
        sample_rate: Sample rate of the audio
        stats: RunningStatistics object to update

    Returns:
        Dict with chunk-specific analysis results
    """
    chunk_duration = len(audio_chunk) / sample_rate

    # Detect silence/voice in this chunk
    intervals = librosa.effects.split(audio_chunk, top_db=20)
    speaking_time = sum((end - start) / sample_rate for start, end in intervals)

    # Update time statistics
    stats.update_time_stats(chunk_duration, speaking_time)

    # Calculate pause durations within this chunk
    pause_durations = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            current_end = intervals[i][1] / sample_rate
            next_start = intervals[i + 1][0] / sample_rate
            pause_duration = next_start - current_end
            if pause_duration > 0.1:  # Only count pauses > 100ms
                pause_durations.append(pause_duration)

    stats.update_pause_stats(pause_durations)

    # Pitch analysis (use 8kHz for efficiency)
    if sample_rate != 8000:
        chunk_8k = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=8000)
        sr_8k = 8000
    else:
        chunk_8k = audio_chunk
        sr_8k = sample_rate

    try:
        # Use YIN instead of pYIN for much faster pitch detection
        f0 = librosa.yin(
            chunk_8k, sr=sr_8k,
            fmin=65,    # Male speech range
            fmax=300,   # Female speech range
            frame_length=512,   # Smaller for speed
            hop_length=128      # Faster hop
        )

        # Update pitch statistics (filter finite values)
        voiced_f0 = f0[np.isfinite(f0)]
        if len(voiced_f0) > 0:
            stats.update_pitch_stats(voiced_f0)
    except Exception as e:
        print(f"Pitch analysis failed for chunk: {e}")

    # Voice quality analysis (basic version for chunks)
    try:
        # Simple jitter/shimmer approximation
        if len(voiced_f0) > 10:
            # Jitter: coefficient of variation of F0
            jitter = np.std(voiced_f0) / np.mean(voiced_f0) if np.mean(voiced_f0) > 0 else 0

            # Shimmer: approximate using RMS variation
            rms = librosa.feature.rms(y=chunk_8k, frame_length=256, hop_length=128)[0]
            shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0

            # HNR approximation (simplified)
            hnr = 10  # Placeholder - would need proper HNR calculation

            stats.update_voice_quality(jitter, shimmer, hnr)
    except Exception as e:
        print(f"Voice quality analysis failed for chunk: {e}")

    return {
        'chunk_duration': chunk_duration,
        'speaking_time': speaking_time,
        'pause_count': len(pause_durations)
    }


def analyze_interview_response_optimized(audio_path, ideal_answer_text="", ideal_keywords=None, use_large_model=False):
    """
    Optimized interview response analysis using chunked processing and incremental statistics.

    Args:
        audio_path: Path to audio file
        ideal_answer_text: Expected answer text for comparison
        ideal_keywords: List of expected keywords
        use_large_model: Whether to use large-v3 model (slower but more accurate)

    Returns:
        Comprehensive analysis results
    """
    if ideal_keywords is None:
        ideal_keywords = []

    # Choose model based on use case
    model_name = "large-v3" if use_large_model else "medium.en"

    # Initialize running statistics
    stats = RunningStatistics()

    # Load audio for chunked processing
    y_full, sr_full = librosa.load(audio_path, sr=16000)  # 16kHz for ASR

    # Process in chunks for efficiency
    chunk_samples = int(5.0 * sr_full)  # 5-second chunks
    overlap_samples = int(0.5 * sr_full)  # 0.5 second overlap

    # Get voiced segments first for efficiency
    voiced_segments = get_voiced_segments(audio_path)

    if not voiced_segments:
        # No voice detected
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

    # Process voiced segments efficiently
    for start_time, end_time in voiced_segments:
        start_sample = int(start_time * sr_full)
        end_sample = int(end_time * sr_full)

        # Extract segment
        segment_audio = y_full[start_sample:end_sample]

        # Analyze chunk
        chunk_results = analyze_audio_chunk(segment_audio, sr_full, stats)

        # Transcribe this segment
        if len(segment_audio) >= sr_full:  # At least 1 second
            try:
                model = model_manager.get_model(model_name)
                segments, _ = model.transcribe(
                    segment_audio,
                    language="en",
                    beam_size=3,
                    vad_filter=False
                )
                segment_text = " ".join([seg.text for seg in segments]).strip()

                if segment_text:
                    stats.update_transcript(segment_text)

                    # Analyze fillers in this segment
                    filler_count, _ = detect_fillers_repetitions(segment_text)
                    stats.update_filler_stats({"total": filler_count})

            except Exception as e:
                print(f"Transcription failed for segment {start_time}-{end_time}: {e}")

    # Get final statistics
    final_stats = stats.get_current_stats()

    # Calculate content analysis
    transcript = final_stats['transcript']
    semantic_similarity = calculate_semantic_similarity(transcript, ideal_answer_text) if transcript else 0
    keyword_coverage = calculate_keyword_coverage(transcript, ideal_keywords) if transcript else 0

    # Calculate scores using the optimized data
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

    # Generate comprehensive feedback
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
        # Core transcript and metrics
        "transcribed_text": transcript,
        "total_words": final_stats['total_words'],
        "total_duration": final_stats['total_duration'],
        "speaking_time": final_stats['speaking_time'],

        # Fluency metrics
        "wpm": fluency_results['wpm'],
        "pause_ratio": fluency_results['pause_ratio'],
        "filler_count": fluency_results['filler_count'],
        "long_pause_count": stats.long_pause_count,

        # Pitch metrics
        "pitch_mean": final_stats['pitch_mean'],
        "pitch_std": final_stats['pitch_std'],
        "pitch_range": final_stats['pitch_range'],
        "pitch_min": final_stats['pitch_min'],
        "pitch_max": final_stats['pitch_max'],

        # Voice quality metrics
        "avg_jitter": final_stats['avg_jitter'],
        "avg_shimmer": final_stats['avg_shimmer'],
        "avg_hnr": final_stats['avg_hnr'],

        # Content analysis
        "semantic_similarity": semantic_similarity,
        "keyword_coverage": keyword_coverage,

        # Scores and feedback
        **feedback_data,

        # Processing metadata
        "processing_method": "optimized_chunked",
        "model_used": model_name,
        "voiced_segments_count": len(voiced_segments)
    }



def speech_to_text(audio_path, model_name="medium.en", use_vad=True, min_speech_duration=1000):
    """
    Transcribe audio using faster-whisper with configurable model.
    Can accept either a file path (str) or numpy array.
    """
    try:
        # Get the appropriate model
        model = model_manager.get_model(model_name)
        
        # Configure VAD based on parameter
        vad_filter = use_vad
        vad_parameters = None
        
        if vad_filter:
            vad_parameters = {
                "min_speech_duration_ms": min_speech_duration,
                "max_speech_duration_s": 30,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            }
        
        # Handle both file paths and numpy arrays
        segments, info = model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            # Add these parameters for better accuracy on first run
            condition_on_previous_text=False,
            temperature=0,
            best_of=2
        )
        
        # Join all segments to get full text
        transcribed_text = " ".join([segment.text for segment in segments])
        
        # Log transcription info for debugging
        print(f"📝 Transcription completed:")
        print(f"   Language: {info.language if info else 'Unknown'}")
        print(f"   Text length: {len(transcribed_text)} characters")
        print(f"   Model: {model_name}")
        
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"❌ Error during speech-to-text transcription with {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return ""
        
def fluency_score(results):
    """
    Calculate fluency score (0-100) based on WPM, pause ratio, and fillers.
    """
    wpm_norm = min(1.0, results['wpm'] / 150)  # 150 WPM target
    fluency = 100 * (0.4 * wpm_norm + 
                     0.3 * (1 - results['pause_ratio']) + 
                     0.2 * (1 - min(1.0, results['filler_count'] / 50)) + 
                     0.1 * 0.8)  # Base score component
    return max(0, min(100, fluency))




def clarity_score(results, ideal_answer, ideal_keywords):
    """
    Calculate clarity score (0-100) based on semantic similarity and keyword coverage.
    """
    clarity = 100 * (0.5 * results['semantic_similarity'] + 
                     0.3 * results['keyword_coverage'] + 
                     0.2 * 0.9)  # Base score component
    return max(0, min(100, clarity))

def compute_overall_score_independent(results):
    """
    Research-safe overall score using ONLY independent metrics.
    """

    # --- Content ---
    semantic_similarity = results.get("semantic_similarity", 0.0)
    keyword_coverage = results.get("keyword_coverage", 0.0)

    semantic_score = semantic_similarity * 100
    keyword_score = keyword_coverage * 100

    # --- Pitch stability proxy (independent) ---
    pitch_mean = results.get("pitch_mean", 0.0)
    pitch_range = results.get("pitch_range", 0.0)

    if pitch_mean > 0:
        pitch_stability = 1.0 - min(pitch_range / pitch_mean, 1.0)
    else:
        pitch_stability = 0.0

    pitch_score = pitch_stability * 100

    # --- Engagement ---
    speaking_time = results.get("speaking_time", 0.0)
    total_duration = results.get("total_duration", 0.0)

    engagement = (speaking_time / total_duration) if total_duration > 0 else 0.0
    engagement_score = engagement * 100

    # --- Final weighted score ---
    overall_score = (
        0.40 * semantic_score +
        0.30 * keyword_score +
        0.15 * pitch_score +
        0.15 * engagement_score
    )

    return round(overall_score, 2)


def generate_comprehensive_feedback(results):
    """
    Generate comprehensive qualitative feedback.
    IMPORTANT:
    - This function does NOT compute scores.
    - It ONLY interprets already-computed, research-safe metrics.
    """

    # ------------------------------------------------------------------
    # Overall score MUST be precomputed using independent metrics
    # ------------------------------------------------------------------
    overall_score = float(results.get("overall_score", 0.0))

    # Performance level interpretation
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

    # ------------------------------------------------------------------
    # Detailed feedback (descriptive only, no scoring)
    # ------------------------------------------------------------------
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

    # Pitch-related qualitative feedback (independent interpretation)
    pitch_mean = results.get("pitch_mean", 0.0)
    pitch_range = results.get("pitch_range", 0.0)

    if pitch_mean > 0:
        pitch_variability_ratio = pitch_range / pitch_mean
        if pitch_variability_ratio > 0.6:
            detailed_feedback.append("Pitch varies significantly, which may reduce clarity.")
        elif pitch_variability_ratio < 0.15:
            detailed_feedback.append("Pitch is very flat; adding variation can improve engagement.")

    # Engagement feedback
    speaking_time = results.get("speaking_time", 0.0)
    total_duration = results.get("total_duration", 0.0)

    if total_duration > 0:
        engagement_ratio = speaking_time / total_duration
        if engagement_ratio < 0.5:
            detailed_feedback.append("There are long silent gaps; try to maintain a steadier response.")

    # ------------------------------------------------------------------
    # Improvement suggestions (actionable, capped)
    # ------------------------------------------------------------------
    improvement_suggestions = []

    if semantic_similarity < 0.5:
        improvement_suggestions.append("Focus more directly on answering the question.")

    if keyword_coverage < 0.6:
        improvement_suggestions.append("Include more relevant technical keywords.")

    if pitch_mean > 0 and pitch_range / pitch_mean > 0.5:
        improvement_suggestions.append("Work on maintaining a more consistent pitch.")

    if total_duration > 0 and speaking_time / total_duration < 0.6:
        improvement_suggestions.append("Reduce long pauses to improve flow.")

    # Limit to top 5 suggestions
    improvement_suggestions = improvement_suggestions[:5]

    # ------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------
    return {
        "overall_score": overall_score,
        "performance_level": performance_level,
        "performance_feedback": performance_feedback,
        "detailed_feedback": detailed_feedback,
        "improvement_suggestions": improvement_suggestions,
        "summary": f"Overall Performance: {performance_level} ({overall_score:.1f}/100)"
    }


def analyze_interview_response(audio_file_path, ideal_answer_text, ideal_keywords):
    """
    Comprehensive speech analysis for interview responses.
    Analyzes fluency, pitch, voice quality, and content relevance.
    """
    # Get transcription
    transcribed_text = speech_to_text(audio_file_path)

    # Comprehensive fluency analysis
    fluency_results = analyze_fluency_comprehensive(audio_file_path, transcribed_text)

    # Comprehensive pitch analysis
    pitch_results = analyze_pitch_comprehensive(audio_file_path)

    # Voice quality analysis
    voice_quality_results = analyze_voice_quality(audio_file_path)

    # Content analysis
    filler_count, repetitions = detect_fillers_repetitions(transcribed_text)
    semantic_similarity_score = calculate_semantic_similarity(transcribed_text, ideal_answer_text)
    keyword_coverage_score = calculate_keyword_coverage(transcribed_text, ideal_keywords)

    # Build comprehensive results dictionary
    results = {
        # Transcription
        "transcribed_text": transcribed_text,

        # Fluency metrics
        **fluency_results,

        # Pitch metrics
        **pitch_results,

        # Voice quality metrics
        **voice_quality_results,

        # Content metrics
        "filler_count": filler_count,
        "repetitions": repetitions,
        "semantic_similarity": semantic_similarity_score,
        "keyword_coverage": keyword_coverage_score,

        # Legacy scores for backward compatibility
        "fluency_score": fluency_results.get('fluency_score', 0),
        "clarity_score": clarity_score({
            'semantic_similarity': semantic_similarity_score,
            'keyword_coverage': keyword_coverage_score,
            'filler_count': filler_count
        }, ideal_answer_text, ideal_keywords)
    }

    # Generate comprehensive feedback
    results["overall_score"] = compute_overall_score_independent(results)
    feedback_results = generate_comprehensive_feedback(results)
    results.update(feedback_results)

    return results

def compute_research_metrics(stats: RunningStatistics) -> dict:
    """Computes FINAL interview metrics with proper validation."""
    
    final_stats = stats.get_current_stats()
    
    total_words = final_stats["total_words"]
    speaking_time = final_stats["speaking_time"]
    total_duration = final_stats["total_duration"]
    
    # 🔥 VALIDATION: Ensure durations make sense
    if total_duration <= 0:
        return {
            "wpm": 0,
            "speaking_time_ratio": 0,
            "pause_ratio": 1.0,
            "long_pause_count": 0,
            "filler_frequency_per_min": 0,
            "avg_semantic_similarity": 0,
            "avg_keyword_coverage": 0,
            "overall_relevance": 0,
            "data_quality": "INVALID_DURATION"
        }
    
    # 1️⃣ Words Per Minute (WPM)
    if speaking_time > 1.0:
        wpm = (total_words / speaking_time) * 60
    else:
        wpm = 0
    
    # 2️⃣ Speaking Time Ratio (0-1)
    speaking_time_ratio = speaking_time / total_duration
    speaking_time_ratio = max(0.0, min(1.0, speaking_time_ratio))
    
    # 3️⃣ Pause Ratio (0-1)
    pause_ratio = 1.0 - speaking_time_ratio
    
    # 4️⃣ Long Pause Count (5s threshold)
    long_pause_count = stats.long_pause_count
    
    # 🔥 NEW: Get Q&A metrics
    avg_semantic = final_stats.get('avg_semantic_similarity', 0)
    avg_keyword = final_stats.get('avg_keyword_coverage', 0)
    question_count = final_stats.get('question_count', 0)
    
   # 🔥 NEW: Reduce keyword weight to 20% of overall
    overall_relevance = (avg_semantic * 0.8) + (avg_keyword * 0.2)
    
    # 5️⃣ Filler Frequency (per minute)
    total_fillers = sum(stats.filler_counts.values())
    if speaking_time > 1.0:
        filler_frequency_per_min = total_fillers / (speaking_time / 60)
    else:
        filler_frequency_per_min = 0
    
    # 🔥 SANITY CHECK: Log impossible values
    print(f"\n📊 FINAL INTERVIEW METRICS:")
    print(f"   Questions answered: {question_count}")
    print(f"   Avg Semantic Similarity: {avg_semantic:.3f}")
    print(f"   Avg Keyword Coverage: {avg_keyword:.3f}")
    print(f"   Overall Relevance: {overall_relevance:.3f}")
    print(f"   Total Duration: {total_duration:.1f}s")
    print(f"   Speaking Time: {speaking_time:.1f}s")
    print(f"   Speaking Ratio: {speaking_time_ratio:.3f}")
    print(f"   Pause Ratio: {pause_ratio:.3f}")
    print(f"   Long Pauses: {long_pause_count}")
    print(f"   WPM: {wpm:.1f}")
    
    return {
        "wpm": round(wpm, 2),
        "speaking_time_ratio": round(speaking_time_ratio, 3),
        "pause_ratio": round(pause_ratio, 3),
        "long_pause_count": long_pause_count,
        "filler_frequency_per_min": round(filler_frequency_per_min, 2),
        "avg_semantic_similarity": round(avg_semantic, 3),
        "avg_keyword_coverage": round(avg_keyword, 3),
        "overall_relevance": round(overall_relevance, 3),
        "questions_answered": question_count,
        "data_quality": "VALID" if speaking_time_ratio < 0.95 else "QUESTIONABLE_HIGH_SPEECH"
    }

def finalize_interview(stats: RunningStatistics,
                       user_answer: str,
                       expected_answer: str) -> dict:
    """
    FINAL research-safe output.
    Called ONLY on clean interview end.
    """

    metrics = compute_research_metrics(stats)

    # 🔒 VALIDITY GATE (CRITICAL)
    analysis_valid = True

    if stats.speaking_time < 3 or stats.total_words < 5:
        analysis_valid = False

    # 🔥 Use the pre-calculated Q&A scores from stats
    # (No need to calculate again - they're already stored)

    return {
        "metrics": metrics,
        "semantic_similarity": metrics.get('avg_semantic_similarity', 0),  # Average across all questions
        "keyword_coverage": metrics.get('avg_keyword_coverage', 0),
        "overall_relevance": metrics.get('overall_relevance', 0),
        "questions_answered": metrics.get('questions_answered', 0),
        "analysis_valid": analysis_valid
    }

if __name__ == '__main__':
    print("Interview Analyzer module loaded and ready.")