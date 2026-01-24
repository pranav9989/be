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
        self.transcript_parts = []

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

    def update_transcript(self, new_text):
        """Update transcription statistics incrementally."""
        if new_text:
            self.transcript_parts.append(new_text)
            words_in_segment = len(new_text.split())
            self.total_words += words_in_segment

    def update_time_stats(self, segment_duration, speaking_duration):
        """Update time-based statistics."""
        self.total_duration += segment_duration
        self.speaking_time += speaking_duration

    def update_pause_stats(self, pause_durations):
        """Update pause statistics."""
        for pause in pause_durations:
            self.pause_durations.append(pause)
            if pause > 1.0:  # Long pause threshold
                self.long_pause_count += 1

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

    def get_current_stats(self):
        """Get current computed statistics."""
        # Calculate derived statistics
        transcript = " ".join(self.transcript_parts)

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

        return {
            'transcript': transcript,
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
            'avg_hnr': avg_hnr
        }


def analyze_audio_chunk_fast(pcm_chunk, sample_rate, stats: RunningStatistics):
    """
    Fast incremental analysis for a single PCM chunk (≈5 sec)
    This replaces the heavy analysis and runs in milliseconds.

    Args:
        pcm_chunk: Numpy array of int16 PCM samples
        sample_rate: Sample rate (typically 16000)
        stats: RunningStatistics object to update incrementally
    """
    # Convert int16 PCM → float32 for librosa
    audio = pcm_chunk.astype(np.float32) / 32768.0
    duration = len(audio) / sample_rate

    # --- Silence detection ---
    intervals = librosa.effects.split(audio, top_db=25)
    speaking_time = sum((e - s) / sample_rate for s, e in intervals)

    stats.update_time_stats(duration, speaking_time)

    # --- Pause stats ---
    pauses = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            pause = (intervals[i+1][0] - intervals[i][1]) / sample_rate
            if pause > 0.1:  # Only count pauses > 100ms
                pauses.append(pause)
    stats.update_pause_stats(pauses)

    # --- FAST pitch analysis (YIN, not pYIN) ---
    try:
        # Downsample to 8kHz for efficiency (sufficient for pitch detection)
        audio_8k = librosa.resample(audio, orig_sr=sample_rate, target_sr=8000)

        # Use YIN algorithm (much faster than pYIN)
        f0 = librosa.yin(
            audio_8k,
            fmin=65,    # Male speech range
            fmax=300,   # Female speech range
            sr=8000,
            frame_length=512,   # Smaller frames for speed
            hop_length=128      # Faster hop
        )

        # Filter out NaN/infinite values
        f0 = f0[np.isfinite(f0)]
        if len(f0) > 0:
            stats.update_pitch_stats(f0)
    except Exception as e:
        print(f"Fast pitch analysis failed: {e}")

    # --- Lightweight voice quality ---
    try:
        # Simple RMS-based shimmer approximation
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        if len(rms) > 1:
            shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
            # Placeholder values for jitter and HNR (could be improved)
            jitter = 0  # Would need proper calculation
            hnr = 10    # Approximate good HNR
            stats.update_voice_quality(jitter=jitter, shimmer=shimmer, hnr=hnr)
    except Exception as e:
        print(f"Fast voice quality analysis failed: {e}")

def analyze_fluency_comprehensive(audio_path, transcribed_text):
    """
    Comprehensive fluency analysis including WPM, pause patterns, articulation rate, and speech flow.
    Returns: dict with detailed fluency metrics
    """
    y, sr = librosa.load(audio_path)

    # Silence detection - split returns list of (start, end) tuples in samples
    intervals = librosa.effects.split(y, top_db=20)

    # Calculate speaking time and total duration
    speaking_time = sum((end - start) / sr for start, end in intervals)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Calculate pause ratio
    pause_ratio = max(0, (total_duration - speaking_time) / total_duration) if total_duration > 0 else 1.0

    # Calculate words per minute (WPM)
    words = len(transcribed_text.split()) if transcribed_text else 0
    wpm = (words / speaking_time * 60) if speaking_time > 3 else 0

    # Analyze pause patterns
    pause_durations = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            current_end = intervals[i][1] / sr
            next_start = intervals[i + 1][0] / sr
            pause_duration = next_start - current_end
            if pause_duration > 0.1:  # Only count pauses longer than 100ms
                pause_durations.append(pause_duration)

    # Pause statistics
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
    max_pause_duration = max(pause_durations) if pause_durations else 0
    num_long_pauses = len([p for p in pause_durations if p > 1.0])  # Pauses > 1 second

    # Articulation rate (syllables per second during speaking time)
    # Rough syllable estimation: ~1.5 syllables per word for English
    estimated_syllables = words * 1.5
    articulation_rate = estimated_syllables / speaking_time if speaking_time > 0 else 0

    # Speech efficiency (ratio of speaking time to total time)
    speech_efficiency = speaking_time / total_duration if total_duration > 0 else 0

    # Fluency scores
    # WPM score (ideal range: 120-160 WPM)
    if 120 <= wpm <= 160:
        wpm_score = 100
    elif 100 <= wpm <= 180:
        wpm_score = 80
    elif 80 <= wpm <= 200:
        wpm_score = 60
    else:
        wpm_score = max(0, 100 - abs(140 - wpm) * 0.5)

    # Pause score (lower pause ratio is better, but some pauses are natural)
    if pause_ratio < 0.3:  # Less than 30% pauses
        pause_score = 100
    elif pause_ratio < 0.5:
        pause_score = 80
    elif pause_ratio < 0.7:
        pause_score = 60
    else:
        pause_score = max(20, 100 - (pause_ratio - 0.3) * 200)

    # Articulation score (ideal: 4-6 syllables per second)
    if 4 <= articulation_rate <= 6:
        articulation_score = 100
    elif 3 <= articulation_rate <= 7:
        articulation_score = 80
    elif 2 <= articulation_rate <= 8:
        articulation_score = 60
    else:
        articulation_score = max(20, 100 - abs(5 - articulation_rate) * 10)

    # Overall fluency score
    fluency_score = (wpm_score * 0.4 + pause_score * 0.3 + articulation_score * 0.3)

    # Generate feedback
    feedback_parts = []
    if wpm < 100:
        feedback_parts.append("speaking rate is slow")
    elif wpm > 180:
        feedback_parts.append("speaking rate is fast")

    if pause_ratio > 0.5:
        feedback_parts.append("too many pauses")
    elif pause_ratio < 0.1:
        feedback_parts.append("minimal natural pauses")

    if articulation_rate < 3:
        feedback_parts.append("articulation is slow and deliberate")
    elif articulation_rate > 7:
        feedback_parts.append("speech is rushed")

    if num_long_pauses > 3:
        feedback_parts.append("several long pauses detected")

    fluency_feedback = "Fluency analysis: " + "; ".join(feedback_parts) if feedback_parts else "Speech fluency is good"

    return {
        "wpm": float(wpm),
        "pause_ratio": float(pause_ratio),
        "speaking_time": float(speaking_time),
        "total_duration": float(total_duration),
        "avg_pause_duration": float(avg_pause_duration),
        "max_pause_duration": float(max_pause_duration),
        "num_long_pauses": int(num_long_pauses),
        "articulation_rate": float(articulation_rate),
        "speech_efficiency": float(speech_efficiency),
        "wpm_score": float(wpm_score),
        "pause_score": float(pause_score),
        "articulation_score": float(articulation_score),
        "fluency_score": float(fluency_score),
        "fluency_feedback": fluency_feedback
    }

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

def calculate_semantic_similarity(user_answer, ideal_answer):
    if not user_answer or not ideal_answer:
        return 0.0
    user_embedding = embedder.encode([user_answer], normalize_embeddings=True)[0]
    ideal_embedding = embedder.encode([ideal_answer], normalize_embeddings=True)[0]
    
    similarity = cosine_similarity([user_embedding], [ideal_embedding])[0][0]
    return float(similarity)

def calculate_keyword_coverage(user_answer, ideal_keywords):
    if not user_answer or not ideal_keywords:
        return 0.0
    doc = nlp(user_answer.lower())
    
    found_keywords = []
    for keyword in ideal_keywords:
        if keyword.lower() in doc.text:
            found_keywords.append(keyword)
            
    if not ideal_keywords:
        return 0.0 # Avoid division by zero if there are no ideal keywords
            
    coverage = len(found_keywords) / len(ideal_keywords)
    return coverage

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

if __name__ == '__main__':
    # Example usage (you'll need an audio file for this)
    # from pydub import AudioSegment
    # from pydub.playback import play
    
    # # Create a dummy audio file for testing
    # # This part requires pydub and ffmpeg/ffprobe installed
    # # pip install pydub
    # # You might also need to install ffmpeg/ffprobe
    # # e.g., on Windows: choco install ffmpeg
    # # on Mac: brew install ffmpeg
    
    # print("Creating a dummy audio file for testing...")
    # audio = AudioSegment.silent(duration=1000)  # 1 second of silence
    # audio = audio.append(AudioSegment.sine(440, duration=2000), crossfade=0) # 2 seconds of tone
    # audio.export("dummy_audio.wav", format="wav")
    # print("Dummy audio file 'dummy_audio.wav' created.")
    
    # audio_file_path = "dummy_audio.wav"
    # ideal_answer = "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms, including object-oriented, imperative, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation."
    # ideal_keywords = ["Python", "interpreted", "high-level", "versatility", "web development", "data science", "artificial intelligence", "automation", "object-oriented"]

    # # Ensure the dummy audio file exists before analysis
    # if os.path.exists(audio_file_path):
    #     print(f"Analyzing {audio_file_path}...")
    #     results = analyze_interview_response(audio_file_path, ideal_answer, ideal_keywords)
    #     for key, value in results.items():
    #         if isinstance(value, float):
    #             print(f"{key}: {value:.2f}")
    #         else:
    #             print(f"{key}: {value}")
    # else:
    #     print(f"Test audio file '{audio_file_path}' not found. Skipping example analysis.")
    
    print("Interview Analyzer module loaded and ready.")
