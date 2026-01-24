
# AI-Powered Interview Preparation Platform
## Technical Implementation Report

---

### **1. System Overview**
This report documents the actual technical implementation of the Interview Preparation Platform. The system is a hybrid AI application capable of both **static document analysis** (Resume RAG) and **real-time audio processing** (Live Interviews).

It is built as a **Client-Server application**:
*   **Frontend**: React.js (User Interface & Audio Capture)
*   **Backend**: Python Flask (Signal Processing, AI Orchestration, WebSocket Server)

---

### **2. Implemented Technologies**

#### **2.1 Core AI & Machine Learning**
We have implemented a multi-model AI system where different specific tasks are handled by the most efficient model available:

1.  **Google Gemini (LLM)**:
    *   **Role**: The "Reasoning Engine".
    *   **Implementation**: Used in `rag.py` and `app.py`.
    *   **Function**: It generates context-aware interview questions and provides the final qualitative feedback on user answers.

2.  **Sentence Transformers (`all-MiniLM-L6-v2`)**:
    *   **Role**: The "Matchmaker".
    *   **Implementation**: Used in `interview_analyzer.py` and `resume_processor.py`.
    *   **Function**: It converts text (Resume descriptors + Job Descriptions) into mathematical vectors to calculate how "similar" they are. This drives the **Semantic Similarity Score**.

3.  **Faster-Whisper (Local ASR)**:
    *   **Role**: Local Speech-to-Text Engine.
    *   **Implementation**: Loaded via `WhisperModelManager` in `interview_analyzer.py`.
    *   **Function**: Provides offline or fallback transcription capabilities directly on the server without needing external APIs.

4.  **AssemblyAI**:
    *   **Role**: Real-time Streaming Transcription.
    *   **Implementation**: Integrated via `AssemblyAIWebSocketStreamer` in `app.py`.
    *   **Function**: Converts live audio streams into text with low latency during the interview.

#### **2.2 Signal Processing (The "Physics" Layer)**
Unlike standard chatbots, our system implements "Hard Science" signal processing to analyze *how* the user speaks, not just *what* they say. This code is located in `interview_analyzer.py`.

*   **Pitch Detection (YIN Algorithm)**: We use `librosa.yin` to track the fundamental frequency (F0) of the user's voice. This allows us to calculate **Pitch Stability** and **Range**.
*   **Voice Quality (Shimmer/Jitter)**: We implement statistical analysis on raw PCM audio waves to detect amplitude perturbations (Shimmer), giving a "Confidence Score" based on voice steadiness.
*   **Welford’s Algorithm**: Implemented in the `RunningStatistics` class. This allows us to calculate statistical mean and variance on a live data stream efficiently, without storing hours of raw audio in memory.

---

### **3. Detailed Architecture Flow**

#### **3.1 The Resume Analysis Engine (RAG Pipeline)**
This workflow is fully implemented to personalize the experience.
1.  **Ingestion**: User uploads a PDF/DOCX.
2.  **Extraction**: `PyPDF2` extracts raw text.
3.  **Vectorization**: `SentenceTransformer` converts resume text into vectors.
4.  **Indexing**: **FAISS** (Facebook AI Similarity Search) creates a searchable index of the user's skills.
5.  **Retrieval**: When Gemini generates a question, it first "queries" this FAISS index to understand the user's specific project history.

#### **3.2 The Live Interview Engine (Hybrid Processing)**
This is the most complex implemented feature, using a **Parallel Processing Architecture**.

**Step 1: Audio Capture (Frontend)**
*   The React app uses the browser's `MediaRecorder API`.
*   It slices microphone input into **4096-byte PCM chunks**.
*   These chunks are emitted via **Socket.IO** to the backend every ~100ms.

**Step 2: The Fork (Backend)**
Upon receiving an audio packet in `app.py`, the server splits the data into two parallel streams:

*   **Stream A: The Signal Stream (Local Processing)**
    *   The raw bytes are converted to a **NumPy array**.
    *   `analyze_audio_chunk_fast()` executes immediately.
    *   It calculates volume (RMS) and Pitch (YIN) in milliseconds.
    *   These metrics update the `RunningStatistics` object instantly.

*   **Stream B: The Semantic Stream (External Processing)**
    *   The same bytes are forwarded to **AssemblyAI**.
    *   AssemblyAI returns text transcripts asynchronously.
    *   The backend pushes this text back to the Frontend for the "Live Captioning" effect.

**Step 3: Final Synthesis**
When the user clicks "Stop":
1.  We combine the **Signal Metrics** (WPM, Pitch Stability) from Stream A.
2.  We take the **Full Transcript** from Stream B.
3.  We pass both to **Gemini**.
4.  Gemini generates a human-readable feedback report (e.g., "You sounded confident (stable pitch), but your answer lacked technical depth regarding React Hooks.").

---

### **4. Summary of Key Metrics Implemented**

The system currently calculates the following metrics based on actual code:
*   **WPM (Words Per Minute)**: Calculated dynamically during speech.
*   **Pause Ratio**: Percentage of time spent silent vs. speaking.
*   **Pitch Stability**: Coefficient of variation in voice frequency.
*   **Semantic Similarity**: How closely the answer matches an "Ideal Answer" vector.
*   **Keyword Coverage**: Percentage of required technical terms (e.g., "React", "Database") mentions in the answer.

### **5. Conclusion**
The implemented system serves as a functional MVP (Minimum Viable Product) for a high-end interview coach. It successfully integrates three distinct fields of engineering: **Web Development** (React/Flask), **Signal Processing** (Librosa/NumPy), and **Generative AI** (Gemini/RAG).
