// hooks/useResumeInterviewStreaming.js
import { useState, useRef, useEffect, useCallback } from 'react';
import io from 'socket.io-client';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const useResumeInterviewStreaming = (userId) => {
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [liveTranscript, setLiveTranscript] = useState('');
    const [finalTranscript, setFinalTranscript] = useState('');
    const [interviewDone, setInterviewDone] = useState(false);
    const [analysis, setAnalysis] = useState(null);
    const [error, setError] = useState(null);
    const [status, setStatus] = useState('Ready');
    const [timeRemaining, setTimeRemaining] = useState(30 * 60);
    const [messages, setMessages] = useState([]);
    const [currentTurn, setCurrentTurn] = useState('INTERVIEWER');
    const [metrics, setMetrics] = useState(null);
    const [sessionId, setSessionId] = useState(null);
    const [coachingFeedback, setCoachingFeedback] = useState(null);
    const [isInterviewerSpeaking, setIsInterviewerSpeaking] = useState(false);
    const [isFinalizing, setIsFinalizing] = useState(false);

    // 🔥 Live metrics for real-time display - MATCH AGENTIC INTERVIEW
    const [liveWpm, setLiveWpm] = useState(0);
    const [livePitch, setLivePitch] = useState({ mean: 0, stability: 0, range: 0 });
    const [pitchHistory, setPitchHistory] = useState([]);
    const [pitchTimestamps, setPitchTimestamps] = useState([]);
    const [stabilityHistory, setStabilityHistory] = useState([]);

    const socketRef = useRef(null);
    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const streamRef = useRef(null);
    const startTimeRef = useRef(null);
    const backendReadyRef = useRef(false);
    const pendingAudioRef = useRef([]);
    const audioPlayingRef = useRef(false);
    const currentAudioRef = useRef(null);
    const hardStopRef = useRef(false);
    const userTurnStartRef = useRef(null);
    const lastWpmUpdateRef = useRef(0);
    const lastPitchUpdateRef = useRef(0);

    // 🔥 Refs to avoid stale closures
    const currentTurnRef = useRef(currentTurn);
    const interviewDoneRef = useRef(interviewDone);
    const isInterviewerSpeakingRef = useRef(isInterviewerSpeaking);
    const isFinalizingRef = useRef(isFinalizing);

    // Sync refs with state
    useEffect(() => {
        currentTurnRef.current = currentTurn;
    }, [currentTurn]);

    useEffect(() => {
        interviewDoneRef.current = interviewDone;
    }, [interviewDone]);

    useEffect(() => {
        isInterviewerSpeakingRef.current = isInterviewerSpeaking;
    }, [isInterviewerSpeaking]);

    useEffect(() => {
        isFinalizingRef.current = isFinalizing;
    }, [isFinalizing]);

    const cleanupAudio = useCallback(async () => {
        console.log('🧹 Cleaning up audio resources...');

        if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current = null;
        }
        audioPlayingRef.current = false;
        setIsInterviewerSpeaking(false);

        if (processorRef.current) {
            processorRef.current.disconnect();
            processorRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (audioContextRef.current) {
            await audioContextRef.current.close();
            audioContextRef.current = null;
        }

        backendReadyRef.current = false;
        pendingAudioRef.current = [];
    }, []);

    const floatTo16BitPCM = useCallback((float32Array) => {
        if (!float32Array || float32Array.length === 0) return new ArrayBuffer(0);
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        let offset = 0;
        for (let i = 0; i < float32Array.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return buffer;
    }, []);

    const speakWithMurf = useCallback(async (text) => {
        if (hardStopRef.current || interviewDoneRef.current) {
            console.log('🚫 TTS blocked');
            return;
        }
        if (!text) return;

        try {
            audioPlayingRef.current = true;
            setIsInterviewerSpeaking(true);
            setStatus("🗣️ Interviewer speaking...");

            const res = await fetch(`${API_BASE_URL}/api/tts/murf`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text.substring(0, 1000), username: userId.toString() })
            });

            if (!res.ok) throw new Error(`TTS failed: ${res.status}`);

            const audioBlob = await res.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            currentAudioRef.current = audio;

            return new Promise((resolve) => {
                audio.onended = () => {
                    audioPlayingRef.current = false;
                    setIsInterviewerSpeaking(false);
                    currentAudioRef.current = null;
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                };
                audio.onerror = (err) => {
                    console.error("Audio error:", err);
                    audioPlayingRef.current = false;
                    setIsInterviewerSpeaking(false);
                    currentAudioRef.current = null;
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                };
                audio.play().catch(err => {
                    console.error("Playback failed:", err);
                    audioPlayingRef.current = false;
                    setIsInterviewerSpeaking(false);
                    currentAudioRef.current = null;
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                });
            });
        } catch (err) {
            console.error("TTS Error:", err);
            audioPlayingRef.current = false;
            setIsInterviewerSpeaking(false);
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }, [userId]);

    // 🔥 MANUAL SUBMIT - like agentic flow
    const submitAnswer = useCallback(() => {
        if (!socketRef.current?.connected) {
            console.log("❌ Socket not connected");
            return;
        }

        if (currentTurnRef.current !== 'USER') {
            console.log("❌ Not user turn, cannot submit");
            return;
        }

        if (isFinalizingRef.current) {
            console.log("❌ Already finalizing, please wait");
            return;
        }

        console.log("📤 Resume: submitting answer manually");
        setIsFinalizing(true);
        setStatus('⏳ Processing your answer...');

        socketRef.current.emit("user_done_speaking", {
            user_id: userId
        });
    }, [userId]);

    const stopRecording = useCallback(async () => {
        if (hardStopRef.current) return;
        console.log('🛑 Ending interview');
        hardStopRef.current = true;
        setInterviewDone(true);
        setIsRecording(false);
        setCurrentTurn('DONE');
        setStatus('🛑 Interview ended');
        setTimeRemaining(0);

        await cleanupAudio();
        if (socketRef.current?.connected) {
            socketRef.current.emit('stop_resume_interview', { user_id: userId });
        }
    }, [userId, cleanupAudio]);

    const startRecording = useCallback(async () => {
        try {
            setError(null);
            setAnalysis(null);
            setCoachingFeedback(null);
            setSessionId(null);
            setLiveTranscript('');
            setFinalTranscript('');
            setInterviewDone(false);
            setStatus('🎤 Starting interview...');
            setCurrentTurn('INTERVIEWER');
            setTimeRemaining(30 * 60);
            setIsRecording(true);
            setMessages([]);
            setLiveWpm(0);
            setLivePitch({ mean: 0, stability: 0, range: 0 });
            setPitchHistory([]);
            setPitchTimestamps([]);
            setStabilityHistory([]);
            setMetrics(null);
            setIsInterviewerSpeaking(false);
            setIsFinalizing(false);
            hardStopRef.current = false;
            lastWpmUpdateRef.current = 0;
            lastPitchUpdateRef.current = 0;

            await cleanupAudio();
            socketRef.current.emit('start_resume_interview', {
                user_id: userId,
                job_description: ''
            });
        } catch (err) {
            console.error('Error starting interview:', err);
            setError('Failed to start: ' + err.message);
            setStatus('❌ Failed to start');
        }
    }, [userId, cleanupAudio]);

    // Update WPM calculation
    const updateWPM = useCallback((text) => {
        if (userTurnStartRef.current && text && text.trim()) {
            const words = text.trim().split(/\s+/).length;
            const elapsed = (Date.now() - userTurnStartRef.current) / 1000;
            const minutes = elapsed / 60;
            const wpm = minutes > 0 ? Math.round(words / minutes) : 0;

            if (Math.abs(wpm - lastWpmUpdateRef.current) > 5) {
                lastWpmUpdateRef.current = wpm;
                setLiveWpm(wpm);
            }
        }
    }, []);

    // Initialize WebSocket connection
    useEffect(() => {
        socketRef.current = io(API_BASE_URL, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: 5
        });

        socketRef.current.on('connect', () => {
            setIsConnected(true);
            setError(null);
            console.log('✅ Connected to WebSocket');
        });

        socketRef.current.on('disconnect', () => {
            setIsConnected(false);
            setStatus('Disconnected');
            console.log('❌ Disconnected from WebSocket');
        });

        // 🔥 WPM updates from backend
        socketRef.current.on('wpm_update', (data) => {
            setLiveWpm(data.wpm);
            lastWpmUpdateRef.current = data.wpm;
        });

        // 🔥 Pitch updates from backend
        socketRef.current.on('pitch_update', (data) => {
            const pitchData = {
                mean: data.current_pitch || data.mean || 0,
                stability: data.stability || 0,
                range: data.range || 0
            };
            setLivePitch(pitchData);

            if (pitchData.mean > 0) {
                setPitchHistory(prev => {
                    const newHistory = [...prev, pitchData.mean];
                    if (newHistory.length > 30) newHistory.shift();
                    return newHistory;
                });

                setStabilityHistory(prev => {
                    const newHistory = [...prev, pitchData.stability];
                    if (newHistory.length > 30) newHistory.shift();
                    return newHistory;
                });

                setPitchTimestamps(prev => {
                    const newTimes = [...prev, new Date().toLocaleTimeString()];
                    if (newTimes.length > 30) newTimes.shift();
                    return newTimes;
                });
            }
        });

        // 🔥 Backend ready signal
        socketRef.current.on('resume_backend_ready', () => {
            console.log('✅ BACKEND READY - sending audio');
            backendReadyRef.current = true;

            if (pendingAudioRef.current.length > 0 && currentTurnRef.current === 'USER') {
                console.log(`📤 Flushing ${pendingAudioRef.current.length} buffered chunks`);
                pendingAudioRef.current.forEach((bufferedChunk) => {
                    socketRef.current.emit('audio_chunk', {
                        user_id: userId,
                        audio: bufferedChunk
                    });
                });
                pendingAudioRef.current = [];
            }

            if (currentTurnRef.current === 'USER') {
                setStatus('🎤 Speak now...');
            }
        });

        // 🔥 Force stop speaking (when answer is finalized)
        socketRef.current.on("force_stop_speaking", () => {
            console.log("🛑 Force stop speaking received");
            setIsFinalizing(true);
            setCurrentTurn('INTERVIEWER');
            setStatus('⏳ Processing your answer...');
        });

        // 🔥 User answer complete (backend done processing)
        socketRef.current.on("user_answer_complete", (data) => {
            console.log("✅ Resume answer complete:", data.answer);
            setIsFinalizing(false);

            setMessages(prev => [
                ...prev,
                { role: "user", text: data.answer },
                ...(data.gold_answer ? [{
                    role: "gold",
                    text: data.gold_answer
                }] : [])
            ]);

            setLiveTranscript('');
            setCurrentTurn('INTERVIEWER');
        });

        // 🔥 Interview started
        socketRef.current.on('resume_interview_started', async (data) => {
            console.log('🎤 Resume interview started');
            setIsRecording(true);
            setInterviewDone(false);
            setLiveTranscript('');
            setFinalTranscript('');
            startTimeRef.current = Date.now();
            setCurrentTurn('INTERVIEWER');
            setIsFinalizing(false);
            backendReadyRef.current = false;
            pendingAudioRef.current = [];

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                });
                streamRef.current = stream;

                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                await audioContextRef.current.audioWorklet.addModule('/pcm-processor.js');

                processorRef.current = new AudioWorkletNode(audioContextRef.current, 'pcm-processor');

                processorRef.current.port.onmessage = (event) => {
                    const float32Samples = event.data;
                    const pcmBuffer = floatTo16BitPCM(float32Samples);

                    if (hardStopRef.current || interviewDoneRef.current) return;
                    // 🔥 MODIFIED: Check finalizing ref before sending
                    if (currentTurnRef.current !== 'USER') return;
                    if (isFinalizingRef.current) return;

                    if (!backendReadyRef.current) {
                        if (pendingAudioRef.current.length < 200) {
                            pendingAudioRef.current.push(pcmBuffer);
                        }
                        return;
                    }

                    if (socketRef.current?.connected) {
                        socketRef.current.emit('audio_chunk', {
                            user_id: userId,
                            audio: pcmBuffer
                        });
                    }
                };

                const source = audioContextRef.current.createMediaStreamSource(stream);
                source.connect(processorRef.current);
                const silentGain = audioContextRef.current.createGain();
                silentGain.gain.value = 0;
                processorRef.current.connect(silentGain);
                silentGain.connect(audioContextRef.current.destination);

                console.log('✅ Audio pipeline ready');

            } catch (err) {
                console.error('Microphone error:', err);
                setError('Microphone access failed: ' + err.message);
                setStatus('❌ Microphone error');
                cleanupAudio();
            }
        });

        // 🔥 First question
        socketRef.current.on('resume_question', async (data) => {
            if (hardStopRef.current || interviewDoneRef.current) return;
            console.log('🗣️ Question:', data.question);

            // 🔥 Reset finalizing state for new user turn
            setIsFinalizing(false);
            setCurrentTurn('INTERVIEWER');
            setStatus("🗣️ Interviewer speaking...");
            setLiveTranscript('');
            setMessages(prev => [...prev, { role: "interviewer", text: data.question }]);
            await speakWithMurf(data.question);

            if (socketRef.current?.connected) {
                socketRef.current.emit("interviewer_done", { user_id: userId });
            }

            setCurrentTurn('USER');
            setStatus('🎤 Your turn... Click Submit when done');
            userTurnStartRef.current = Date.now();
        });

        // 🔥 Next question
        socketRef.current.on('resume_next_question', async (data) => {
            if (hardStopRef.current || interviewDoneRef.current) return;
            console.log('🗣️ Next question:', data.question);

            // 🔥 Reset finalizing state for new user turn
            setIsFinalizing(false);
            setCurrentTurn('INTERVIEWER');
            setStatus("🗣️ Interviewer speaking...");
            setLiveTranscript('');
            setMessages(prev => [...prev, { role: "interviewer", text: data.question }]);
            await speakWithMurf(data.question);

            if (socketRef.current?.connected) {
                socketRef.current.emit("interviewer_done", { user_id: userId });
            }

            setCurrentTurn('USER');
            setStatus('🎤 Your turn... Click Submit when done');
            userTurnStartRef.current = Date.now();
        });

        // 🔥 Live transcript
        socketRef.current.on('live_transcript', (data) => {
            if (currentTurnRef.current === 'USER' && !isFinalizingRef.current) {
                setLiveTranscript(data.text);
                updateWPM(data.text);
            }
        });

        // 🔥 Timer update
        socketRef.current.on('timer_update', (data) => {
            if (data.time_remaining !== undefined) {
                setTimeRemaining(data.time_remaining);
            }
        });

        // 🔥 Interview complete with full metrics and coaching
        socketRef.current.on('resume_interview_complete', (data) => {
            console.log('🎯 Interview complete:', data);
            setInterviewDone(true);
            setIsRecording(false);
            setCurrentTurn('DONE');
            setStatus('✅ Completed');
            setIsFinalizing(false);

            if (data.metrics) {
                setMetrics(data.metrics);
                console.log('📊 Metrics received:', data.metrics);
            }
            if (data.session_id) {
                setSessionId(data.session_id);
            }
            if (data.coaching_feedback) {
                setCoachingFeedback(data.coaching_feedback);
                console.log('📝 Coaching feedback received');
            }
            if (data.success) {
                setAnalysis(data);
            }
            if (data.final_transcript) {
                setFinalTranscript(data.final_transcript);
            }

            cleanupAudio();
        });

        // 🔥 Error handler
        socketRef.current.on('interview_error', (data) => {
            console.error('Error:', data.error);
            setError(data.error);
            setIsRecording(false);
            setStatus('❌ Error');
            setIsFinalizing(false);

            cleanupAudio();
        });

        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            cleanupAudio();
        };
    }, [userId, cleanupAudio, floatTo16BitPCM, speakWithMurf, updateWPM]);

    const formatTime = (seconds) => {
        if (typeof seconds !== 'number' || !isFinite(seconds) || seconds <= 0) return '00:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return {
        // Core state
        isConnected,
        isRecording,
        liveTranscript,
        finalTranscript,
        interviewDone,
        analysis,
        error,
        status,
        timeRemaining: formatTime(timeRemaining),
        currentTurn,
        messages,

        // Metrics and results
        metrics,
        sessionId,
        coachingFeedback,

        // Live metrics for real-time display
        liveWpm,
        livePitch,
        pitchHistory,
        pitchTimestamps,
        stabilityHistory,

        // CRITICAL: isInterviewerSpeaking flag
        isInterviewerSpeaking,

        // 🔥 MANUAL SUBMIT ACTION - EXPORTED
        submitAnswer,

        // Actions
        startRecording,
        stopRecording,
    };
};