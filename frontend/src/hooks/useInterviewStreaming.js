import { useState, useRef, useEffect, useCallback } from 'react';
import io from 'socket.io-client';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const useInterviewStreaming = (userId) => {
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [liveTranscript, setLiveTranscript] = useState(''); // Real-time partial during interview
    const [finalTranscript, setFinalTranscript] = useState(''); // Full transcript after completion
    const [interviewDone, setInterviewDone] = useState(false);
    const [wpm, setWpm] = useState(0);
    const [analysis, setAnalysis] = useState(null);
    const [error, setError] = useState(null);
    const [useMock, setUseMock] = useState(false);
    const [isFinalizing, setIsFinalizing] = useState(false);
    const [status, setStatus] = useState('Ready'); // 🔥 Status for user feedback
    const [timeRemaining, setTimeRemaining] = useState(30 * 60);
    const [messages, setMessages] = useState([]);
    const [currentTurn, setCurrentTurn] = useState('INTERVIEWER');
    const [metrics, setMetrics] = useState(null);
    const [sessionPlan, setSessionPlan] = useState(null);
    const [livePitch, setLivePitch] = useState({ mean: 0, stability: 0, range: 0 });
    const [pitchHistory, setPitchHistory] = useState([]); // Store last 30 values for graph
    const [pitchTimestamps, setPitchTimestamps] = useState([]); // Timestamps for graph
    const [stabilityHistory, setStabilityHistory] = useState([]); // Store last 30 stability values

    const socketRef = useRef(null);
    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const streamRef = useRef(null);
    const startTimeRef = useRef(null);
    const backendReadyRef = useRef(false); // Track if backend is ready
    const pendingAudioRef = useRef([]); // Buffer audio before backend is ready
    const isFirstInterviewRef = useRef(true); // 🔥 Track if first interview
    const audioPlayingRef = useRef(false); // Track if TTS audio is playing
    const currentAudioRef = useRef(null); // Store current audio element
    const audioChunksSentRef = useRef(0);
    const hardStopRef = useRef(false); // 🔥 MASTER KILL SWITCH

    // 🔧 FIX 1: Add refs for stale closure issue
    const currentTurnRef = useRef(currentTurn);
    const isFinalizingRef = useRef(isFinalizing);
    const interviewDoneRef = useRef(interviewDone); // 🔥 ADD THIS REF

    // Helper function to clean up audio resources
    const cleanupAudio = useCallback(async () => {
        console.log('🧹 Cleaning up audio resources...');

        // Stop any playing audio
        if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current = null;
        }
        audioPlayingRef.current = false;

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

    // Convert Float32Array to 16-bit PCM (required by AssemblyAI)
    const floatTo16BitPCM = useCallback((float32Array) => {
        if (!float32Array || float32Array.length === 0) {
            return new ArrayBuffer(0);
        }

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
            console.log('🚫 TTS blocked (interview ended)');
            return;
        }

        if (!text) return;

        try {
            audioPlayingRef.current = true;
            setStatus("🗣️ Interviewer speaking...");

            const res = await fetch(`${API_BASE_URL}/api/tts/murf`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text.substring(0, 1000),
                    username: userId.toString()
                })
            });

            if (!res.ok) {
                throw new Error(`TTS failed with status: ${res.status}`);
            }

            const audioBlob = await res.blob();

            // Send interviewer audio to backend for recording
            if (socketRef.current?.connected) {
                const arrayBuffer = await audioBlob.arrayBuffer();
                socketRef.current.emit("interviewer_audio_chunk", {
                    user_id: userId,
                    audio: arrayBuffer
                });
            }

            // Play audio
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            currentAudioRef.current = audio;

            return new Promise((resolve) => {
                audio.onended = () => {
                    audioPlayingRef.current = false;
                    currentAudioRef.current = null;
                    URL.revokeObjectURL(audioUrl);
                    console.log('✅ Audio playback finished');
                    resolve();
                };

                audio.onerror = (err) => {
                    console.error("❌ Audio playback error:", err);
                    audioPlayingRef.current = false;
                    currentAudioRef.current = null;
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                };

                audio.play().catch(err => {
                    console.error("❌ Failed to play audio:", err);
                    audioPlayingRef.current = false;
                    currentAudioRef.current = null;
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                });
            });

        } catch (err) {
            console.error("❌ TTS Error:", err);
            audioPlayingRef.current = false;
            currentAudioRef.current = null;
            // Simulate speaking time
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }, [userId]);

    // Update live WPM calculation
    const updateLiveWPM = useCallback((text) => {
        if (startTimeRef.current && text && text.trim()) {
            const words = text.trim().split(/\s+/).length;
            const minutes = (Date.now() - startTimeRef.current) / 60000;
            const liveWpm = minutes > 0 ? Math.round(words / minutes) : 0;
            setWpm(liveWpm);
        }
    }, []);

    const stopRecording = useCallback(async () => {
        if (hardStopRef.current) return;

        console.log('🛑 HARD STOP: Ending interview NOW');

        hardStopRef.current = true;          // 🔥 KILL SWITCH
        setIsFinalizing(true);
        setInterviewDone(true);
        setIsRecording(false);
        setCurrentTurn('DONE');
        setStatus('🛑 Interview ended');

        // 🔥 STOP TIMER IMMEDIATELY
        setTimeRemaining(0);

        // 🔥 STOP AUDIO / MIC / TTS
        await cleanupAudio();

        // 🔥 INFORM BACKEND (BEST EFFORT)
        if (socketRef.current?.connected) {
            socketRef.current.emit('stop_interview', { user_id: userId });
        }

    }, [userId, cleanupAudio]);

    // 🔥 ADD THIS FUNCTION - Manual submit answer
    const submitAnswer = useCallback(() => {
        if (!socketRef.current?.connected) {
            console.log("📤 Cannot submit - socket not connected");
            return;
        }

        if (currentTurn !== 'USER') {
            console.log("📤 Cannot submit - not user turn");
            return;
        }

        if (isFinalizing) {
            console.log("📤 Cannot submit - already finalizing");
            return;
        }

        console.log("📤 Submitting answer manually");

        socketRef.current.emit("user_done_speaking", {
            user_id: userId
        });

        // Immediately set finalizing to prevent more audio
        //setIsFinalizing(true);
    }, [userId, currentTurn, isFinalizing]);

    // 🔧 FIX 1 (continued): Sync refs with state
    useEffect(() => {
        currentTurnRef.current = currentTurn;
    }, [currentTurn]);

    useEffect(() => {
        isFinalizingRef.current = isFinalizing;
    }, [isFinalizing]);

    useEffect(() => {
        interviewDoneRef.current = interviewDone;
    }, [interviewDone]);

    // Initialize WebSocket connection
    useEffect(() => {
        socketRef.current = io(API_BASE_URL);

        socketRef.current.on('connect', () => {
            setIsConnected(true);
            setError(null);
            console.log('✅ Connected to WebSocket server');
        });

        socketRef.current.on('disconnect', () => {
            setIsConnected(false);
            setStatus('Disconnected');
            console.log('❌ Disconnected from WebSocket server');
        });

        // 🔥 CRITICAL: Listen for backend_ready signal
        socketRef.current.on('backend_ready', (data) => {
            console.log('✅ BACKEND READY - Safe to send audio now!');
            backendReadyRef.current = true;

            // 🔧 FIX: Use ref instead of state for currentTurn check
            if (pendingAudioRef.current.length > 0 && currentTurnRef.current === 'USER') {
                console.log("✅ Backend ready — waiting for audio pipeline to flush");
            } else {
                console.log("✅ Backend ready — waiting for audio pipeline to flush");
            }

            if (currentTurn === 'USER') {
                setStatus('🎤 Speak now...');
            }
        });

        // 🔥 NEW: Force stop speaking handler
        socketRef.current.on("force_stop_speaking", () => {
            console.log("🛑 Force stop speaking received");

            setIsFinalizing(true);     // 🔥 stops audio immediately
            setCurrentTurn('INTERVIEWER');
            setStatus('⏳ Processing your answer...');
        });

        // Listen for interview started
        socketRef.current.on('interview_started', async (data) => {
            console.log('🎤 Interview started', data);
            if (data.session_plan) {
                setSessionPlan(data.session_plan);
            }

            setIsRecording(true);
            setInterviewDone(false);
            setLiveTranscript('');
            setFinalTranscript('');
            setUseMock(data.use_mock || false);
            startTimeRef.current = Date.now();
            setCurrentTurn('INTERVIEWER');

            // 🔥 RESET backend readiness
            backendReadyRef.current = false;
            pendingAudioRef.current = [];

            // 🔥 Show appropriate status message
            if (isFirstInterviewRef.current) {
                setStatus('🔥 Warming up speech recognition...');
                isFirstInterviewRef.current = false;
            } else {
                setStatus('🎤 Starting interview...');
            }

            try {
                // Get microphone permission and setup
                console.log('🎤 Requesting microphone access...');
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });

                streamRef.current = stream;
                console.log('✅ Microphone access granted');

                // Setup AudioContext
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });

                // Load AudioWorklet
                console.log('🔄 Loading audio processor...');
                await audioContextRef.current.audioWorklet.addModule('/pcm-processor.js');

                processorRef.current = new AudioWorkletNode(
                    audioContextRef.current,
                    'pcm-processor'
                );

                audioChunksSentRef.current = 0;

                // 🔧 FIX 2: Use refs in audio handler to avoid stale closure
                processorRef.current.port.onmessage = (event) => {
                    const float32Samples = event.data;
                    const pcmBuffer = floatTo16BitPCM(float32Samples);

                    // 🔴 HARD STOP (interview ended) - USING REF
                    if (hardStopRef.current || interviewDoneRef.current) {
                        return;
                    }

                    // 🔴 Only allow audio during USER turn - USING REFS TO AVOID STALE CLOSURE
                    if (currentTurnRef.current !== 'USER') {
                        return;
                    }

                    // 🔴 Stop after finalization - USING REFS TO AVOID STALE CLOSURE
                    if (isFinalizingRef.current) {
                        return;
                    }

                    // 🔧 FIX 4: Debug log to verify audio flow
                    console.log("🎤 audio tick", {
                        turn: currentTurnRef.current,
                        ready: backendReadyRef.current,
                        finalizing: isFinalizingRef.current,
                        bufferSize: pendingAudioRef.current.length
                    });

                    // 🔥 Buffer until backend ready
                    if (!backendReadyRef.current) {
                        if (pendingAudioRef.current.length < 200) {
                            pendingAudioRef.current.push(pcmBuffer);
                        }

                        if (pendingAudioRef.current.length === 1) {
                            console.log('📦 Buffering audio until backend is ready...');
                        }

                        return;
                    }

                    // 🔥 Flush buffer ONCE
                    if (pendingAudioRef.current.length > 0) {
                        console.log(`🚀 Flushing ${pendingAudioRef.current.length} buffered chunks`);

                        pendingAudioRef.current.forEach((bufferedChunk) => {
                            socketRef.current.emit('audio_chunk', {
                                user_id: userId,
                                audio: bufferedChunk
                            });
                        });

                        pendingAudioRef.current = [];
                    }

                    // 🔥 Send live audio
                    if (socketRef.current?.connected) {
                        socketRef.current.emit('audio_chunk', {
                            user_id: userId,
                            audio: pcmBuffer
                        });
                    }
                };

                // Connect audio nodes
                const source = audioContextRef.current.createMediaStreamSource(stream);
                source.connect(processorRef.current);

                // Silent sink (required)
                const silentGain = audioContextRef.current.createGain();
                silentGain.gain.value = 0;
                processorRef.current.connect(silentGain);
                silentGain.connect(audioContextRef.current.destination);

                console.log('✅ Audio pipeline ready, waiting for backend...');

                // Wait for backend ready (but don't block if using mock)
                if (data.priming_duration) {
                    await new Promise(resolve =>
                        setTimeout(resolve, Math.min(data.priming_duration, 2000))
                    );
                }

                console.log('✅ Backend ready - audio will now be sent during user turn');

            } catch (err) {
                console.error('❌ Error setting up audio:', err);
                setError('Failed to access microphone: ' + err.message);
                setStatus('❌ Microphone error');
                setIsRecording(false);
                cleanupAudio();
            }
        });

        // Intro question from agent
        socketRef.current.on("agent_intro_question", async (data) => {
            if (hardStopRef.current || interviewDoneRef.current) {
                console.log('🚫 Ignoring agent event (interview ended)');
                return;
            }
            console.log('🗣️ Interviewer asking intro question:', data.question);
            setCurrentTurn('INTERVIEWER');
            setStatus("🗣️ Interviewer speaking...");
            setLiveTranscript(''); // Clear any partial transcript

            setMessages(prev => [
                ...prev,
                { role: "interviewer", text: data.question }
            ]);

            // Speak the question
            await speakWithMurf(data.question);

            // After speaking, tell backend interviewer is done
            if (socketRef.current?.connected) {
                socketRef.current.emit("interviewer_done", {
                    user_id: userId
                });
            }

            // Switch to user's turn
            setCurrentTurn('USER');
            setIsFinalizing(false);   // 🔥 IMPORTANT RESET
            setStatus('🎤 Your turn - Answer the question...');
            console.log('✅ Switched to USER turn');
        });

        // 🔥 NEW: Real-time pitch updates from backend
        socketRef.current.on('pitch_update', (data) => {
            setLivePitch({
                mean: data.current_pitch,
                stability: data.stability,
                range: data.range
            });

            // Add to history for graph (keep last 30 points)
            setPitchHistory(prev => {
                const newHistory = [...prev, data.current_pitch];
                if (newHistory.length > 30) newHistory.shift();
                return newHistory;
            });

            // 🔥 NEW: Track stability history separately
            setStabilityHistory(prev => {
                const newHistory = [...prev, data.stability];
                if (newHistory.length > 30) newHistory.shift();
                return newHistory;
            });

            setPitchTimestamps(prev => {
                const newTimes = [...prev, new Date().toLocaleTimeString()];
                if (newTimes.length > 30) newTimes.shift();
                return newTimes;
            });

            // Optional: Log for debugging (remove in production)
            console.log('🎤 Pitch update:', data);
        });

        // Next question from agent
        socketRef.current.on("agent_next_question", async (data) => {
            if (hardStopRef.current || interviewDoneRef.current) {
                console.log('🚫 Ignoring agent event (interview ended)');
                return;
            }
            console.log('🗣️ Interviewer asking next question:', data.question);
            setCurrentTurn('INTERVIEWER');
            setStatus("🗣️ Interviewer speaking...");
            setLiveTranscript(''); // Clear any partial transcript

            setMessages(prev => [
                ...prev,
                { role: "interviewer", text: data.question }
            ]);

            // Speak the question
            await speakWithMurf(data.question);

            // After speaking, tell backend interviewer is done
            if (socketRef.current?.connected) {
                socketRef.current.emit("interviewer_done", {
                    user_id: userId
                });
            }

            // Switch to user's turn
            setCurrentTurn('USER');
            setIsFinalizing(false);   // 🔥 IMPORTANT RESET
            setStatus('🎤 Your turn - Answer the question...');
            console.log('✅ Switched to USER turn');
        });

        // User answer complete (final transcript)
        socketRef.current.on('user_answer_complete', (data) => {
            console.log('✅ User answer complete:', data.answer);

            setIsFinalizing(false);

            // Add user's answer to messages
            setMessages(prev => [
                ...prev,
                { role: "user", text: data.answer }
            ]);

            // Clear live transcript after answer is processed
            setLiveTranscript('');

            // Switch back to interviewer's turn (they'll ask next question)
            setCurrentTurn('INTERVIEWER');
            setStatus('⏳ Waiting for next question...');
        });

        // Listen for live transcript updates from AssemblyAI (partial)
        socketRef.current.on('live_transcript', (data) => {
            console.log('📝 Partial transcript:', data.text);
            setLiveTranscript(data.text);
            updateLiveWPM(data.text);
        });

        // In useInterviewStreaming.js, find the interview_complete handler (around line 350)

        socketRef.current.on('interview_complete', (data) => {
            console.log('🎯 Interview complete with metrics:', data);
            console.log('🔑 Session ID from event:', data.session_id);  // DEBUG LOG

            setIsFinalizing(false);
            setInterviewDone(true);
            setIsRecording(false);
            setCurrentTurn('DONE');
            setStatus('✅ Interview completed - Analysis ready');

            // Set final transcript
            if (data.transcript) {
                setFinalTranscript(data.transcript);
            }

            // Store metrics for display
            if (data.metrics) {
                setMetrics(data.metrics);
                console.log('📊 Metrics received:', data.metrics);
            }

            // Process analysis - ENSURE session_id is included
            if (data.success) {
                console.log('✅ Analysis completed with metrics');
                setAnalysis({
                    ...data,
                    session_id: data.session_id,  // 🔥 EXPLICITLY SET THIS
                    audio_saved: data.audio_saved || false,
                    audio_filepath: data.audio_filepath || null
                });
            } else {
                console.error('❌ Interview completed but not successful');
                setError('Interview failed to complete properly');
            }

            // Clean up audio resources
            cleanupAudio();
        });

        // Listen for errors
        socketRef.current.on('interview_error', (data) => {
            console.log('❌ Interview error:', data.error);
            setError(data.error);
            setIsRecording(false);
            setStatus('❌ Error occurred');
            setCurrentTurn('ERROR');
            cleanupAudio();
        });

        // Interview complete signal from agent
        socketRef.current.on('agent_interview_complete', (data) => {
            console.log('🎉 Agent interview complete:', data.message);
            setStatus('⏳ Finalizing interview...');

            // Stop recording after a short delay
            setTimeout(() => {
                stopRecording();
            }, 1000);
        });

        // Cleanup on unmount
        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            cleanupAudio();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [userId, cleanupAudio, floatTo16BitPCM, stopRecording, updateLiveWPM, speakWithMurf]);

    // Track current turn changes
    useEffect(() => {
        console.log(`🔄 Turn changed to: ${currentTurn}`);
    }, [currentTurn]);

    // Countdown timer
    useEffect(() => {
        if (!isRecording || interviewDone || hardStopRef.current) return;

        const interval = setInterval(() => {
            setTimeRemaining(prev => {
                if (prev <= 1) {
                    clearInterval(interval);
                    stopRecording();
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(interval);
    }, [isRecording, interviewDone, stopRecording]);


    const startRecording = useCallback(async (stressTest = false) => {
        try {
            setError(null);
            setAnalysis(null);
            setLiveTranscript('');
            setFinalTranscript('');
            setInterviewDone(false);
            setIsFinalizing(false);
            setStatus('🎤 Starting interview...');
            setCurrentTurn('INTERVIEWER');
            setTimeRemaining(30 * 60);
            setIsRecording(true);
            setMessages([]);

            // 🔥 CRITICAL: Reset the hard stop flag
            hardStopRef.current = false;

            // Clean up any previous audio resources
            await cleanupAudio();

            // Start the interview
            socketRef.current.emit('start_interview', {
                user_id: userId,
                stress_test: stressTest
            });

        } catch (err) {
            console.error('Error starting interview:', err);
            setError('Failed to start interview: ' + err.message);
            setStatus('❌ Failed to start');
        }
    }, [userId, cleanupAudio]);

    const formatTime = (seconds) => {
        if (typeof seconds !== 'number' || !isFinite(seconds) || seconds <= 0) {
            return '00:00';
        }

        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);

        return `${mins.toString().padStart(2, '0')}:${secs
            .toString()
            .padStart(2, '0')}`;
    };


    return {
        // State
        isConnected,
        isRecording,
        transcript: !interviewDone ? liveTranscript : finalTranscript,
        wpm,
        analysis,
        error,
        useMock,
        isFinalizing,
        status,
        timeRemaining: formatTime(timeRemaining),
        currentTurn,
        messages,
        sessionPlan: sessionPlan,

        // Actions
        startRecording,
        stopRecording,

        // Live data
        liveTranscript,
        finalTranscript,
        interviewDone,
        liveWpm: wpm,
        finalAnalysis: analysis,
        isInterviewerSpeaking: currentTurn === 'INTERVIEWER' || audioPlayingRef.current,
        livePitch,
        pitchHistory,
        pitchTimestamps,
        stabilityHistory,
        submitAnswer,
    };
};