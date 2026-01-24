import { useState, useRef, useEffect, useCallback } from 'react';
import io from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

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
    const [status, setStatus] = useState('Ready'); // 🔥 ADDED: Status for user feedback

    const socketRef = useRef(null);
    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const streamRef = useRef(null);
    const startTimeRef = useRef(null);
    const backendReadyRef = useRef(false); // Track if backend is ready
    const pendingAudioRef = useRef([]); // Buffer audio before backend is ready
    const isFirstInterviewRef = useRef(true); // 🔥 Track if first interview

    // Initialize WebSocket connection
    useEffect(() => {
        socketRef.current = io('http://localhost:5000');

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

            // Flush any buffered audio
            if (pendingAudioRef.current.length > 0) {
                console.log(`📤 Flushing ${pendingAudioRef.current.length} buffered audio chunks...`);
                pendingAudioRef.current.forEach(buffer => {
                    socketRef.current.emit('audio_chunk', {
                        user_id: userId,
                        audio: buffer
                    });
                });
                pendingAudioRef.current = [];
            }

            setStatus('🎤 Speak now...');
        });

        // Listen for interview started
        socketRef.current.on('interview_started', async (data) => {
            console.log('🎤 Interview started', data);

            setIsRecording(true);
            setInterviewDone(false);
            setLiveTranscript('');
            setFinalTranscript('');
            setUseMock(data.use_mock || false);
            startTimeRef.current = Date.now();

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
                        noiseSuppression: true
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

                processorRef.current.port.onmessage = (event) => {
                    const float32Samples = event.data;
                    const pcmBuffer = floatTo16BitPCM(float32Samples);

                    // 🔥 BUFFER audio until backend is ready
                    if (!backendReadyRef.current) {
                        if (pendingAudioRef.current.length < 200) {
                            pendingAudioRef.current.push(pcmBuffer);
                        }
                        if (pendingAudioRef.current.length === 1) {
                            console.log('📦 Buffering audio until backend is ready...');
                        }
                        return;
                    }

                    // Send audio to backend
                    socketRef.current.emit('audio_chunk', {
                        user_id: userId,
                        audio: pcmBuffer
                    });
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

                await new Promise(resolve =>
                    setTimeout(resolve, data.priming_duration + 500)
                );


                console.log('✅ Backend ready - audio will now be sent');
                setStatus('🎤 Speak now...');

            } catch (err) {
                console.error('❌ Error setting up audio:', err);
                setError('Failed to access microphone: ' + err.message);
                setStatus('❌ Microphone error');
                setIsRecording(false);
            }
        });

        // Listen for live transcript updates from AssemblyAI (partial)
        socketRef.current.on('live_transcript', (data) => {
            console.log('📝 Partial transcript:', data.text);
            setLiveTranscript(data.text);
            updateLiveWPM(data.text);
        });

        // Listen for final transcript parts (accumulate during interview)
        socketRef.current.on('final_transcript', (data) => {
            console.log('📝 Final transcript part:', data.text);
            if (!interviewDone) {
                // During interview: accumulate final transcript parts
                setFinalTranscript(prev => prev + (prev ? " " : "") + data.text);
            }
            setLiveTranscript(''); // Clear partial after final
        });

        // Listen for interview completion
        socketRef.current.on('interview_complete', (data) => {
            console.log('🎯 Interview complete:', data);
            setIsFinalizing(false);
            setInterviewDone(true);
            setIsRecording(false);
            setStatus('✅ Interview completed');

            // Set final transcript
            if (data.transcript) {
                setFinalTranscript(data.transcript);
            } else if (finalTranscript) {
                // Use accumulated transcript if no final from backend
                console.log('⚠️ No transcript in completion data, using accumulated');
            }

            // Process analysis
            if (data.success) {
                console.log('✅ Analysis completed instantly');
                setAnalysis({
                    ...data,
                    // Ensure we have audio path for display
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
            cleanupAudio();
        });

        // Cleanup on unmount
        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            cleanupAudio();
        };
    }, [userId]);

    // Helper function to clean up audio resources
    const cleanupAudio = useCallback(async () => {
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

    const startRecording = useCallback(async () => {
        try {
            setError(null);
            setAnalysis(null);
            setLiveTranscript('');
            setFinalTranscript('');
            setInterviewDone(false);
            setStatus('🎤 Starting interview...');

            // Clean up any previous audio resources
            await cleanupAudio();

            // Start the interview
            socketRef.current.emit('start_interview', { user_id: userId });

        } catch (err) {
            console.error('Error starting interview:', err);
            setError('Failed to start interview: ' + err.message);
            setStatus('❌ Failed to start');
        }
    }, [userId, cleanupAudio]);

    // Convert Float32Array to 16-bit PCM (required by AssemblyAI)
    const floatTo16BitPCM = useCallback((float32Array) => {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        let offset = 0;

        for (let i = 0; i < float32Array.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return buffer;
    }, []);

    // Update live WPM calculation
    const updateLiveWPM = useCallback((text) => {
        if (startTimeRef.current && text.trim()) {
            const words = text.trim().split(/\s+/).length;
            const minutes = (Date.now() - startTimeRef.current) / 60000;
            const liveWpm = minutes > 0 ? Math.round(words / minutes) : 0;
            setWpm(liveWpm);
        }
    }, []);

    // Perform final analysis using your existing backend logic
    const performFinalAnalysis = useCallback(async (audioPath, transcript) => {
        try {
            console.log('🔍 Performing final analysis on:', audioPath);
            setStatus('🔍 Analyzing...');

            const response = await fetch(`${API_BASE}/api/analyze_audio_final`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    audio_path: audioPath,
                    transcript: transcript
                })
            });

            if (response.ok) {
                const analysisData = await response.json();
                console.log('✅ Final analysis complete:', analysisData);
                setAnalysis(analysisData);
                setStatus('✅ Analysis complete');
            } else {
                console.error('❌ Final analysis failed');
                setError('Analysis failed');
                setStatus('❌ Analysis failed');
            }
        } catch (err) {
            console.error('❌ Error in final analysis:', err);
            setError('Analysis error: ' + err.message);
            setStatus('❌ Analysis error');
        }
    }, []);

    const stopRecording = useCallback(async () => {
        if (!isRecording) return;

        console.log('⏳ Stopping interview...');
        setIsFinalizing(true);
        setStatus('⏳ Finalizing...');

        // Stop audio processing first
        if (processorRef.current) {
            processorRef.current.port.onmessage = null;
        }

        // Give a moment for any final audio to be sent
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Stop the interview on backend
        socketRef.current.emit('stop_interview', { user_id: userId });

        // Note: Don't clean up audio context yet - backend might still be processing

    }, [isRecording, userId]);

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
        status, // 🔥 ADDED: Status message for user feedback

        // Actions
        startRecording,
        stopRecording,

        // Live data access
        liveTranscript,
        finalTranscript,
        interviewDone,
        liveWpm: wpm,
        finalAnalysis: analysis
    };
};