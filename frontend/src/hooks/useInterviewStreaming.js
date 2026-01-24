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

    const socketRef = useRef(null);
    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const streamRef = useRef(null);
    const startTimeRef = useRef(null);
    const backendReadyRef = useRef(false); // 🔥 ADDED THIS LINE
    const pendingAudioRef = useRef([]);

    // Initialize WebSocket connection
    useEffect(() => {
        socketRef.current = io('http://localhost:5000');

        socketRef.current.on('connect', () => {
            setIsConnected(true);
            setError(null);

            // Join interview room
            socketRef.current.emit('join_interview', { user_id: userId });
        });

        socketRef.current.on('disconnect', () => {
            setIsConnected(false);
        });

        // Listen for live transcript updates from AssemblyAI (partial)
        socketRef.current.on('live_transcript', (data) => {
            console.log('📝 Partial transcript:', data.text);
            setLiveTranscript(data.text);
            updateLiveWPM(data.text);
        });

        // Listen for final transcript parts (accumulate during interview)
        socketRef.current.on('final_transcript', (data) => {
            console.log('📝 Final transcript:', data.text);
            if (!interviewDone) {
                // During interview: accumulate final transcript parts
                setFinalTranscript(prev => prev + " " + data.text);
            }
            setLiveTranscript(''); // Clear partial after final
        });

        // Listen for interview completion
        socketRef.current.on('interview_complete', (data) => {
            console.log('🎯 Interview complete:', data);
            setIsFinalizing(false);
            setInterviewDone(true);
            setIsRecording(false); // Fix status display
            setFinalTranscript(data.transcript); // Set clean final transcript

            // 🚀 FAST: Analysis is now included directly in the completion event!
            if (data.success && data.overall_score !== undefined) {
                console.log('✅ Analysis completed instantly with pre-aggregated stats');
                setAnalysis(data);
            } else {
                // Fallback to old method if analysis not included
                console.log('⚠️ Analysis not included, falling back to separate analysis call');
                performFinalAnalysis(data.audio_path, data.transcript);
            }
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

            // 🎤 NOW start microphone
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            streamRef.current = stream;

            // 🎧 Create AudioContext
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            const source = audioContextRef.current.createMediaStreamSource(stream);

            // 🔊 Load AudioWorklet
            await audioContextRef.current.audioWorklet.addModule('/pcm-processor.js');

            processorRef.current = new AudioWorkletNode(
                audioContextRef.current,
                'pcm-processor'
            );

            processorRef.current.port.onmessage = (event) => {
                const float32Samples = event.data;
                const pcmBuffer = floatTo16BitPCM(float32Samples);

                // Buffer audio until backend is 100% ready
                if (!backendReadyRef.current) {
                    pendingAudioRef.current.push(pcmBuffer);
                    return;
                }

                // Flush buffered audio ONCE
                if (pendingAudioRef.current.length > 0) {
                    pendingAudioRef.current.forEach(buf => {
                        socketRef.current.emit('audio_chunk', {
                            user_id: userId,
                            audio: buf
                        });
                    });
                    pendingAudioRef.current = [];
                }

                socketRef.current.emit('audio_chunk', {
                    user_id: userId,
                    audio: pcmBuffer
                });

            };

            source.connect(processorRef.current);

            // 🔇 Silent sink (required)
            const silentGain = audioContextRef.current.createGain();
            silentGain.gain.value = 0;
            processorRef.current.connect(silentGain);
            silentGain.connect(audioContextRef.current.destination);

            // 🔥 PRIME ASSEMBLYAI WITH 300ms SILENCE
            const silence = new Int16Array(16000 * 0.3);
            socketRef.current.emit('audio_chunk', {
                user_id: userId,
                audio: silence.buffer
            });

            backendReadyRef.current = true;
        });


        // Listen for errors
        socketRef.current.on('interview_error', (data) => {
            console.log('❌ Interview error:', data.error);
            setError(data.error);
            setIsRecording(false);
        });

        // Cleanup
        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            stopRecording();
        };
    }, [userId]);


    const startRecording = useCallback(async () => {
        // 🔥 Always reset backend readiness
        backendReadyRef.current = false;
        pendingAudioRef.current = []; // 🔥 ADD THIS

        try {
            setError(null);

            // Clean up any previous audio resources
            if (processorRef.current) {
                processorRef.current.disconnect();
                processorRef.current = null;
            }
            if (audioContextRef.current) {
                await audioContextRef.current.close();
                audioContextRef.current = null;
            }
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
                streamRef.current = null;
            }

            // Reset UI state
            setAnalysis(null);
            setLiveTranscript('');
            setFinalTranscript('');
            setInterviewDone(false);

            // 🚨 DO NOT start mic here
            // 🚀 ONLY tell backend to start
            socketRef.current.emit('start_interview', { user_id: userId });

        } catch (err) {
            console.error('Error starting interview:', err);
            setError('Failed to start interview');
        }
    }, [userId]);


    // Lightweight analysis function (defined before use)
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

            // Call your existing analysis endpoint
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
            } else {
                console.error('❌ Final analysis failed');
                setError('Analysis failed');
            }
        } catch (err) {
            console.error('❌ Error in final analysis:', err);
            setError('Analysis error');
        }
    }, []);

    const stopRecording = useCallback(async () => {
        if (!isRecording) return;

        console.log('⏳ Waiting 3s to flush final transcript...');
        setIsFinalizing(true);

        // 🔥 STOP backend AFTER delay (keep audio alive)
        setTimeout(async () => {
            // Stop worklet
            if (processorRef.current) {
                processorRef.current.disconnect();
                processorRef.current = null;
            }

            // Stop mic tracks
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
                streamRef.current = null;
            }

            // NOW close audio context
            if (audioContextRef.current) {
                await audioContextRef.current.close();
                audioContextRef.current = null;
            }

            socketRef.current.emit('stop_interview', { user_id: userId });
        }, 3000);

    }, [isRecording, userId]);


    return {
        // State
        isConnected,
        isRecording,
        transcript: !interviewDone ? liveTranscript : finalTranscript, // Show live during interview, final after
        wpm,
        analysis,
        error,
        useMock, // Whether using mock transcription (no AssemblyAI API)
        isFinalizing, // 🔥 ADDED THIS LINE

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