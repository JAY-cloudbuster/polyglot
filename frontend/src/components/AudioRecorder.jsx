import { useState, useRef, useCallback } from 'react'

/**
 * AudioRecorder — captures microphone audio via MediaRecorder API.
 * Returns audio Blob to parent on stop.
 */
export default function AudioRecorder({ onRecorded, disabled }) {
    const [recording, setRecording] = useState(false)
    const [duration, setDuration] = useState(0)
    const mediaRecorderRef = useRef(null)
    const chunksRef = useRef([])
    const timerRef = useRef(null)

    const startRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            })

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm',
            })

            // CRITICAL FIX: Reset the chunks array *before* we start recording
            chunksRef.current = []
            mediaRecorderRef.current = mediaRecorder

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data)
            }

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
                stream.getTracks().forEach((t) => t.stop())
                clearInterval(timerRef.current)
                setDuration(0)
                // Pass the blob up, then immediately clear the chunks array
                // so the next recording starts completely fresh
                onRecorded?.(blob)
                chunksRef.current = []
            }

            mediaRecorder.start(250) // collect data every 250ms
            setRecording(true)
            setDuration(0)

            timerRef.current = setInterval(() => {
                setDuration((d) => d + 1)
            }, 1000)
        } catch (err) {
            console.error('Microphone access error:', err)
            alert('Microphone access denied. Please allow microphone permissions.')
        }
    }, [onRecorded])

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
            setRecording(false)
        }
    }, [])

    const formatTime = (s) => {
        const mins = Math.floor(s / 60)
        const secs = s % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    return (
        <button
            type="button"
            id="record-button"
            className={`record-btn ${recording ? 'record-btn--recording' : ''}`}
            onClick={recording ? stopRecording : startRecording}
            disabled={disabled && !recording}
        >
            <div className="record-btn__icon">
                {recording ? (
                    <svg viewBox="0 0 24 24">
                        <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                ) : (
                    <svg viewBox="0 0 24 24">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" />
                        <line x1="12" y1="19" x2="12" y2="23" stroke="white" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                )}
            </div>

            <span className="record-btn__label">
                {recording ? `Recording — ${formatTime(duration)}` : 'Record Voice'}
            </span>

            {recording ? (
                <div className="waveform">
                    {[...Array(7)].map((_, i) => (
                        <div key={i} className="waveform__bar" />
                    ))}
                </div>
            ) : (
                <span className="record-btn__sublabel">Click to start microphone</span>
            )}
        </button>
    )
}
