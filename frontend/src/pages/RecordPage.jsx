import { useState, useRef, useCallback } from 'react'
import { analyzeAudio } from '../services/api'
import { downloadReport } from '../services/reportGenerator'
import VerdictPanel from '../components/VerdictPanel'

export default function RecordPage() {
    const [recording, setRecording] = useState(false)
    const [duration, setDuration] = useState(0)
    const [audioBlob, setAudioBlob] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const mediaRecorderRef = useRef(null)
    const chunksRef = useRef([])
    const timerRef = useRef(null)

    const formatTime = (s) => {
        const m = Math.floor(s / 60)
        const sec = s % 60
        return `${m}:${sec.toString().padStart(2, '0')}`
    }

    const startRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
            })

            const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus' : 'audio/webm'

            const recorder = new MediaRecorder(stream, { mimeType: mime })
            chunksRef.current = []
            mediaRecorderRef.current = recorder

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data)
            }

            recorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
                stream.getTracks().forEach(t => t.stop())
                clearInterval(timerRef.current)
                setAudioBlob(blob)
                chunksRef.current = []
            }

            recorder.start(250)
            setRecording(true)
            setDuration(0)
            setResult(null)
            setError(null)
            setAudioBlob(null)

            timerRef.current = setInterval(() => setDuration(d => d + 1), 1000)
        } catch {
            setError('Microphone access denied. Please allow microphone permissions.')
        }
    }, [])

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
            setRecording(false)
        }
    }, [])

    const handleAnalyze = useCallback(async () => {
        if (!audioBlob) return
        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await analyzeAudio(audioBlob)
            setResult(res)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }, [audioBlob])

    const reset = () => {
        setAudioBlob(null)
        setResult(null)
        setError(null)
        setDuration(0)
    }

    return (
        <div className="page animate-in">
            <div className="page__header">
                <h1 className="page__title">Record voice</h1>
                <p className="page__desc">
                    Use your microphone to record a voice sample for analysis.
                </p>
            </div>

            <div className="card hover-lift">
                <div className="card__label">Microphone</div>

                <div className="rec-area">
                    {recording && (
                        <div className="wave-bars">
                            {[...Array(5)].map((_, i) => <div key={i} className="wave-bar" />)}
                        </div>
                    )}

                    <button
                        className={`rec-btn ${recording ? 'rec-btn--active' : ''}`}
                        onClick={recording ? stopRecording : startRecording}
                        disabled={loading}
                    >
                        <div className="rec-btn__dot" />
                    </button>

                    {recording ? (
                        <div className="rec-timer">{formatTime(duration)}</div>
                    ) : audioBlob ? (
                        <span className="rec-label">Recording captured · {formatTime(duration)}</span>
                    ) : (
                        <span className="rec-label">Tap to start recording</span>
                    )}
                </div>
            </div>

            {audioBlob && !result && (
                <button
                    className="analyze-btn"
                    onClick={handleAnalyze}
                    disabled={loading}
                >
                    {loading ? (
                        <>
                            <div className="spinner" />
                            Analyzing…
                        </>
                    ) : (
                        'Analyze recording'
                    )}
                </button>
            )}

            {loading && (
                <div className="loading-state">
                    <div className="spinner" style={{ margin: '0 auto' }} />
                    <p className="loading-state__text">Running deepfake detection…</p>
                </div>
            )}

            {error && <div className="error-msg">{error}</div>}

            {result && (
                <div className="animate-in">
                    <VerdictPanel
                        verdict={result.verdict}
                        confidence={result.confidence}
                        reasoning={result.reasoning}
                    />
                    <div className="result-actions">
                        <button className="report-btn" onClick={() => downloadReport(result)}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            Download Report
                        </button>
                        <button className="reset-btn" onClick={reset}>
                            Record again
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
