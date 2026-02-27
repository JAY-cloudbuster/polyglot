import { useState, useCallback } from 'react'
import AudioRecorder from '../components/AudioRecorder'
import FileUploader from '../components/FileUploader'
import VerdictPanel from '../components/VerdictPanel'
import ConfidenceMetrics from '../components/ConfidenceMetrics'
import LivenessPrompt from '../components/LivenessPrompt'
import { analyzeAudio } from '../services/api'

/**
 * Home page — main application flow.
 */
export default function Home() {
    const [audioBlob, setAudioBlob] = useState(null)
    const [fileName, setFileName] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const handleRecorded = useCallback((blob) => {
        setAudioBlob(blob)
        setFileName('🎙 Live Recording')
        setResult(null)
        setError(null)
    }, [])

    const handleFileSelected = useCallback((file) => {
        setAudioBlob(file)
        setFileName(file.name)
        setResult(null)
        setError(null)
    }, [])

    const clearAudio = useCallback(() => {
        setAudioBlob(null)
        setFileName(null)
        setResult(null)
        setError(null)
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

    return (
        <>
            {/* Hero */}
            <section className="hero">
                <h1 className="hero__title">
                    Detect <span className="hero__title-gradient">Voice Deepfakes</span> in Seconds
                </h1>
                <p className="hero__subtitle">
                    Record your voice or upload an audio file. Our acoustic AI analyzes speech
                    patterns to identify synthetic or manipulated audio in real time.
                </p>
            </section>

            {/* Audio Input */}
            <div className="section-label">Audio Input</div>
            <div className="audio-input">
                <AudioRecorder onRecorded={handleRecorded} disabled={loading} />
                <FileUploader onFileSelected={handleFileSelected} disabled={loading} />
            </div>

            {/* Selected File Info */}
            {fileName && (
                <div className="file-info">
                    <span>📎 {fileName}</span>
                    {!loading && (
                        <button className="file-info__remove" onClick={clearAudio} title="Remove">×</button>
                    )}
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="error-msg">
                    <span className="error-msg__icon">⚠️</span>
                    {error}
                </div>
            )}

            {/* Analyze Button */}
            <button
                id="analyze-button"
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={!audioBlob || loading}
            >
                {loading ? (
                    <>
                        <span className="loading__spinner" style={{ width: 20, height: 20, borderWidth: 2, margin: 0 }} />
                        Analyzing...
                    </>
                ) : (
                    <>🔬 Analyze Audio</>
                )}
            </button>

            {/* Loading */}
            {loading && (
                <div className="loading">
                    <div className="loading__spinner" />
                    <p className="loading__text">Running acoustic deepfake detection…</p>
                </div>
            )}

            {/* Results */}
            {result && (
                <>
                    <VerdictPanel
                        verdict={result.verdict}
                        confidence={result.confidence}
                        reasoning={result.reasoning}
                    />

                    <ConfidenceMetrics
                        confidence={result.confidence}
                        verdict={result.verdict}
                        featuresAnalyzed={result.features_analyzed}
                        featureBreakdown={result.feature_breakdown}
                    />

                    <LivenessPrompt />
                </>
            )}
        </>
    )
}
