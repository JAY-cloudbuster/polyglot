import { useState, useCallback, useRef } from 'react'
import { analyzeAudio } from '../services/api'
import { downloadReport } from '../services/reportGenerator'
import VerdictPanel from '../components/VerdictPanel'

export default function UploadPage() {
    const [file, setFile] = useState(null)
    const [dragOver, setDragOver] = useState(false)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const inputRef = useRef(null)

    const ACCEPTED = '.wav,.mp3,.webm,.ogg,.flac,.m4a,.aac,.mp4'

    const handleFile = (f) => {
        if (f && f.size > 0) {
            setFile(f)
            setResult(null)
            setError(null)
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragOver(false)
        const f = e.dataTransfer.files[0]
        handleFile(f)
    }

    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B'
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB'
        return (bytes / 1048576).toFixed(1) + ' MB'
    }

    const handleAnalyze = useCallback(async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await analyzeAudio(file)
            setResult(res)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }, [file])

    const reset = () => {
        setFile(null)
        setResult(null)
        setError(null)
        if (inputRef.current) inputRef.current.value = ''
    }

    return (
        <div className="page animate-in">
            <div className="page__header">
                <h1 className="page__title">Upload audio</h1>
                <p className="page__desc">
                    Select an audio file from your device to check for deepfake speech.
                </p>
            </div>

            <div className="card hover-lift">
                <div className="card__label">Audio file</div>

                {!file ? (
                    <div
                        className={`upload-zone ${dragOver ? 'upload-zone--dragover' : ''}`}
                        onClick={() => inputRef.current?.click()}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                    >
                        <div className="upload-zone__icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="17 8 12 3 7 8" />
                                <line x1="12" y1="3" x2="12" y2="15" />
                            </svg>
                        </div>
                        <p className="upload-zone__text">
                            Drop audio file here or click to browse
                        </p>
                        <p className="upload-zone__hint">
                            WAV, MP3, WebM, OGG, FLAC — up to 25 MB
                        </p>
                        <input
                            ref={inputRef}
                            type="file"
                            accept={ACCEPTED}
                            style={{ display: 'none' }}
                            onChange={(e) => handleFile(e.target.files[0])}
                        />
                    </div>
                ) : (
                    <div className="file-pill animate-in">
                        <span className="file-pill__name">{file.name}</span>
                        <span className="file-pill__size">{formatSize(file.size)}</span>
                        {!loading && (
                            <button className="file-pill__remove" onClick={reset}>×</button>
                        )}
                    </div>
                )}
            </div>

            {file && !result && (
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
                        'Analyze file'
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
                            Upload another file
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
