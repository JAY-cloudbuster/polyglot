import { useState, useRef } from 'react'

/**
 * FileUploader — drag-and-drop or click-to-upload for .wav/.mp3 audio files.
 */
export default function FileUploader({ onFileSelected, disabled }) {
    const [dragOver, setDragOver] = useState(false)
    const inputRef = useRef(null)

    const handleFile = (file) => {
        if (!file) return

        const validTypes = [
            'audio/wav', 'audio/x-wav', 'audio/wave',
            'audio/mpeg', 'audio/mp3', 'audio/ogg',
            'audio/webm', 'audio/flac',
        ]

        // Also check extension as fallback
        const ext = file.name.split('.').pop()?.toLowerCase()
        const validExts = ['wav', 'mp3', 'ogg', 'webm', 'flac']

        if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
            alert('Please upload a valid audio file (.wav, .mp3, .ogg, .webm, .flac)')
            return
        }

        onFileSelected?.(file)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragOver(false)
        const file = e.dataTransfer.files?.[0]
        handleFile(file)
    }

    const handleDragOver = (e) => {
        e.preventDefault()
        setDragOver(true)
    }

    const handleDragLeave = () => setDragOver(false)

    const handleChange = (e) => {
        const file = e.target.files?.[0]
        handleFile(file)
        // Reset input so re-uploading same file works
        if (inputRef.current) inputRef.current.value = ''
    }

    return (
        <div
            id="file-uploader"
            className={`upload-area ${dragOver ? 'upload-area--dragover' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
            <div className="upload-area__icon">
                <svg viewBox="0 0 24 24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
            </div>

            <span className="upload-area__label">Upload Audio</span>
            <span className="upload-area__sublabel">Drag & drop or click (.wav, .mp3)</span>

            <input
                ref={inputRef}
                type="file"
                accept=".wav,.mp3,.ogg,.webm,.flac,audio/*"
                onChange={handleChange}
                disabled={disabled}
            />
        </div>
    )
}
