import { useEffect, useRef, useState, useCallback } from 'react'

/* ─── Animated Processing Simulator ─── */
function ProcessingSimulator() {
    const [simState, setSimState] = useState('idle') // idle | running | done
    const [activeStep, setActiveStep] = useState(-1)
    const [completedSteps, setCompletedSteps] = useState([])
    const [logs, setLogs] = useState([])
    const [progress, setProgress] = useState(0)
    const logsEndRef = useRef(null)
    const timerRef = useRef(null)
    const canvasRef = useRef(null)
    const animFrameRef = useRef(null)

    const steps = [
        {
            id: 'upload',
            icon: '📤',
            title: 'Audio Upload',
            subtitle: 'Browser → Server',
            duration: 1200,
            color: '#6366f1',
            logs: [
                '> User clicks "Analyze" button',
                '> Creating FormData with audio blob...',
                '> POST /analyze (multipart/form-data)',
                '> Content-Type: audio/wav, Size: 2.4 MB',
                '✓ Upload complete — 200 OK',
            ],
            visual: 'upload',
        },
        {
            id: 'validate',
            icon: '🛡️',
            title: 'Server Validation',
            subtitle: 'FastAPI Endpoint',
            duration: 800,
            color: '#8b5cf6',
            logs: [
                '> FastAPI receives UploadFile object',
                '> Checking content type: audio/wav ✓',
                '> Reading bytes into memory buffer...',
                '> File size: 2,457,600 bytes (< 10 MB limit) ✓',
                '✓ Validation passed',
            ],
            visual: 'validate',
        },
        {
            id: 'decode',
            icon: '🔊',
            title: 'Audio Decoding',
            subtitle: 'librosa · soundfile',
            duration: 1400,
            color: '#a855f7',
            logs: [
                '> librosa.load(audio_bytes, sr=None)',
                '> Detected format: WAV (PCM, 44100 Hz)',
                '> Decoding audio samples...',
                '> Original: 44100 Hz, 2 channels, 5.2s',
                '> Resampling to 16000 Hz mono...',
                '> Normalizing amplitude to [-1.0, 1.0]',
                '✓ Waveform ready: 83,200 samples',
            ],
            visual: 'waveform',
        },
        {
            id: 'features',
            icon: '🧬',
            title: 'wav2vec2 Feature Extraction',
            subtitle: 'HuggingFace Transformers',
            duration: 2000,
            color: '#ec4899',
            logs: [
                '> Loading Wav2Vec2FeatureExtractor...',
                '> Tokenizing waveform → input_values tensor',
                '> Tensor shape: [1, 83200]',
                '> Forward pass through 12 transformer layers...',
                '> Layer 1/12... Layer 4/12... Layer 8/12...',
                '> Layer 12/12 complete',
                '> Extracting hidden states → logits',
                '✓ Feature extraction complete',
            ],
            visual: 'neural',
        },
        {
            id: 'classify',
            icon: '⚖️',
            title: 'Neural Classification',
            subtitle: '2-class Softmax',
            duration: 1000,
            color: '#f43f5e',
            logs: [
                '> Applying classification head...',
                '> Raw logits: [3.241, -2.876]',
                '> Softmax probabilities:',
                '  → P(FAKE)  = 0.9977',
                '  → P(REAL)  = 0.0023',
                '> Predicted class: "FAKE" (index: 0)',
                '✓ Classification complete',
            ],
            visual: 'classify',
        },
        {
            id: 'bayesian',
            icon: '📐',
            title: 'Bayesian Calibration',
            subtitle: 'Gaussian Smoothing',
            duration: 1200,
            color: '#f97316',
            logs: [
                '> Applying Gaussian smoothing kernel...',
                '> σ = 0.15, window = 5 samples',
                '> Computing posterior probability...',
                '> Prior P(FAKE) = 0.5 (uniform)',
                '> Likelihood ratio: 434.2',
                '> Calibrated confidence: 0.8714',
                '✓ Bayesian score: 87.14%',
            ],
            visual: 'bayesian',
        },
        {
            id: 'forensics',
            icon: '🔬',
            title: 'Acoustic Forensics',
            subtitle: '7 Feature Metrics',
            duration: 1600,
            color: '#eab308',
            logs: [
                '> Extracting pitch contour (yin algorithm)...',
                '  → pitch_std: 12.3 Hz (LOW — suspicious)',
                '> Computing spectral centroid...',
                '  → spectral_centroid_std: 450.2 Hz',
                '> Measuring RMS dynamic range...',
                '  → rms_dynamic_range: 5.2× (COMPRESSED)',
                '> Spectral flatness: 0.008',
                '> Silence noise: 0.001 (CLEAN — suspicious)',
                '> HF/LF ratio: 0.07',
                '> Zero crossing std: 0.12',
                '✓ All 7 metrics extracted',
            ],
            visual: 'forensics',
        },
        {
            id: 'llm',
            icon: '💬',
            title: 'LLM Explanation',
            subtitle: 'Groq · Llama 3 8B',
            duration: 1800,
            color: '#34d399',
            logs: [
                '> Composing prompt with verdict + metrics...',
                '> POST api.groq.com/v1/chat/completions',
                '> Model: llama3-8b-8192',
                '> Streaming response tokens...',
                '> "The wav2vec2 neural network detected',
                '>  synthetic speech patterns with 87%',
                '>  confidence. Key acoustic anomalies',
                '>  include unnaturally flat pitch..."',
                '✓ Forensic explanation generated (147 tokens)',
            ],
            visual: 'llm',
        },
        {
            id: 'response',
            icon: '📊',
            title: 'JSON Response',
            subtitle: 'Final Output',
            duration: 800,
            color: '#22d3ee',
            logs: [
                '> Assembling response JSON...',
                '> {',
                '>   "verdict": "FAKE",',
                '>   "confidence": 0.8714,',
                '>   "reasoning": "The wav2vec2...",',
                '>   "features_analyzed": 7,',
                '>   "timestamp": "2026-03-30T..."',
                '> }',
                '✓ Response sent → 200 OK (2.8s total)',
            ],
            visual: 'response',
        },
    ]

    // Waveform canvas drawing
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        const w = canvas.width
        const h = canvas.height
        let t = 0

        const drawFrame = () => {
            ctx.clearRect(0, 0, w, h)

            if (activeStep === -1 && simState === 'idle') {
                // Idle — flat line with subtle pulse
                ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)'
                ctx.lineWidth = 2
                ctx.beginPath()
                for (let x = 0; x < w; x++) {
                    const y = h / 2 + Math.sin(x * 0.02 + t * 0.02) * 3
                    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
                }
                ctx.stroke()
            } else if (activeStep >= 0 && activeStep <= 2) {
                // Upload / Validate / Decode → waveform building
                const col = steps[Math.max(0, activeStep)]?.color || '#6366f1'
                ctx.strokeStyle = col
                ctx.lineWidth = 2
                ctx.shadowColor = col
                ctx.shadowBlur = 8
                ctx.beginPath()
                const amp = activeStep === 2 ? 30 : 8 + activeStep * 8
                const freq = activeStep === 2 ? 0.06 : 0.03
                for (let x = 0; x < w; x++) {
                    const noise = Math.sin(x * 0.15 + t * 0.05) * 5
                    const y = h / 2 + Math.sin(x * freq + t * 0.03) * amp + noise
                    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
                }
                ctx.stroke()
                ctx.shadowBlur = 0
            } else if (activeStep >= 3 && activeStep <= 4) {
                // Neural network — particle grid
                const col = steps[activeStep]?.color || '#ec4899'
                const cols = 20
                const rows = 6
                const gapX = w / (cols + 1)
                const gapY = h / (rows + 1)
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const cx = gapX * (c + 1)
                        const cy = gapY * (r + 1)
                        const pulse = Math.sin(t * 0.04 + c * 0.5 + r * 0.3) * 0.5 + 0.5
                        const radius = 2 + pulse * 3
                        ctx.beginPath()
                        ctx.arc(cx, cy, radius, 0, Math.PI * 2)
                        ctx.fillStyle = col
                        ctx.globalAlpha = 0.3 + pulse * 0.7
                        ctx.fill()
                        ctx.globalAlpha = 1
                        // connections
                        if (c < cols - 1) {
                            ctx.strokeStyle = col
                            ctx.globalAlpha = 0.1 + pulse * 0.15
                            ctx.lineWidth = 1
                            ctx.beginPath()
                            ctx.moveTo(cx, cy)
                            ctx.lineTo(gapX * (c + 2), gapY * (r + 1))
                            ctx.stroke()
                            ctx.globalAlpha = 1
                        }
                    }
                }
            } else if (activeStep === 5) {
                // Bayesian — gaussian curve
                const col = steps[activeStep]?.color || '#f97316'
                ctx.strokeStyle = col
                ctx.lineWidth = 2.5
                ctx.shadowColor = col
                ctx.shadowBlur = 10
                ctx.beginPath()
                for (let x = 0; x < w; x++) {
                    const xn = (x - w * 0.65) / (w * 0.12)
                    const gauss = Math.exp(-0.5 * xn * xn)
                    const y = h - 10 - gauss * (h - 30) * (0.8 + Math.sin(t * 0.03) * 0.2)
                    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
                }
                ctx.stroke()
                ctx.shadowBlur = 0
                // Fill under curve
                ctx.lineTo(w, h - 10)
                ctx.lineTo(0, h - 10)
                ctx.closePath()
                ctx.fillStyle = col.replace(')', ', 0.08)').replace('rgb', 'rgba')
                ctx.globalAlpha = 0.3
                ctx.fill()
                ctx.globalAlpha = 1
                // Threshold line
                ctx.setLineDash([5, 5])
                ctx.strokeStyle = 'rgba(255,255,255,0.2)'
                ctx.lineWidth = 1
                ctx.beginPath()
                ctx.moveTo(w * 0.5, 0)
                ctx.lineTo(w * 0.5, h)
                ctx.stroke()
                ctx.setLineDash([])
            } else if (activeStep === 6) {
                // Forensics — bar chart
                const metrics = [0.27, 0.38, 0.35, 0.4, 0.1, 0.47, 0.48]
                const barW = w / (metrics.length * 2 + 1)
                const col = steps[activeStep]?.color || '#eab308'
                metrics.forEach((val, i) => {
                    const animVal = val * Math.min(1, (0.5 + Math.sin(t * 0.04 + i) * 0.5))
                    const bx = barW * (i * 2 + 1)
                    const bh = animVal * (h - 20)
                    const gradient = ctx.createLinearGradient(bx, h - 10, bx, h - 10 - bh)
                    gradient.addColorStop(0, col)
                    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.5)')
                    ctx.fillStyle = gradient
                    ctx.beginPath()
                    ctx.roundRect(bx, h - 10 - bh, barW, bh, [4, 4, 0, 0])
                    ctx.fill()
                })
            } else if (activeStep === 7) {
                // LLM — typing effect blocks
                const col = steps[activeStep]?.color || '#34d399'
                const lines = 5
                const lineH = 10
                const gap = 8
                const startY = h / 2 - ((lines * (lineH + gap)) / 2)
                for (let i = 0; i < lines; i++) {
                    const maxW = w * (0.4 + Math.random() * 0.4)
                    const fillW = maxW * Math.min(1, (t * 0.01 - i * 0.5))
                    if (fillW > 0) {
                        ctx.fillStyle = col
                        ctx.globalAlpha = 0.15 + (i % 2) * 0.1
                        ctx.beginPath()
                        ctx.roundRect(20, startY + i * (lineH + gap), Math.max(0, fillW), lineH, 3)
                        ctx.fill()
                        ctx.globalAlpha = 1
                    }
                }
                // Cursor blink
                if (Math.floor(t * 0.06) % 2 === 0) {
                    const lastLine = Math.min(lines - 1, Math.floor(t * 0.01))
                    const curX = 22 + (w * 0.6) * Math.min(1, (t * 0.01 - lastLine * 0.5))
                    ctx.fillStyle = col
                    ctx.fillRect(Math.min(curX, w - 25), startY + lastLine * (lineH + gap), 2, lineH)
                }
            } else if (activeStep === 8 || simState === 'done') {
                // Response — JSON brackets animation
                const col = '#22d3ee'
                ctx.font = '14px JetBrains Mono, monospace'
                ctx.fillStyle = col
                ctx.globalAlpha = 0.6 + Math.sin(t * 0.05) * 0.4
                const jsonLines = [
                    '{ "verdict": "FAKE",',
                    '  "confidence": 0.8714,',
                    '  "features": 7,',
                    '  "status": "200 OK" }',
                ]
                jsonLines.forEach((line, i) => {
                    const alpha = Math.min(1, (t * 0.02 - i * 0.8))
                    if (alpha > 0) {
                        ctx.globalAlpha = alpha * 0.7
                        ctx.fillText(line, 20, 25 + i * 22)
                    }
                })
                ctx.globalAlpha = 1
            }

            t++
            animFrameRef.current = requestAnimationFrame(drawFrame)
        }

        animFrameRef.current = requestAnimationFrame(drawFrame)
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
        }
    }, [activeStep, simState])

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [logs])

    const runSimulation = useCallback(() => {
        if (simState === 'running') return
        setSimState('running')
        setActiveStep(-1)
        setCompletedSteps([])
        setLogs([])
        setProgress(0)

        let stepIndex = 0
        const totalDuration = steps.reduce((s, st) => s + st.duration, 0)
        let elapsed = 0

        const runStep = () => {
            if (stepIndex >= steps.length) {
                setSimState('done')
                setProgress(100)
                setLogs(prev => [...prev, { text: '━━━ PROCESSING COMPLETE ━━━', type: 'success' }])
                return
            }

            const step = steps[stepIndex]
            setActiveStep(stepIndex)
            setLogs(prev => [...prev, { text: `\n── ${step.title.toUpperCase()} ──`, type: 'header' }])

            // Drip logs one by one
            let logIdx = 0
            const logInterval = setInterval(() => {
                if (logIdx < step.logs.length) {
                    setLogs(prev => [...prev, {
                        text: step.logs[logIdx],
                        type: step.logs[logIdx].startsWith('✓') ? 'success' : 'log'
                    }])
                    logIdx++
                } else {
                    clearInterval(logInterval)
                }
            }, step.duration / (step.logs.length + 1))

            // Progress
            const progressInterval = setInterval(() => {
                elapsed += 50
                setProgress(Math.min(99, (elapsed / totalDuration) * 100))
            }, 50)

            timerRef.current = setTimeout(() => {
                clearInterval(progressInterval)
                setCompletedSteps(prev => [...prev, stepIndex])
                stepIndex++
                runStep()
            }, step.duration)
        }

        setTimeout(() => runStep(), 300)
    }, [simState])

    const resetSimulation = () => {
        if (timerRef.current) clearTimeout(timerRef.current)
        setSimState('idle')
        setActiveStep(-1)
        setCompletedSteps([])
        setLogs([])
        setProgress(0)
    }

    return (
        <div className="pd-sim">
            {/* Header */}
            <div className="pd-sim__header">
                <div className="pd-sim__header-left">
                    <div className={`pd-sim__status ${simState === 'running' ? 'pd-sim__status--active' : simState === 'done' ? 'pd-sim__status--done' : ''}`}>
                        <div className="pd-sim__status-dot" />
                        {simState === 'idle' ? 'Ready' : simState === 'running' ? 'Processing' : 'Complete'}
                    </div>
                    <div className="pd-sim__timer">
                        {simState === 'running' && <span className="pd-sim__timer-pulse">●</span>}
                        {progress.toFixed(0)}%
                    </div>
                </div>
                <div className="pd-sim__actions">
                    {simState !== 'running' && (
                        <button className="pd-sim__btn pd-sim__btn--start" onClick={runSimulation}>
                            {simState === 'done' ? '↻ Replay' : '▶ Start Processing'}
                        </button>
                    )}
                    {simState === 'running' && (
                        <button className="pd-sim__btn pd-sim__btn--stop" onClick={resetSimulation}>
                            ■ Stop
                        </button>
                    )}
                </div>
            </div>

            {/* Progress Bar */}
            <div className="pd-sim__progress">
                <div
                    className="pd-sim__progress-fill"
                    style={{
                        width: `${progress}%`,
                        background: simState === 'done'
                            ? 'linear-gradient(90deg, #34d399, #22d3ee)'
                            : 'linear-gradient(90deg, #6366f1, #a855f7, #ec4899)'
                    }}
                />
            </div>

            {/* Main Content - Two Panels */}
            <div className="pd-sim__body">
                {/* Left: Step List */}
                <div className="pd-sim__steps">
                    {steps.map((step, i) => (
                        <div
                            key={step.id}
                            className={`pd-sim__step
                                ${activeStep === i ? 'pd-sim__step--active' : ''}
                                ${completedSteps.includes(i) ? 'pd-sim__step--done' : ''}
                                ${activeStep > i && !completedSteps.includes(i) ? 'pd-sim__step--past' : ''}
                            `}
                            style={{ '--step-color': step.color }}
                        >
                            <div className="pd-sim__step-marker">
                                {completedSteps.includes(i) ? (
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                                        <path d="M5 13l4 4L19 7" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                ) : activeStep === i ? (
                                    <div className="pd-sim__step-spinner" />
                                ) : (
                                    <span>{step.icon}</span>
                                )}
                            </div>
                            <div className="pd-sim__step-info">
                                <div className="pd-sim__step-title">{step.title}</div>
                                <div className="pd-sim__step-sub">{step.subtitle}</div>
                            </div>
                            {activeStep === i && (
                                <div className="pd-sim__step-active-indicator" />
                            )}
                        </div>
                    ))}
                </div>

                {/* Right: Canvas + Terminal */}
                <div className="pd-sim__panel">
                    {/* Canvas Visualization */}
                    <div className="pd-sim__canvas-wrap">
                        <div className="pd-sim__canvas-label">
                            {activeStep >= 0 ? steps[activeStep]?.title : simState === 'done' ? 'Complete' : 'Waiting for input...'}
                        </div>
                        <canvas
                            ref={canvasRef}
                            width={460}
                            height={120}
                            className="pd-sim__canvas"
                        />
                    </div>

                    {/* Terminal Log */}
                    <div className="pd-sim__terminal">
                        <div className="pd-sim__terminal-bar">
                            <div className="pd-sim__terminal-dots">
                                <span /><span /><span />
                            </div>
                            <div className="pd-sim__terminal-title">processing_log</div>
                        </div>
                        <div className="pd-sim__terminal-body">
                            {logs.length === 0 && (
                                <div className="pd-sim__terminal-empty">
                                    Press "Start Processing" to begin the simulation...
                                </div>
                            )}
                            {logs.map((log, i) => (
                                <div
                                    key={i}
                                    className={`pd-sim__log pd-sim__log--${log.type}`}
                                    style={{ animationDelay: `${i * 30}ms` }}
                                >
                                    {log.text}
                                </div>
                            ))}
                            <div ref={logsEndRef} />
                        </div>
                    </div>

                    {/* Final Verdict Card */}
                    {simState === 'done' && (
                        <div className="pd-sim__verdict animate-in">
                            <div className="pd-sim__verdict-header">
                                <div className="pd-sim__verdict-dot" />
                                <span className="pd-sim__verdict-label">FAKE</span>
                                <span className="pd-sim__verdict-badge">Deepfake Detected</span>
                            </div>
                            <div className="pd-sim__verdict-gauge">
                                <div className="pd-sim__verdict-gauge-track">
                                    <div className="pd-sim__verdict-gauge-fill" />
                                </div>
                                <div className="pd-sim__verdict-gauge-text">87.14%</div>
                            </div>
                            <p className="pd-sim__verdict-reason">
                                "The wav2vec2 neural network detected synthetic speech patterns
                                with 87% confidence. Key anomalies: flat pitch variation (12.3 Hz),
                                compressed dynamic range (5.2×), unnaturally clean silence."
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}


export default function ProjectDetails() {
    const sectionsRef = useRef([])

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('pd-visible')
                    }
                })
            },
            { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
        )

        sectionsRef.current.forEach((el) => {
            if (el) observer.observe(el)
        })

        return () => observer.disconnect()
    }, [])

    const addRef = (el) => {
        if (el && !sectionsRef.current.includes(el)) {
            sectionsRef.current.push(el)
        }
    }

    return (
        <div className="pd">
            {/* ─── Hero ─── */}
            <section className="pd-hero">
                <div className="pd-hero__glow" />
                <span className="pd-hero__badge">
                    <span className="pd-hero__badge-dot" />
                    System Architecture
                </span>
                <h1 className="pd-hero__title">
                    How Polyglot Ghost<br />Detects Deepfakes
                </h1>
                <p className="pd-hero__sub">
                    A complete technical walkthrough of our AI-powered voice authentication
                    platform — from audio input to forensic verdict.
                </p>
                <div className="pd-hero__scroll-hint">
                    <div className="pd-hero__scroll-mouse">
                        <div className="pd-hero__scroll-wheel" />
                    </div>
                    <span>Scroll to explore</span>
                </div>
            </section>

            {/* ─── Architecture Overview ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Architecture</div>
                <h2 className="pd-section__title">System Overview</h2>
                <p className="pd-section__desc">
                    Polyglot Ghost uses a modern, decoupled microservices architecture designed
                    for scalability, security, and real-time performance.
                </p>

                <div className="pd-arch">
                    <div className="pd-arch__layer pd-arch__layer--frontend">
                        <div className="pd-arch__layer-header">
                            <div className="pd-arch__layer-icon">🖥️</div>
                            <div>
                                <div className="pd-arch__layer-name">Presentation Layer</div>
                                <div className="pd-arch__layer-tech">React + Vite · Vercel CDN</div>
                            </div>
                        </div>
                        <div className="pd-arch__layer-items">
                            <div className="pd-arch__chip">WebAudio Capture</div>
                            <div className="pd-arch__chip">GSAP Animations</div>
                            <div className="pd-arch__chip">PDF Reports</div>
                            <div className="pd-arch__chip">Glassmorphism UI</div>
                        </div>
                    </div>

                    <div className="pd-arch__connector">
                        <div className="pd-arch__connector-line" />
                        <div className="pd-arch__connector-label">HTTPS · FormData</div>
                        <div className="pd-arch__connector-line" />
                    </div>

                    <div className="pd-arch__layer pd-arch__layer--ai">
                        <div className="pd-arch__layer-header">
                            <div className="pd-arch__layer-icon">🧠</div>
                            <div>
                                <div className="pd-arch__layer-name">AI Inference Engine</div>
                                <div className="pd-arch__layer-tech">Python · FastAPI · PyTorch</div>
                            </div>
                        </div>
                        <div className="pd-arch__layer-items">
                            <div className="pd-arch__chip">wav2vec2 Model</div>
                            <div className="pd-arch__chip">Bayesian Scoring</div>
                            <div className="pd-arch__chip">Acoustic Forensics</div>
                            <div className="pd-arch__chip">librosa DSP</div>
                        </div>
                    </div>

                    <div className="pd-arch__connector">
                        <div className="pd-arch__connector-line" />
                        <div className="pd-arch__connector-label">REST API · JSON</div>
                        <div className="pd-arch__connector-line" />
                    </div>

                    <div className="pd-arch__layer pd-arch__layer--llm">
                        <div className="pd-arch__layer-header">
                            <div className="pd-arch__layer-icon">💬</div>
                            <div>
                                <div className="pd-arch__layer-name">Explainable AI</div>
                                <div className="pd-arch__layer-tech">Groq API · Llama 3 8B</div>
                            </div>
                        </div>
                        <div className="pd-arch__layer-items">
                            <div className="pd-arch__chip">Forensic Reasoning</div>
                            <div className="pd-arch__chip">Local Fallback Engine</div>
                            <div className="pd-arch__chip">Zero-API Capable</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ─── Behind the Scenes — Interactive Animated Simulator ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Interactive</div>
                <h2 className="pd-section__title">Behind the Scenes</h2>
                <p className="pd-section__desc">
                    Watch exactly what happens when you submit audio for analysis.
                    Click "Start Processing" to see each step animate in real-time.
                </p>
                <ProcessingSimulator />
            </section>

            {/* ─── ML Pipeline ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">ML Pipeline</div>
                <h2 className="pd-section__title">Detection Pipeline</h2>
                <p className="pd-section__desc">
                    Every audio file passes through a 5-stage pipeline — from raw bytes to a
                    human-readable forensic verdict in under 3 seconds.
                </p>

                <div className="pd-pipeline">
                    {[
                        {
                            step: '01',
                            title: 'Audio Ingestion',
                            desc: 'Raw audio file (MP3, WAV, WebM, OGG, FLAC) is uploaded or recorded via browser microphone using WebAudio API.',
                            icon: '🎙️',
                            color: '#6366f1',
                            details: ['Any format accepted', 'Max 10MB file size', 'In-memory processing']
                        },
                        {
                            step: '02',
                            title: 'Preprocessing',
                            desc: 'Audio is decoded using librosa, resampled to 16kHz mono, and normalized to [-1, 1] range for consistent analysis.',
                            icon: '⚡',
                            color: '#a855f7',
                            details: ['16kHz sample rate', 'Mono channel', 'Amplitude normalization']
                        },
                        {
                            step: '03',
                            title: 'wav2vec2 Inference',
                            desc: 'Preprocessed waveform is fed through HuggingFace wav2vec2 transformer model for deep feature extraction and binary classification.',
                            icon: '🧬',
                            color: '#ec4899',
                            details: ['400MB pretrained model', '2-class softmax', 'Tensor extraction']
                        },
                        {
                            step: '04',
                            title: 'Bayesian Scoring',
                            desc: 'Raw model output is calibrated using Gaussian smoothing and Bayesian posterior probability to produce a confidence score.',
                            icon: '📊',
                            color: '#f97316',
                            details: ['Gaussian smoothing', 'Posterior probability', '0.50–0.99 range']
                        },
                        {
                            step: '05',
                            title: 'Forensic Report',
                            desc: 'Acoustic metrics + verdict are sent to Groq LLM (or local engine) to generate a human-readable forensic explanation.',
                            icon: '📄',
                            color: '#34d399',
                            details: ['7 forensic metrics', 'LLM reasoning', 'Offline fallback']
                        }
                    ].map((stage, i) => (
                        <div className="pd-stage" key={i} style={{ '--stage-color': stage.color }}>
                            <div className="pd-stage__marker">
                                <div className="pd-stage__number">{stage.step}</div>
                                {i < 4 && <div className="pd-stage__line" />}
                            </div>
                            <div className="pd-stage__content">
                                <div className="pd-stage__icon">{stage.icon}</div>
                                <h3 className="pd-stage__title">{stage.title}</h3>
                                <p className="pd-stage__desc">{stage.desc}</p>
                                <div className="pd-stage__details">
                                    {stage.details.map((d, j) => (
                                        <span className="pd-stage__tag" key={j}>{d}</span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ─── Acoustic Forensics ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Forensics</div>
                <h2 className="pd-section__title">Acoustic Analysis</h2>
                <p className="pd-section__desc">
                    Beyond deep learning, we extract 7 handcrafted forensic features that
                    reveal microscopic anomalies in synthetic speech.
                </p>

                <div className="pd-forensics">
                    {[
                        { name: 'Pitch Standard Deviation', unit: 'Hz', desc: 'Measures how much the voice pitch varies. Deepfakes tend to have unnaturally flat or uniform pitch.', fake: '~12 Hz', real: '~45 Hz' },
                        { name: 'Spectral Centroid Std', unit: 'Hz', desc: 'Tracks the "brightness" variation of the audio. Synthetic voices have compressed spectral characteristics.', fake: '~450 Hz', real: '~1200 Hz' },
                        { name: 'RMS Dynamic Range', unit: '×', desc: 'Ratio between loudest and quietest parts. AI voices typically have narrow dynamic range.', fake: '~5×', real: '~15×' },
                        { name: 'Spectral Flatness', unit: '', desc: 'Measures how noise-like vs tonal the spectrum is. Values near 0 = tonal, near 1 = noise.', fake: '~0.008', real: '~0.02' },
                        { name: 'Silence Noise Level', unit: 'dB', desc: 'Background noise in silent segments. AI audio often has unnaturally clean silence.', fake: '~0.001', real: '~0.01' },
                        { name: 'HF / LF Ratio', unit: '', desc: 'Balance between high and low frequencies. Deepfakes often lack natural high-frequency content.', fake: '~0.07', real: '~0.15' },
                        { name: 'Zero Crossing Std', unit: '', desc: 'How often the waveform crosses zero. Relates to voice texture and breathiness.', fake: '~0.12', real: '~0.25' },
                    ].map((f, i) => (
                        <div className="pd-metric" key={i}>
                            <div className="pd-metric__index">{String(i + 1).padStart(2, '0')}</div>
                            <div className="pd-metric__body">
                                <h4 className="pd-metric__name">{f.name}</h4>
                                <p className="pd-metric__desc">{f.desc}</p>
                                <div className="pd-metric__compare">
                                    <div className="pd-metric__val pd-metric__val--fake">
                                        <span className="pd-metric__val-label">Fake</span>
                                        <span className="pd-metric__val-num">{f.fake}</span>
                                    </div>
                                    <div className="pd-metric__val pd-metric__val--real">
                                        <span className="pd-metric__val-label">Real</span>
                                        <span className="pd-metric__val-num">{f.real}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ─── Tech Stack ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Stack</div>
                <h2 className="pd-section__title">Technology Stack</h2>
                <p className="pd-section__desc">
                    Built with modern, production-grade technologies across the entire stack.
                </p>

                <div className="pd-stack">
                    {[
                        { cat: 'Frontend', items: [
                            { name: 'React 18', desc: 'Component-based UI library' },
                            { name: 'Vite 5', desc: 'Lightning-fast build tool' },
                            { name: 'GSAP', desc: 'Professional-grade animation' },
                            { name: 'jsPDF', desc: 'Client-side PDF generation' },
                            { name: 'React Router', desc: 'Client-side routing' },
                        ]},
                        { cat: 'AI Engine', items: [
                            { name: 'FastAPI', desc: 'Async Python web framework' },
                            { name: 'PyTorch', desc: 'Deep learning framework' },
                            { name: 'wav2vec2', desc: 'Self-supervised speech model' },
                            { name: 'librosa', desc: 'Audio analysis & DSP' },
                            { name: 'Groq API', desc: 'Ultra-fast LLM inference' },
                        ]},
                        { cat: 'DevOps', items: [
                            { name: 'Docker', desc: 'Containerized deployment' },
                            { name: 'Vercel', desc: 'Frontend edge CDN' },
                            { name: 'HF Spaces', desc: 'AI model hosting' },
                            { name: 'GitHub Actions', desc: 'CI/CD pipeline' },
                            { name: 'pytest', desc: '34 unit tests' },
                        ]},
                    ].map((group, i) => (
                        <div className="pd-stack__group" key={i}>
                            <h3 className="pd-stack__cat">{group.cat}</h3>
                            <div className="pd-stack__items">
                                {group.items.map((item, j) => (
                                    <div className="pd-stack__item" key={j}>
                                        <div className="pd-stack__name">{item.name}</div>
                                        <div className="pd-stack__desc">{item.desc}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ─── Data Flow Diagram ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Data Flow</div>
                <h2 className="pd-section__title">Request Lifecycle</h2>
                <p className="pd-section__desc">
                    Follow a single audio analysis request from the user's browser
                    all the way through the system and back.
                </p>

                <div className="pd-flow">
                    {[
                        { icon: '👤', label: 'User', detail: 'Records audio or uploads file via browser' },
                        { icon: '🌐', label: 'Frontend', detail: 'Captures audio blob → FormData POST' },
                        { icon: '🚀', label: 'FastAPI', detail: 'Validates file, reads bytes into memory' },
                        { icon: '🔊', label: 'librosa', detail: 'Decodes audio → 16kHz mono waveform' },
                        { icon: '🧬', label: 'wav2vec2', detail: 'Neural network extracts speech features' },
                        { icon: '📐', label: 'Bayesian', detail: 'Gaussian-calibrated confidence score' },
                        { icon: '🔬', label: 'Forensics', detail: '7 acoustic metrics extracted' },
                        { icon: '💬', label: 'Groq LLM', detail: 'Generates forensic explanation text' },
                        { icon: '📊', label: 'Response', detail: 'JSON verdict + confidence + reasoning' },
                        { icon: '📄', label: 'Report', detail: 'User downloads PDF forensic report' },
                    ].map((node, i) => (
                        <div className="pd-flow__node" key={i}>
                            <div className="pd-flow__icon">{node.icon}</div>
                            <div className="pd-flow__label">{node.label}</div>
                            <div className="pd-flow__detail">{node.detail}</div>
                            {i < 9 && (
                                <div className="pd-flow__arrow">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                        <path d="M12 4L12 20M12 20L6 14M12 20L18 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </section>

            {/* ─── Privacy & Security ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Security</div>
                <h2 className="pd-section__title">Privacy by Design</h2>
                <p className="pd-section__desc">
                    Built from the ground up with privacy as a core architectural principle, not an afterthought.
                </p>

                <div className="pd-privacy">
                    {[
                        { icon: '🔒', title: 'Zero Data Retention', desc: 'Audio is processed entirely in RAM and immediately garbage-collected. Nothing is ever stored to disk or a database.' },
                        { icon: '🛡️', title: 'Stateless Architecture', desc: 'No database. No user accounts. No cookies. No tracking. Every request is completely independent.' },
                        { icon: '🔑', title: 'API Key Protection', desc: 'Groq API key is stored in a .env file on the server, never committed to source control or exposed to the frontend.' },
                        { icon: '🌍', title: 'CORS Configuration', desc: 'Cross-Origin Resource Sharing is configurable so you can lock down which domains can access the API.' },
                    ].map((item, i) => (
                        <div className="pd-privacy__card" key={i}>
                            <div className="pd-privacy__icon">{item.icon}</div>
                            <h3 className="pd-privacy__title">{item.title}</h3>
                            <p className="pd-privacy__desc">{item.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ─── API Endpoints ─── */}
            <section className="pd-section" ref={addRef}>
                <div className="pd-section__label">Endpoints</div>
                <h2 className="pd-section__title">API Reference</h2>
                <p className="pd-section__desc">
                    The core API endpoints that power Polyglot Ghost's detection capabilities.
                </p>

                <div className="pd-api">
                    <div className="pd-api__card">
                        <div className="pd-api__method pd-api__method--post">POST</div>
                        <div className="pd-api__route">/analyze</div>
                        <p className="pd-api__desc">
                            Upload audio for deepfake detection. Accepts multipart/form-data with an
                            <code>audio</code> field. Returns verdict, confidence, forensic reasoning,
                            feature breakdown, and timestamp.
                        </p>
                        <div className="pd-api__response">
                            <div className="pd-api__response-label">Response</div>
                            <pre className="pd-api__code">{`{
  "verdict": "FAKE",
  "confidence": 0.87,
  "reasoning": "Synthetic speech patterns...",
  "features_analyzed": 7,
  "feature_breakdown": {
    "pitch_std_hz": 12.3,
    "spectral_centroid_std": 450.2,
    "rms_dynamic_range": 5.2
  },
  "timestamp": "2026-03-30T00:00:00Z"
}`}</pre>
                        </div>
                    </div>

                    <div className="pd-api__card">
                        <div className="pd-api__method pd-api__method--get">GET</div>
                        <div className="pd-api__route">/health</div>
                        <p className="pd-api__desc">
                            Health check endpoint for monitoring. Returns service status and
                            can be used by Docker healthchecks and load balancers.
                        </p>
                        <div className="pd-api__response">
                            <div className="pd-api__response-label">Response</div>
                            <pre className="pd-api__code">{`{
  "status": "ok",
  "service": "polyglot-ghost-ai"
}`}</pre>
                        </div>
                    </div>
                </div>
            </section>

            {/* ─── Bottom CTA ─── */}
            <section className="pd-section pd-cta" ref={addRef}>
                <div className="pd-cta__glow" />
                <h2 className="pd-cta__title">Ready to detect deepfakes?</h2>
                <p className="pd-cta__desc">
                    Try it now — record your voice or upload a suspicious audio file.
                </p>
                <div className="pd-cta__buttons">
                    <a href="/record" className="pd-cta__btn pd-cta__btn--primary">
                        🎙️ Record Voice
                    </a>
                    <a href="/upload" className="pd-cta__btn pd-cta__btn--secondary">
                        📁 Upload File
                    </a>
                </div>
            </section>
        </div>
    )
}
