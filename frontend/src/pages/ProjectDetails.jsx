import { useEffect, useRef } from 'react'

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
