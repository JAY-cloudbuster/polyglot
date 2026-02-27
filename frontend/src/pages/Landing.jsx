import { Link } from 'react-router-dom'

export default function Landing() {
    return (
        <div className="landing">
            {/* Hero */}
            <section className="landing__hero">
                <span className="landing__badge">Voice Authentication</span>
                <h1 className="landing__title">
                    Detect voice deepfakes<br />before they cause harm
                </h1>
                <p className="landing__subtitle">
                    Upload an audio file or record from your microphone.
                    Our engine analyzes the waveform and tells you if
                    the voice is real or AI&#8209;generated.
                </p>
                <div className="landing__ctas">
                    <Link to="/record" className="landing__cta landing__cta--primary">
                        Record voice
                    </Link>
                    <Link to="/upload" className="landing__cta landing__cta--secondary">
                        Upload file
                    </Link>
                </div>
            </section>

            {/* Stats */}
            <div className="landing__stats fade-in">
                <div className="stat">
                    <div className="stat__value">96%</div>
                    <div className="stat__label">Accuracy</div>
                </div>
                <div className="stat">
                    <div className="stat__value">&lt;3s</div>
                    <div className="stat__label">Detection</div>
                </div>
                <div className="stat">
                    <div className="stat__value">6+</div>
                    <div className="stat__label">Formats</div>
                </div>
                <div className="stat">
                    <div className="stat__value">PDF</div>
                    <div className="stat__label">Reports</div>
                </div>
            </div>

            {/* Features */}
            <div className="landing__features fade-in">
                <div className="feature-card">
                    <div className="feature-card__icon">🎯</div>
                    <div className="feature-card__title">Deep learning</div>
                    <p className="feature-card__desc">
                        wav2vec2 model trained on real and AI&#8209;generated
                        speech. Processes raw audio waveforms directly.
                    </p>
                </div>
                <div className="feature-card">
                    <div className="feature-card__icon">💬</div>
                    <div className="feature-card__title">Explainable</div>
                    <p className="feature-card__desc">
                        Every verdict includes a forensic explanation
                        of why the audio was classified the way it was.
                    </p>
                </div>
                <div className="feature-card">
                    <div className="feature-card__icon">📄</div>
                    <div className="feature-card__title">PDF reports</div>
                    <p className="feature-card__desc">
                        Download detailed forensic reports with verdicts,
                        confidence scores, and technical measurements.
                    </p>
                </div>
            </div>

            {/* How it works */}
            <div className="landing__section-label">Process</div>
            <h2 className="landing__section-title">How it works</h2>
            <p className="landing__section-subtitle">Three steps to verify any voice</p>

            <div className="steps fade-in">
                <div className="step">
                    <div className="step__number">1</div>
                    <div className="step__title">Provide audio</div>
                    <p className="step__desc">
                        Record your microphone or upload a file in any format.
                    </p>
                </div>
                <div className="step">
                    <div className="step__number">2</div>
                    <div className="step__title">Analysis</div>
                    <p className="step__desc">
                        The engine runs the audio through a deep learning model.
                    </p>
                </div>
                <div className="step">
                    <div className="step__number">3</div>
                    <div className="step__title">Get verdict</div>
                    <p className="step__desc">
                        Receive a verdict with confidence score and download a report.
                    </p>
                </div>
            </div>
        </div>
    )
}
