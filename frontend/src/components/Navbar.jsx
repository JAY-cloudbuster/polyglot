import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
    const { pathname } = useLocation()
    const [showAbout, setShowAbout] = useState(false)

    return (
        <>
            <nav className="navbar">
                <div className="navbar__inner">
                    <Link to="/" className="navbar__brand">
                        <div className="navbar__logo">P</div>
                        Polyglot Ghost
                    </Link>

                    <div className="navbar__right">
                        <Link
                            to="/record"
                            className={`navbar__link ${pathname === '/record' ? 'navbar__link--active' : ''}`}
                        >
                            Record
                        </Link>
                        <Link
                            to="/upload"
                            className={`navbar__link ${pathname === '/upload' ? 'navbar__link--active' : ''}`}
                        >
                            Upload
                        </Link>
                        <button
                            className="about-trigger"
                            onClick={() => setShowAbout(true)}
                            title="About"
                        >
                            ?
                        </button>
                    </div>
                </div>
            </nav>

            {showAbout && (
                <div className="about-overlay" onClick={() => setShowAbout(false)}>
                    <div className="about-modal animate-in" onClick={e => e.stopPropagation()}>
                        <button className="about-modal__close" onClick={() => setShowAbout(false)}>×</button>

                        <div className="about-modal__avatar">K</div>
                        <div className="about-modal__name">Kjaye</div>
                        <div className="about-modal__role">developer / builder</div>
                        <p className="about-modal__bio">
                            Building Polyglot Ghost — an AI-powered voice deepfake
                            detection platform. Using deep learning (wav2vec2) and
                            LLM reasoning to protect against synthetic voice fraud.
                        </p>
                        <div className="about-modal__links">
                            <a
                                href="https://github.com/JAY-cloudbuster"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="about-modal__link"
                            >
                                GitHub
                            </a>
                            <a
                                href="https://github.com/JAY-cloudbuster/polyglot"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="about-modal__link"
                            >
                                Source
                            </a>
                        </div>
                    </div>
                </div>
            )}
        </>
    )
}
