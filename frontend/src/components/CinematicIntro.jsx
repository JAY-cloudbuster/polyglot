import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'

/**
 * CinematicIntro — Full-screen Polyglot Ghost preloader.
 * 
 * 5-Phase GSAP Timeline:
 *   1. Ghost SVG fades in (1.5s)
 *   2. Ghost SVG fades out (1s)
 *   3. Text fades in with blur-to-focus (2s)
 *   4. Text fades out (1.5s)
 *   5. Overlay dissolves → reveals main site
 */
export default function CinematicIntro({ onComplete }) {
    const overlayRef = useRef(null)
    const ghostRef = useRef(null)
    const textRef = useRef(null)
    const glitchLineRefs = useRef([])
    const [unmount, setUnmount] = useState(false)

    useEffect(() => {
        const ctx = gsap.context(() => {
            const tl = gsap.timeline({
                onComplete: () => {
                    setUnmount(true)
                    onComplete?.()
                },
            })

            // Initial state
            gsap.set(ghostRef.current, { opacity: 0 })
            gsap.set(textRef.current, { opacity: 0, filter: 'blur(12px)' })

            // Phase 1: Ghost fades in
            tl.to(ghostRef.current, {
                opacity: 1,
                duration: 1.5,
                ease: 'power2.inOut',
            })

            // Glitch scan lines flicker during ghost hold
            tl.to(glitchLineRefs.current, {
                opacity: 0.15,
                duration: 0.3,
                stagger: { each: 0.05, repeat: 3, yoyo: true },
                ease: 'steps(2)',
            }, '-=0.5')

            // Hold for 1s
            tl.to({}, { duration: 1 })

            // Phase 2: Ghost fades out
            tl.to(ghostRef.current, {
                opacity: 0,
                duration: 1,
                ease: 'power2.inOut',
            })

            // Phase 3: Text fades in with blur-to-focus
            tl.to(textRef.current, {
                opacity: 1,
                filter: 'blur(0px)',
                duration: 2,
                ease: 'power3.out',
            })

            // Hold for 2s
            tl.to({}, { duration: 2 })

            // Phase 4: Text fades out
            tl.to(textRef.current, {
                opacity: 0,
                duration: 1.5,
                ease: 'power2.inOut',
            })

            // Phase 5: Overlay dissolves
            tl.to(overlayRef.current, {
                opacity: 0,
                duration: 1.2,
                ease: 'power2.inOut',
            })
        })

        return () => ctx.revert()
    }, [onComplete])

    if (unmount) return null

    return (
        <div
            ref={overlayRef}
            style={{
                position: 'fixed',
                inset: 0,
                zIndex: 9999,
                background: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                overflow: 'hidden',
            }}
        >
            {/* Scan lines overlay */}
            <div style={{
                position: 'absolute',
                inset: 0,
                pointerEvents: 'none',
                background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.015) 2px, rgba(255,255,255,0.015) 4px)',
            }} />

            {/* Glitch flicker bars */}
            {[...Array(5)].map((_, i) => (
                <div
                    key={i}
                    ref={el => glitchLineRefs.current[i] = el}
                    style={{
                        position: 'absolute',
                        left: 0,
                        right: 0,
                        height: `${2 + Math.random() * 4}px`,
                        top: `${15 + i * 18}%`,
                        background: `linear-gradient(90deg, transparent 0%, rgba(0,255,180,0.08) ${20 + i * 10}%, rgba(120,0,255,0.06) ${60 + i * 5}%, transparent 100%)`,
                        opacity: 0,
                        pointerEvents: 'none',
                    }}
                />
            ))}

            {/* Ghost SVG — Circuit Board / Glitch Aesthetic */}
            <div ref={ghostRef} style={{ opacity: 0 }}>
                <svg
                    width="160"
                    height="200"
                    viewBox="0 0 160 200"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    style={{ filter: 'drop-shadow(0 0 30px rgba(0,255,180,0.25))' }}
                >
                    {/* Ghost body — pixelated circuit style */}
                    <defs>
                        <linearGradient id="ghostGrad" x1="0" y1="0" x2="160" y2="200" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stopColor="#00ffb4" />
                            <stop offset="100%" stopColor="#7c3aed" />
                        </linearGradient>
                        <filter id="glitchNoise">
                            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" result="noise" />
                            <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.5" />
                        </filter>
                    </defs>

                    {/* Outer ghost shell — pixel blocks */}
                    <g filter="url(#glitchNoise)" opacity="0.95">
                        {/* Head dome */}
                        <rect x="40" y="10" width="80" height="10" fill="url(#ghostGrad)" opacity="0.6" />
                        <rect x="30" y="20" width="100" height="10" fill="url(#ghostGrad)" opacity="0.7" />
                        <rect x="20" y="30" width="120" height="10" fill="url(#ghostGrad)" opacity="0.8" />
                        <rect x="15" y="40" width="130" height="10" fill="url(#ghostGrad)" opacity="0.85" />
                        <rect x="10" y="50" width="140" height="10" fill="url(#ghostGrad)" />

                        {/* Body */}
                        <rect x="10" y="60" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="70" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="80" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="90" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="100" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="110" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="120" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="130" width="140" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="140" width="140" height="10" fill="url(#ghostGrad)" />

                        {/* Wavy bottom tentacles */}
                        <rect x="10" y="150" width="30" height="10" fill="url(#ghostGrad)" />
                        <rect x="10" y="160" width="20" height="10" fill="url(#ghostGrad)" opacity="0.8" />
                        <rect x="10" y="170" width="10" height="10" fill="url(#ghostGrad)" opacity="0.5" />

                        <rect x="50" y="150" width="25" height="10" fill="url(#ghostGrad)" />
                        <rect x="55" y="160" width="15" height="10" fill="url(#ghostGrad)" opacity="0.7" />

                        <rect x="85" y="150" width="25" height="10" fill="url(#ghostGrad)" />
                        <rect x="90" y="160" width="15" height="10" fill="url(#ghostGrad)" opacity="0.7" />

                        <rect x="120" y="150" width="30" height="10" fill="url(#ghostGrad)" />
                        <rect x="130" y="160" width="20" height="10" fill="url(#ghostGrad)" opacity="0.8" />
                        <rect x="140" y="170" width="10" height="10" fill="url(#ghostGrad)" opacity="0.5" />
                    </g>

                    {/* Eyes — hollow circuit nodes */}
                    <rect x="40" y="65" width="22" height="18" rx="2" fill="#000" />
                    <rect x="98" y="65" width="22" height="18" rx="2" fill="#000" />
                    {/* Eye inner glow */}
                    <rect x="44" y="69" width="14" height="10" rx="1" fill="#00ffb4" opacity="0.9" />
                    <rect x="102" y="69" width="14" height="10" rx="1" fill="#00ffb4" opacity="0.9" />
                    {/* Pupil pixels */}
                    <rect x="48" y="72" width="4" height="4" fill="#000" />
                    <rect x="106" y="72" width="4" height="4" fill="#000" />

                    {/* Circuit traces inside body */}
                    <line x1="30" y1="100" x2="55" y2="100" stroke="#00ffb4" strokeWidth="1" opacity="0.3" />
                    <line x1="55" y1="100" x2="55" y2="120" stroke="#00ffb4" strokeWidth="1" opacity="0.3" />
                    <line x1="55" y1="120" x2="80" y2="120" stroke="#00ffb4" strokeWidth="1" opacity="0.3" />
                    <circle cx="80" cy="120" r="2" fill="#00ffb4" opacity="0.4" />

                    <line x1="130" y1="95" x2="105" y2="95" stroke="#7c3aed" strokeWidth="1" opacity="0.3" />
                    <line x1="105" y1="95" x2="105" y2="115" stroke="#7c3aed" strokeWidth="1" opacity="0.3" />
                    <line x1="105" y1="115" x2="80" y2="115" stroke="#7c3aed" strokeWidth="1" opacity="0.3" />
                    <circle cx="80" cy="115" r="2" fill="#7c3aed" opacity="0.4" />

                    {/* Small circuit nodes */}
                    <circle cx="35" cy="105" r="1.5" fill="#00ffb4" opacity="0.5" />
                    <circle cx="125" cy="105" r="1.5" fill="#7c3aed" opacity="0.5" />
                    <circle cx="70" cy="135" r="1.5" fill="#00ffb4" opacity="0.3" />
                    <circle cx="90" cy="135" r="1.5" fill="#7c3aed" opacity="0.3" />
                </svg>
            </div>

            {/* Title Text */}
            <div
                ref={textRef}
                style={{
                    opacity: 0,
                    position: 'absolute',
                    fontFamily: "'Inter', 'SF Mono', 'Fira Code', monospace",
                    fontSize: 'clamp(14px, 3vw, 28px)',
                    fontWeight: 700,
                    letterSpacing: '0.45em',
                    color: '#fff',
                    textTransform: 'uppercase',
                    userSelect: 'none',
                    textAlign: 'center',
                    whiteSpace: 'nowrap',
                }}
            >
                P O L Y G L O T {'  '} G H O S T
                <div style={{
                    marginTop: '16px',
                    fontSize: '0.35em',
                    fontWeight: 400,
                    letterSpacing: '0.35em',
                    color: 'rgba(0,255,180,0.5)',
                    fontFamily: "'Inter', sans-serif",
                }}>
                    ACOUSTIC DEEPFAKE DETECTION
                </div>
            </div>
        </div>
    )
}
