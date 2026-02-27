import { useEffect, useRef } from 'react'

/**
 * RhythmBackground — subtle equalizer bars animating in the background.
 * Creates the illusion of music visualization playing.
 */
export default function RhythmBackground() {
    const canvasRef = useRef(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        let animId

        const resize = () => {
            canvas.width = window.innerWidth
            canvas.height = window.innerHeight
        }
        resize()
        window.addEventListener('resize', resize)

        const barCount = 60
        const bars = Array.from({ length: barCount }, (_, i) => ({
            x: (i / barCount) * canvas.width,
            height: Math.random() * 40 + 10,
            speed: Math.random() * 0.015 + 0.005,
            phase: Math.random() * Math.PI * 2,
            maxH: Math.random() * 60 + 20,
        }))

        const draw = (time) => {
            ctx.clearRect(0, 0, canvas.width, canvas.height)

            for (const bar of bars) {
                const t = time * bar.speed
                const h = bar.maxH * (0.3 + 0.7 * Math.abs(Math.sin(t + bar.phase)))
                const x = bar.x
                const w = canvas.width / barCount - 2

                // Bottom bars
                const gradient = ctx.createLinearGradient(x, canvas.height, x, canvas.height - h)
                gradient.addColorStop(0, 'rgba(37, 99, 235, 0.06)')
                gradient.addColorStop(1, 'rgba(37, 99, 235, 0.01)')
                ctx.fillStyle = gradient
                ctx.fillRect(x, canvas.height - h, w, h)

                // Top bars (mirrored, even more subtle)
                const topH = h * 0.5
                const gradient2 = ctx.createLinearGradient(x, 0, x, topH)
                gradient2.addColorStop(0, 'rgba(37, 99, 235, 0.04)')
                gradient2.addColorStop(1, 'rgba(37, 99, 235, 0.005)')
                ctx.fillStyle = gradient2
                ctx.fillRect(x, 0, w, topH)
            }

            animId = requestAnimationFrame(draw)
        }

        animId = requestAnimationFrame(draw)

        return () => {
            cancelAnimationFrame(animId)
            window.removeEventListener('resize', resize)
        }
    }, [])

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 0,
            }}
        />
    )
}
