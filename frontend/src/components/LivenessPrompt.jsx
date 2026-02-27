import { useState } from 'react'
import { checkLiveness } from '../services/api'

/**
 * LivenessPrompt — semantic liveness verification via text input.
 */
export default function LivenessPrompt() {
    const [text, setText] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const prompt = 'Please say your name and today\'s date.'

    const handleSubmit = async () => {
        if (!text.trim()) return

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await checkLiveness(text.trim(), prompt)
            setResult(res)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit()
        }
    }

    return (
        <div id="liveness-prompt" className="glass-card liveness">
            <div className="section-label">Liveness Verification</div>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                Prompt: <em>"{prompt}"</em>
            </p>

            <div className="liveness__input-row">
                <input
                    className="liveness__input"
                    type="text"
                    placeholder="Type the spoken response here..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={loading}
                />
                <button
                    className="liveness__submit"
                    onClick={handleSubmit}
                    disabled={loading || !text.trim()}
                >
                    {loading ? 'Checking...' : 'Verify'}
                </button>
            </div>

            {error && (
                <div className="error-msg" style={{ marginTop: '0.75rem' }}>
                    <span className="error-msg__icon">⚠️</span>
                    {error}
                </div>
            )}

            {result && (
                <div className="liveness__result">
                    <div className="liveness__score">
                        Relevance: <strong style={{ color: result.relevance_score > 0.5 ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
                            {(result.relevance_score * 100).toFixed(0)}%
                        </strong>
                        {' · '}
                        Live: <strong style={{ color: result.is_live ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                            {result.is_live ? 'YES' : 'NO'}
                        </strong>
                        {result.source && (
                            <span style={{ marginLeft: '0.5rem', color: 'var(--text-muted)', fontSize: '0.7rem' }}>
                                ({result.source})
                            </span>
                        )}
                    </div>
                    {result.reasoning && (
                        <p className="liveness__reasoning">{result.reasoning}</p>
                    )}
                </div>
            )}
        </div>
    )
}
