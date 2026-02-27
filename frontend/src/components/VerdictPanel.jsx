/**
 * VerdictPanel — displays REAL or FAKE verdict with reasoning.
 */
export default function VerdictPanel({ verdict, confidence, reasoning }) {
    if (!verdict) return null

    const isReal = verdict === 'REAL'
    const pct = (confidence * 100).toFixed(1)

    return (
        <div className={`verdict ${isReal ? 'verdict--real' : 'verdict--fake'}`}>
            <div className="verdict__header">
                <div className="verdict__dot" />
                <span className="verdict__label">{verdict}</span>
            </div>
            <p className="verdict__confidence">
                {isReal
                    ? `Genuine human speech detected — ${pct}% confidence`
                    : `Synthetic or AI-generated speech detected — ${pct}% confidence`}
            </p>
            {reasoning && (
                <p className="verdict__reasoning">{reasoning}</p>
            )}
        </div>
    )
}
