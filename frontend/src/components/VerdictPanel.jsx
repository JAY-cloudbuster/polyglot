/**
 * VerdictPanel — displays REAL or FAKE verdict with AI reasoning.
 */
export default function VerdictPanel({ verdict, confidence, reasoning }) {
    if (!verdict) return null

    const isReal = verdict === 'REAL'
    const pct = (confidence * 100).toFixed(1)

    return (
        <div id="verdict-panel" className={`verdict ${isReal ? 'verdict--real' : 'verdict--fake'}`}>
            <div className="verdict__badge">
                <span className="verdict__badge-icon">{isReal ? '🛡️' : '⚠️'}</span>
                {verdict}
            </div>
            <p className="verdict__description">
                {isReal
                    ? `This audio appears to be from a genuine human speaker (${pct}% confidence).`
                    : `This audio shows characteristics of synthetic or manipulated speech (${pct}% confidence).`}
            </p>
            {reasoning && (
                <p className="verdict__reasoning" style={{
                    marginTop: '0.75rem',
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)',
                    fontStyle: 'italic',
                    lineHeight: 1.5,
                    borderTop: '1px solid rgba(255,255,255,0.08)',
                    paddingTop: '0.75rem'
                }}>
                    💡 {reasoning}
                </p>
            )}
        </div>
    )
}
