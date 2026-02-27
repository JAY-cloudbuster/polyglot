/**
 * ConfidenceMetrics — circular gauge + feature breakdown table.
 */
export default function ConfidenceMetrics({ confidence, verdict, featuresAnalyzed, featureBreakdown }) {
    if (confidence == null) return null

    const isReal = verdict === 'REAL'
    const pct = (confidence * 100).toFixed(1)

    // SVG circular gauge
    const radius = 48
    const circumference = 2 * Math.PI * radius
    const offset = circumference - (confidence * circumference)

    return (
        <div id="confidence-metrics" className="glass-card metrics">
            {/* Circular Gauge */}
            <div className="metrics__gauge">
                <svg className="metrics__gauge-svg" viewBox="0 0 120 120">
                    <circle
                        className="metrics__gauge-bg"
                        cx="60" cy="60" r={radius}
                    />
                    <circle
                        className={`metrics__gauge-fill ${isReal ? 'metrics__gauge-fill--real' : 'metrics__gauge-fill--fake'}`}
                        cx="60" cy="60" r={radius}
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                    />
                </svg>
                <div className="metrics__gauge-text">
                    <div className="metrics__gauge-value" style={{ color: isReal ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {pct}%
                    </div>
                    <div className="metrics__gauge-label">Confidence</div>
                </div>
            </div>

            {/* Feature Details */}
            <div className="metrics__details">
                <div className="section-label">Feature Analysis</div>

                {featuresAnalyzed && (
                    <div className="metrics__detail-row">
                        <span className="metrics__detail-name">Features analyzed</span>
                        <span className="metrics__detail-value">{featuresAnalyzed}</span>
                    </div>
                )}

                {featureBreakdown && Object.entries(featureBreakdown).map(([name, value]) => (
                    <div className="metrics__detail-row" key={name}>
                        <span className="metrics__detail-name">{name}</span>
                        <span className="metrics__detail-value">{value}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}
