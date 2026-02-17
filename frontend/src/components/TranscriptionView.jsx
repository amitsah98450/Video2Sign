import { useState } from "react";
import { textToSign } from "../api";

function TranscriptionView({ transcription, onConvertToSign }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleConvert = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await textToSign(transcription.full_text);
            onConvertToSign(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="transcription-section">
            <div className="section-header">
                <div>
                    <h2>📝 Transcription Result</h2>
                    <p>
                        {transcription.segments.length} segments detected
                    </p>
                </div>
                {transcription.language && transcription.language !== "unknown" && (
                    <span className="lang-badge">
                        🌐 {transcription.language}
                    </span>
                )}
            </div>

            <div className="segments-list">
                {transcription.segments.map((seg) => (
                    <div key={seg.id} className="segment-item">
                        <div className="segment-time">
                            🕐 {seg.start_formatted} → {seg.end_formatted}
                        </div>
                        <div className="segment-text">{seg.text}</div>
                    </div>
                ))}
            </div>

            <div className="transcription-actions">
                <button
                    className="btn btn-primary"
                    onClick={handleConvert}
                    disabled={loading}
                >
                    {loading ? (
                        <>
                            <span className="spinner" /> Converting...
                        </>
                    ) : (
                        "🤟 Convert to Sign Language"
                    )}
                </button>
            </div>

            {error && (
                <div className="error-message">⚠️ {error}</div>
            )}
        </div>
    );
}

export default TranscriptionView;
