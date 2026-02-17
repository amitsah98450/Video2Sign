function Header({ currentView, onReset, onBack }) {
    const viewLabels = {
        upload: "Upload Video",
        transcription: "Transcription",
        sign: "Sign Language",
    };

    return (
        <header className="app-header">
            <div className="header-inner">
                <div className="header-brand" onClick={onReset}>
                    <span className="header-logo">🤟</span>
                    <span className="header-title">Video to Sign</span>
                </div>

                <div className="header-actions">
                    {currentView !== "upload" && (
                        <div className="header-step">
                            <span className="header-step-dot" />
                            {viewLabels[currentView]}
                        </div>
                    )}

                    {onBack && (
                        <button className="btn-icon" onClick={onBack}>
                            ← Back
                        </button>
                    )}

                    {currentView !== "upload" && (
                        <button className="btn-icon" onClick={onReset}>
                            ↺ New
                        </button>
                    )}
                </div>
            </div>
        </header>
    );
}

export default Header;
