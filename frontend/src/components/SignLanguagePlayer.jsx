import { useState, useEffect, useRef, useCallback } from "react";
import { getSignAssetUrl } from "../api";

function SignLanguagePlayer({ signData }) {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1.5); // seconds per sign
    const [letterIndex, setLetterIndex] = useState(0); // for fingerspelling
    const timerRef = useRef(null);

    const signs = signData.signs || [];
    const currentSign = signs[currentIndex];
    const totalSigns = signs.length;

    // Calculate the actual display — if fingerspelling, cycle through letters
    const getCurrentDisplay = useCallback(() => {
        if (!currentSign) return { letter: "", assetUrl: null };

        if (currentSign.type === "fingerspell") {
            const letters = currentSign.text.split("");
            const idx = letterIndex % letters.length;
            const letter = letters[idx]?.toUpperCase() || "";
            const assetFile = currentSign.assets[idx];
            return {
                letter,
                assetUrl: assetFile ? getSignAssetUrl(assetFile) : null,
                isLetter: true,
            };
        }

        return {
            letter: currentSign.text,
            assetUrl: currentSign.assets[0]
                ? getSignAssetUrl(currentSign.assets[0])
                : null,
            isLetter: false,
        };
    }, [currentSign, letterIndex]);

    // Advance playback
    useEffect(() => {
        if (!isPlaying || totalSigns === 0) return;

        const currentDisplay = getCurrentDisplay();

        timerRef.current = setTimeout(() => {
            if (currentSign?.type === "fingerspell") {
                const totalLetters = currentSign.text.length;
                if (letterIndex < totalLetters - 1) {
                    setLetterIndex((prev) => prev + 1);
                    return;
                }
            }

            // Move to next sign
            setLetterIndex(0);
            if (currentIndex < totalSigns - 1) {
                setCurrentIndex((prev) => prev + 1);
            } else {
                setIsPlaying(false); // End of sequence
            }
        }, currentSign?.type === "fingerspell" ? speed * 400 : speed * 1000);

        return () => clearTimeout(timerRef.current);
    }, [isPlaying, currentIndex, letterIndex, speed, totalSigns, currentSign, getCurrentDisplay]);

    const handlePlayPause = () => {
        if (!isPlaying && currentIndex >= totalSigns - 1) {
            // Restart from beginning
            setCurrentIndex(0);
            setLetterIndex(0);
        }
        setIsPlaying(!isPlaying);
    };

    const handlePrev = () => {
        setIsPlaying(false);
        setLetterIndex(0);
        setCurrentIndex((prev) => Math.max(0, prev - 1));
    };

    const handleNext = () => {
        setIsPlaying(false);
        setLetterIndex(0);
        setCurrentIndex((prev) => Math.min(totalSigns - 1, prev + 1));
    };

    const display = getCurrentDisplay();
    const progressPercent = totalSigns > 0
        ? ((currentIndex + 1) / totalSigns) * 100
        : 0;

    return (
        <div className="sign-section">
            <div className="section-header">
                <div>
                    <h2>🤟 Sign Language Playback</h2>
                    <p>{totalSigns} signs generated from your text</p>
                </div>
            </div>

            <div className="glass-card sign-player-card">
                <div className="sign-display">
                    <div className="sign-image-wrapper">
                        {display.assetUrl ? (
                            <img
                                src={display.assetUrl}
                                alt={display.letter}
                                onError={(e) => {
                                    e.target.style.display = "none";
                                    e.target.nextSibling.style.display = "flex";
                                }}
                            />
                        ) : null}
                        <div
                            className="sign-placeholder"
                            style={{ display: display.assetUrl ? "none" : "flex" }}
                        >
                            {display.letter?.charAt(0)?.toUpperCase() || "?"}
                        </div>
                    </div>

                    <div className="sign-current-word">{display.letter}</div>

                    {currentSign && (
                        <span className="sign-type-badge">
                            {currentSign.type === "fingerspell"
                                ? `Fingerspelling: ${currentSign.text}`
                                : "Sign Word"}
                        </span>
                    )}

                    <div className="sign-controls">
                        <button
                            className="btn-control"
                            onClick={handlePrev}
                            disabled={currentIndex === 0}
                        >
                            ⏮
                        </button>
                        <button className="btn-play" onClick={handlePlayPause}>
                            {isPlaying ? "⏸" : "▶"}
                        </button>
                        <button
                            className="btn-control"
                            onClick={handleNext}
                            disabled={currentIndex >= totalSigns - 1}
                        >
                            ⏭
                        </button>
                    </div>

                    <div className="sign-progress">
                        <div className="sign-progress-bar">
                            <div
                                className="sign-progress-fill"
                                style={{ width: `${progressPercent}%` }}
                            />
                        </div>
                        <span className="sign-progress-text">
                            {currentIndex + 1} / {totalSigns}
                        </span>
                    </div>

                    <div className="speed-control">
                        <span className="speed-label">Speed:</span>
                        <input
                            type="range"
                            className="speed-slider"
                            min="0.5"
                            max="3"
                            step="0.25"
                            value={speed}
                            onChange={(e) => setSpeed(parseFloat(e.target.value))}
                        />
                        <span className="speed-label">{speed}s</span>
                    </div>
                </div>
            </div>

            {/* Word Queue */}
            <div className="glass-card word-queue">
                <div className="word-queue-title">Sign Sequence</div>
                <div className="word-queue-list">
                    {signs.map((sign, i) => (
                        <span
                            key={i}
                            className={`word-chip ${i === currentIndex ? "active" : i < currentIndex ? "done" : ""
                                }`}
                            onClick={() => {
                                setCurrentIndex(i);
                                setLetterIndex(0);
                                setIsPlaying(false);
                            }}
                            style={{ cursor: "pointer" }}
                        >
                            {sign.type === "fingerspell" ? `🔤 ${sign.text}` : sign.text}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default SignLanguagePlayer;
