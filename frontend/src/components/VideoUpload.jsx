import { useState, useRef } from "react";
import { uploadVideo } from "../api";

const ALLOWED_TYPES = [
    "video/mp4", "video/avi", "video/quicktime", "video/x-matroska",
    "video/webm", "audio/mpeg", "audio/wav", "audio/mp4", "audio/ogg",
];

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function VideoUpload({ onComplete }) {
    const [file, setFile] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState("");
    const [error, setError] = useState(null);
    const inputRef = useRef(null);

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) selectFile(droppedFile);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragOver(true);
    };

    const handleDragLeave = () => setDragOver(false);

    const selectFile = (f) => {
        setError(null);
        if (!ALLOWED_TYPES.includes(f.type) && !f.name.match(/\.(mp4|avi|mov|mkv|webm|mp3|wav|m4a|ogg)$/i)) {
            setError("Unsupported file type. Please upload a video or audio file.");
            return;
        }
        setFile(f);
    };

    const handleFileChange = (e) => {
        if (e.target.files[0]) selectFile(e.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setProgress(0);
        setError(null);
        setStatus("Uploading video...");

        try {
            const data = await uploadVideo(file, (pct) => {
                setProgress(pct);
                if (pct === 100) {
                    setStatus("Transcribing with Whisper AI... This may take a moment.");
                }
            });
            setStatus("Done!");
            onComplete(data);
        } catch (err) {
            setError(err.message);
            setUploading(false);
            setProgress(0);
            setStatus("");
        }
    };

    return (
        <div className="upload-section">
            <div className="upload-hero">
                <h1>
                    Convert Video to <span className="gradient-text">Sign Language</span>
                </h1>
                <p>
                    Upload a video or audio file and we'll transcribe it, then convert the
                    text into animated sign language.
                </p>
            </div>

            <div className="glass-card">
                {!uploading ? (
                    <>
                        <div
                            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onClick={() => inputRef.current?.click()}
                        >
                            <div className="upload-zone-content">
                                <span className="upload-icon">📁</span>
                                <h3>Drop your video here</h3>
                                <p>
                                    or <span className="browse-link">browse files</span> to upload
                                </p>
                                <div className="upload-formats">
                                    {["MP4", "AVI", "MOV", "MKV", "WEBM", "MP3", "WAV"].map((fmt) => (
                                        <span key={fmt} className="format-tag">{fmt}</span>
                                    ))}
                                </div>
                            </div>
                            <input
                                ref={inputRef}
                                type="file"
                                accept="video/*,audio/*"
                                onChange={handleFileChange}
                                style={{ display: "none" }}
                            />
                        </div>

                        {file && (
                            <div className="selected-file">
                                <div className="selected-file-info">
                                    <span className="selected-file-icon">🎬</span>
                                    <div>
                                        <div className="selected-file-name">{file.name}</div>
                                        <div className="selected-file-size">
                                            {formatFileSize(file.size)}
                                        </div>
                                    </div>
                                </div>
                                <button
                                    className="btn-remove"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setFile(null);
                                    }}
                                >
                                    ✕
                                </button>
                            </div>
                        )}

                        {file && (
                            <div className="upload-action">
                                <button className="btn btn-primary" onClick={handleUpload}>
                                    🚀 Transcribe Video
                                </button>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="progress-section">
                        <div className="progress-header">
                            <span className="progress-title">
                                {progress < 100 ? "Uploading..." : "Transcribing..."}
                            </span>
                            <span className="progress-percent">{progress}%</span>
                        </div>
                        <div className="progress-bar-wrapper">
                            <div
                                className="progress-bar-fill"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <div className="progress-status">
                            <span className="spinner" />
                            {status}
                        </div>
                    </div>
                )}

                {error && (
                    <div className="error-message">
                        ⚠️ {error}
                    </div>
                )}
            </div>
        </div>
    );
}

export default VideoUpload;
