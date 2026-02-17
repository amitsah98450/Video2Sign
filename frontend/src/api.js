const API_BASE = "http://localhost:5001/api";

/**
 * Upload a video file and get transcription
 * @param {File} videoFile - The video file to upload
 * @param {Function} onProgress - Progress callback (0-100)
 * @returns {Promise<Object>} Transcription result
 */
export async function uploadVideo(videoFile, onProgress) {
    const formData = new FormData();
    formData.append("video", videoFile);

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable && onProgress) {
                onProgress(Math.round((e.loaded / e.total) * 100));
            }
        });

        xhr.addEventListener("load", () => {
            try {
                const data = JSON.parse(xhr.responseText);
                if (xhr.status === 200 && data.success) {
                    resolve(data);
                } else {
                    reject(new Error(data.error || "Upload failed"));
                }
            } catch {
                reject(new Error("Invalid response from server"));
            }
        });

        xhr.addEventListener("error", () =>
            reject(new Error("Network error — is the backend running on port 5000?"))
        );

        xhr.addEventListener("abort", () =>
            reject(new Error("Upload cancelled"))
        );

        xhr.open("POST", `${API_BASE}/upload`);
        xhr.send(formData);
    });
}

/**
 * Convert text to sign language tokens
 * @param {string} text - Text to convert
 * @returns {Promise<Object>} Sign language tokens
 */
export async function textToSign(text) {
    const res = await fetch(`${API_BASE}/text-to-sign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Conversion failed");
    return data;
}

/**
 * Get the URL for a sign asset
 * @param {string} filename - Asset filename
 * @returns {string} Full URL to the asset
 */
export function getSignAssetUrl(filename) {
    return `${API_BASE}/signs/${filename}`;
}

/**
 * Health check
 * @returns {Promise<Object>}
 */
export async function healthCheck() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
}
