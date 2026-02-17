"""
Video to Sign Language — Flask Backend
Handles video upload, Whisper transcription, and text-to-sign conversion.
"""

import os
import sys
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add the project root to sys.path so we can import from ml/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sign_dictionary import text_to_signs

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174"])

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
SIGNS_FOLDER = os.path.join(os.path.dirname(__file__), "signs")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "mp3", "wav", "m4a", "ogg"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIGNS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    """Converts seconds (float) to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ─── Health Check ────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Video to Sign Language API is running"})


# ─── Video Upload & Transcription ────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """
    Upload a video file and transcribe it using Whisper.
    Returns the full transcription and timestamped segments.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # Save the uploaded file with a unique name
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Transcribe using Whisper
        import whisper

        print("Loading Whisper model...")
        model = whisper.load_model("base")

        print(f"Transcribing: {filepath}")
        result = model.transcribe(filepath, task="translate", verbose=False)

        full_text = result["text"].strip()
        segments = []

        for seg in result["segments"]:
            segments.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "start_formatted": format_timestamp(seg["start"]),
                "end_formatted": format_timestamp(seg["end"]),
                "text": seg["text"].strip(),
            })

        # Clean up uploaded file after transcription
        try:
            os.remove(filepath)
        except OSError:
            pass

        return jsonify({
            "success": True,
            "transcription": {
                "full_text": full_text,
                "segments": segments,
                "language": result.get("language", "unknown"),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


# ─── Text to Sign Language ───────────────────────────────────────────────────

@app.route("/api/text-to-sign", methods=["POST"])
def convert_text_to_sign():
    """
    Convert text to sign language tokens.
    Accepts JSON body with 'text' field.
    Returns a list of sign tokens with asset references.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided. Send JSON with 'text' field."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text is empty"}), 400

    try:
        signs = text_to_signs(text)
        return jsonify({
            "success": True,
            "original_text": text,
            "signs": signs,
            "total_signs": len(signs),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500


# ─── Serve Sign Assets ──────────────────────────────────────────────────────

@app.route("/api/signs/<path:filename>", methods=["GET"])
def serve_sign(filename):
    """Serve sign language GIF/image assets."""
    return send_from_directory(SIGNS_FOLDER, filename)


# ─── Run Server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤟 Video to Sign Language API starting...")
    print(f"   Upload folder: {UPLOAD_FOLDER}")
    print(f"   Signs folder:  {SIGNS_FOLDER}")
    app.run(debug=True, host="0.0.0.0", port=5001)
