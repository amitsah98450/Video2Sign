import whisper  # OpenAI's Whisper model for speech-to-text
import sys        # To read command-line arguments


def format_timestamp(seconds):
    """Converts seconds (float) to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_video(video_path):
    """
    Takes a video file path, extracts audio, translates to English,
    and returns timestamped transcription segments.
    
    - task="translate" tells Whisper to translate any language into English
    - result["segments"] gives us timestamped chunks of text
    """
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    print(f"Transcribing: {video_path}")
    result = model.transcribe(video_path, task="translate", verbose=False)
    
    full_text = result["text"]
    segments = result["segments"]
    
    return full_text, segments


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_video>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    full_text, segments = transcribe_video(video_path)
    
    print("\n--- Full Transcription ---")
    print(full_text)
    
    print("\n--- Timestamped Segments ---")
    for seg in segments:
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        print(f"[{start} --> {end}] {seg['text']}")