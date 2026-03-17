"""
transcriber.py
---------------
Handles audio transcription using OpenAI Whisper (runs locally, free).
Falls back to text input if audio not available.
"""

import os
import tempfile
from typing import Optional


def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_path:  Path to audio file (.wav, .mp3, .m4a, .ogg, .flac)
        model_size:  Whisper model ('tiny', 'base', 'small', 'medium', 'large')
                     'base' is fastest and good enough for most evaluations
    
    Returns:
        dict with keys: text, language, duration, confidence, segments
    """
    try:
        import whisper
        import numpy as np
    except ImportError:
        return {
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "confidence": 0.0,
            "segments": [],
            "error": "Whisper not installed. Run: pip install openai-whisper"
        }

    if not os.path.exists(audio_path):
        return {"text": "", "error": f"File not found: {audio_path}"}

    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, verbose=False)

        # Estimate confidence from segment-level no_speech_prob
        segments = result.get("segments", [])
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            confidence = round(1.0 - avg_no_speech, 3)
            duration = segments[-1].get("end", 0.0)
        else:
            confidence = 0.85
            duration = 0.0

        return {
            "text":       result["text"].strip(),
            "language":   result.get("language", "en"),
            "duration":   duration,
            "confidence": confidence,
            "segments":   segments,
            "error":      None
        }

    except Exception as e:
        return {"text": "", "error": str(e)}


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav", model_size: str = "base") -> dict:
    """
    Transcribe audio from bytes (used in Streamlit file upload).
    Saves to a temp file then transcribes.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path, model_size)
    finally:
        os.unlink(tmp_path)

    return result


def get_duration(audio_bytes: bytes, suffix: str = ".wav") -> float:
    """Get duration of audio in seconds."""
    try:
        import wave, contextlib
        if suffix == ".wav":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            with contextlib.closing(wave.open(tmp_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            os.unlink(tmp_path)
            return duration
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Transcribing: {path}")
        result = transcribe_audio(path)
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(f"Language:   {result['language']}")
            print(f"Duration:   {result['duration']:.1f}s")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Transcript: {result['text']}")
    else:
        print("Usage: python transcriber.py path/to/audio.wav")
