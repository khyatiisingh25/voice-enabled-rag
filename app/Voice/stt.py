from .sarvam_client import transcribe_audio


def transcribe(audio_path: str) -> str:
    """
    Convert an audio file into text using Sarvam STT.
    """
    return transcribe_audio(audio_path)