import os
from dotenv import load_dotenv

from sarvamai import SarvamAI

load_dotenv()


def transcribe_audio(audio_path: str) -> str:
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set")

    client = SarvamAI(
        api_subscription_key=api_key
    )

    with open(audio_path, "rb") as audio_file:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe",
        )

    return response.transcript