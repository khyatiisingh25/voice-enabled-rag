from app.Voice.stt import transcribe


audio_path = "samples/Recording.m4a"

text = transcribe(audio_path)

print("TRANSCRIPT:")
print(text)