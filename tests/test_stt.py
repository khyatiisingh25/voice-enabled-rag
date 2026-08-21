from app.Voice.stt import transcribe


def test_transcribe_delegates_to_sarvam(monkeypatch):
    expected_text = "What is retrieval augmented generation?"

    def fake_transcribe_audio(audio_path):
        assert audio_path == "samples/Recording.m4a"
        return expected_text

    monkeypatch.setattr(
        "app.Voice.stt.transcribe_audio",
        fake_transcribe_audio,
    )

    result = transcribe("samples/Recording.m4a")

    assert result == expected_text
