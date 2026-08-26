import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL = "https://voice-enabled-rag-production-860a.up.railway.app";

function App() {
  const [language, setLanguage] = useState("English");
  const [recording, setRecording] = useState(false);
  const [audioFile, setAudioFile] = useState(null);

  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [grounded, setGrounded] = useState(null);

  const [textQuery, setTextQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // -----------------------------
  // TEXT QUERY
  // -----------------------------
  const askTextQuestion = async () => {
    if (!textQuery.trim()) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setGrounded(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: textQuery.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();

      setTranscript(textQuery.trim());
      setAnswer(data.answer || "");
      setSources(data.sources || []);
      setGrounded(data.grounded ?? null);
    } catch (err) {
      setError(
        "Unable to connect to the backend. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------
  // START RECORDING
  // -----------------------------
  const startRecording = async () => {
    try {
      setError("");

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone is not supported by this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const mediaRecorder = new MediaRecorder(stream);

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || "audio/webm",
        });

        stream.getTracks().forEach((track) => track.stop());

        const file = new File(
          [audioBlob],
          `voice-query-${Date.now()}.webm`,
          {
            type: audioBlob.type,
          }
        );

        setAudioFile(file);
        await sendVoiceQuery(file);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();

      setRecording(true);
      setTranscript("");
      setAnswer("");
      setSources([]);
      setGrounded(null);
    } catch (err) {
      setError("Microphone access failed. Please allow microphone permission.");
      setRecording(false);
    }
  };

  // -----------------------------
  // STOP RECORDING
  // -----------------------------
  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  // -----------------------------
  // VOICE QUERY
  // -----------------------------
  const sendVoiceQuery = async (file) => {
    if (!file) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setGrounded(null);

    try {
      const formData = new FormData();

      formData.append("audio", file);
      formData.append("language", language.toLowerCase());

      const response = await fetch(`${API_BASE_URL}/voice/query`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Voice request failed: ${response.status}`);
      }

      const data = await response.json();

      setTranscript(data.transcript || data.query || "");
      setAnswer(data.answer || "");
      setSources(data.sources || []);
      setGrounded(data.grounded ?? null);
    } catch (err) {
      setError(
        "Voice query failed. Check that the backend /voice/query endpoint is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------
  // AUDIO UPLOAD
  // -----------------------------
  const handleAudioUpload = async (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setAudioFile(file);
    await sendVoiceQuery(file);
  };

  // -----------------------------
  // RESET
  // -----------------------------
  const resetResults = () => {
    setTranscript("");
    setAnswer("");
    setSources([]);
    setGrounded(null);
    setError("");
    setTextQuery("");
    setAudioFile(null);
  };

  useEffect(() => {
    return () => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="app">
      {/* Background decoration */}
      <div className="glow glow-one"></div>
      <div className="glow glow-two"></div>

      <main className="container">
        {/* =========================
            HEADER
        ========================== */}
        <header className="hero">
          <div className="team-name">ZERO SIGNAL</div>

          <h1>
            VOICE-ENABLED
            <br />
            <span>RAG MODEL</span>
          </h1>

          <p>
            Real-time voice-powered retrieval and grounded question answering.
          </p>
        </header>

        {/* =========================
            CONTROL PANEL
        ========================== */}
        <section className="control-card">
          <div className="controls">
            <div className="language-control">
              <label htmlFor="language">VOICE LANGUAGE</label>

              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={recording || loading}
              >
                <option>English</option>
                <option>Hindi</option>
              </select>
            </div>

            <button
              className={`record-btn ${recording ? "recording" : ""}`}
              onClick={recording ? stopRecording : startRecording}
              disabled={loading}
            >
              <span className="mic-icon">
                {recording ? "■" : "🎙"}
              </span>

              {recording ? "STOP RECORDING" : "START RECORDING"}
            </button>

            <label className="upload-btn">
              <span>↑</span>
              UPLOAD AUDIO

              <input
                type="file"
                accept="audio/*"
                onChange={handleAudioUpload}
                disabled={loading || recording}
              />
            </label>
          </div>

          {recording && (
            <div className="recording-status">
              <span className="pulse-dot"></span>
              Recording in progress...
            </div>
          )}

          {audioFile && !recording && (
            <div className="file-name">
              Selected audio: <strong>{audioFile.name}</strong>
            </div>
          )}

          {/* TEXT QUERY */}
          <div className="text-query">
            <input
              type="text"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  askTextQuestion();
                }
              }}
              placeholder="Or type your question here..."
              disabled={loading || recording}
            />

            <button
              onClick={askTextQuestion}
              disabled={loading || recording || !textQuery.trim()}
            >
              {loading ? "PROCESSING..." : "ASK"}
            </button>
          </div>
        </section>

        {/* =========================
            ERROR
        ========================== */}
        {error && (
          <div className="error-box">
            <strong>ERROR</strong>
            <span>{error}</span>
          </div>
        )}

        {/* =========================
            TRANSCRIPT
        ========================== */}
        <section className="result-card">
          <div className="section-heading">
            <span>01</span>
            <h2>TRANSCRIPT</h2>
          </div>

          <div className="result-content transcript">
            {transcript ? (
              transcript
            ) : (
              <span className="placeholder">
                Your voice transcript will appear here...
              </span>
            )}
          </div>
        </section>

        {/* =========================
            ANSWER
        ========================== */}
        <section className="result-card answer-card">
          <div className="section-heading">
            <span>02</span>
            <h2>ANSWER</h2>

            {grounded !== null && (
              <div
                className={`grounded-badge ${
                  grounded ? "grounded" : "not-grounded"
                }`}
              >
                <span>{grounded ? "●" : "○"}</span>
                {grounded ? "GROUNDED" : "NOT GROUNDED"}
              </div>
            )}
          </div>

          <div className="result-content answer">
            {loading ? (
              <div className="loading">
                <span></span>
                <span></span>
                <span></span>
                Retrieving answer...
              </div>
            ) : answer ? (
              answer
            ) : (
              <span className="placeholder">
                Your grounded answer will appear here...
              </span>
            )}
          </div>
        </section>

        {/* =========================
            SOURCES
        ========================== */}
        <section className="result-card">
          <div className="section-heading">
            <span>03</span>
            <h2>SOURCES</h2>

            {sources.length > 0 && (
              <div className="source-count">
                {sources.length} source{sources.length !== 1 ? "s" : ""}
              </div>
            )}
          </div>

          <div className="sources">
            {sources.length > 0 ? (
              sources.map((source, index) => (
                <div className="source-item" key={index}>
                  <div className="source-number">
                    {String(index + 1).padStart(2, "0")}
                  </div>

                  <div>
                    <div className="source-name">
                      {typeof source === "string"
                        ? source
                        : source.source ||
                          source.name ||
                          source.metadata?.source ||
                          "Retrieved document"}
                    </div>

                    {typeof source !== "string" && source.text && (
                      <div className="source-text">{source.text}</div>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <span className="placeholder">
                Retrieved sources will appear here...
              </span>
            )}
          </div>
        </section>

        {/* =========================
            ARCHITECTURE
        ========================== */}
        <section className="architecture-card">
          <div className="section-heading">
            <span>04</span>
            <h2>ARCHITECTURE VISIBILITY</h2>
          </div>

          <div className="flow">
            <div className="flow-step">
              <strong>VOICE</strong>
              <span>Input</span>
            </div>

            <div className="arrow">→</div>

            <div className="flow-step">
              <strong>STT</strong>
              <span>Transcription</span>
            </div>

            <div className="arrow">→</div>

            <div className="flow-step">
              <strong>RETRIEVAL</strong>
              <span>Vector Search</span>
            </div>

            <div className="arrow">→</div>

            <div className="flow-step">
              <strong>RAG</strong>
              <span>Generation</span>
            </div>

            <div className="arrow">→</div>

            <div className="flow-step">
              <strong>ANSWER</strong>
              <span>Grounded Output</span>
            </div>
          </div>

          <div className="architecture-note">
            Voice → Speech-to-Text → Chunking / Retrieval → Context →
            Answer Generation → Grounding / Guardrails
          </div>
        </section>

        {/* =========================
            FOOTER
        ========================== */}
        <footer>
          <span>ZERO SIGNAL</span>
          <span>VOICE-ENABLED RAG MODEL</span>
          <span>HACKER HOUSE GOA 2026</span>
        </footer>

        <button className="reset-btn" onClick={resetResults}>
          CLEAR SESSION
        </button>
      </main>
    </div>
  );
}

export default App;
