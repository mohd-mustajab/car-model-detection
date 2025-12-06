import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictions, setPredictions] = useState(null); // array of {label, confidence}
  const [loading, setLoading] = useState(false);

  // backend url (use env var or localhost)
  const BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000/predict";

  const handleFileChange = (e) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setPredictions(null);

    if (f) {
      const reader = new FileReader();
      reader.onload = (ev) => setPreviewUrl(ev.target.result);
      reader.readAsDataURL(f);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please choose an image first.");

    setLoading(true);
    setPredictions(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(BACKEND, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => null);
        throw new Error(`Server error ${res.status}: ${txt ?? res.statusText}`);
      }

      const data = await res.json();

      if (data.error) {
        throw new Error(data.error + (data.detail ? `: ${data.detail}` : ""));
      }

      // normalize — prefer data.predictions (array)
      const preds = data.predictions ?? data.predictions_top3 ?? null;
      if (!Array.isArray(preds) || preds.length === 0) {
        throw new Error("No predictions returned from server.");
      }

      // keep only top 3 (backend should already send top3)
      setPredictions(preds.slice(0, 3));
    } catch (err) {
      console.error("Prediction failed:", err);
      alert("Prediction failed: " + (err.message || err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Car Model Classifier</h1>

      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button className="predict-btn" onClick={handleUpload} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      <div className="content-wrapper">
        <div className="preview-box">
          {previewUrl ? (
            <img src={previewUrl} alt="preview" className="preview-image" />
          ) : (
            <div className="preview-placeholder">Preview Image Here</div>
          )}
        </div>

        <div className="result-box">
          <h2 className="result-title">Prediction</h2>

          {loading && <div className="loader"></div>}

          {!loading && predictions && (
            <div className="pred-list">
              {predictions.map((p, i) => (
                <div className="pred-row" key={i}>
                  <div className="pred-main">
                    <span className="pred-label">{i === 0 ? "Prediction 1: " : `Prediction ${i+1}: `}</span>
                    <span className="pred-name">{p.label ?? `class_${i}`}</span>
                  </div>
                  <div className="pred-confidence">
                    {(typeof p.confidence === "number"
                      ? (p.confidence * 100).toFixed(2)
                      : Number(p.confidence).toFixed(2)) + "%"}
                  </div>
                </div>
              ))}
            </div>
          )}

          {!loading && !predictions && (
            <div className="no-result">No prediction yet — upload an image and click Predict.</div>
          )}
        </div>
      </div>

      <div className="footer">Developed by <b>Mohd Mustajab</b></div>
    </div>
  );
}

export default App;
