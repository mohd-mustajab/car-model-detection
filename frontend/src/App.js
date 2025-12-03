import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const BACKEND = "http://localhost:5000/predict";

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f);

    if (f) {
      const reader = new FileReader();
      reader.onload = (ev) => setPreviewUrl(ev.target.result);
      reader.readAsDataURL(f);
    }
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please choose an image.");

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(BACKEND, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.statusText}`);

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed!");
    }

    setLoading(false);
  };

  return (
    <div className="app-container fade-in">
      <h1 className="app-title">Car Model Classifier</h1>

      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button className="predict-btn" onClick={handleUpload}>
          Predict
        </button>
      </div>

      <div className="content-wrapper">

        <div className="preview-box fade-in">
          {previewUrl ? (
            <img src={previewUrl} alt="" className="preview-image" />
          ) : (
            <div className="preview-placeholder">Preview Image Here</div>
          )}
        </div>

        <div className="result-box fade-in">
          <h2 className="result-title">Prediction</h2>

          {loading && <div className="loader"></div>}

          {!loading && result?.predictions && (
            <div>
              {result.predictions.map((p, i) => (
                <div className="pred-row fade-in" key={i}>
                  <span className="pred-label">{i + 1}. {p.label}</span>
                  <span className="pred-score">{p.score.toFixed(4)}</span>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>

      <div className="footer">
        Developed by <b>Mohd Mustajab</b>
      </div>
    </div>
  );
}

export default App;
