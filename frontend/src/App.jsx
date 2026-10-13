import { useState, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('baseline')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showGradCam, setShowGradCam] = useState(false)
  const [gradCamImage, setGradCamImage] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
      setGradCamImage(null)
      setShowGradCam(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }

  const handleInputChange = (e) => {
    const file = e.target.files[0]
    handleFileSelect(file)
  }

  const handleAnalyze = async () => {
    if (!image) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', image)
    formData.append('model', model)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to analyze image')
    } finally {
      setLoading(false)
    }
  }

  const handleCompare = async () => {
    if (!image) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', image)

    try {
      const response = await fetch(`${API_URL}/compare`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to compare models')
    } finally {
      setLoading(false)
    }
  }

  const handleGradCam = async () => {
    if (!image) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', image)
    formData.append('model', model)

    try {
      const response = await fetch(`${API_URL}/gradcam`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setGradCamImage(data.heatmap_image)
      setShowGradCam(true)
    } catch (err) {
      setError(err.message || 'Failed to generate Grad-CAM')
    } finally {
      setLoading(false)
    }
  }

  const formatConfidence = (val) => {
    return (val * 100).toFixed(1) + '%'
  }

  return (
    <div className="app">
      <header className="header">
        <h1>LeafGuard AI</h1>
        <p className="subtitle">Crop leaf disease identification</p>
      </header>

      <main className="main">
        <div className="upload-section">
          <div
            className={`upload-area ${image ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <p>Click or drag image here</p>
                <span className="upload-hint">Supports JPG, PNG</span>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleInputChange}
              style={{ display: 'none' }}
            />
          </div>
        </div>

        <div className="controls-section">
          <div className="model-selector">
            <label htmlFor="model-select">Model</label>
            <select
              id="model-select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="baseline">Baseline</option>
              <option value="cbam">CBAM</option>
              <option value="compare">Compare both</option>
            </select>
          </div>

          <button
            className="analyze-button"
            onClick={model === 'compare' ? handleCompare : handleAnalyze}
            disabled={!image || loading}
          >
            {loading ? 'Analyzing...' : model === 'compare' ? 'Compare models' : 'Analyze leaf'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && (
          <div className="results-section">
            {model === 'compare' ? (
              <div className="comparison-results">
                <div className="model-result">
                  <h3>Baseline</h3>
                  <div className="prediction">
                    <span className="disease-name">{result.baseline.predicted_class}</span>
                    <span className="confidence">{formatConfidence(result.baseline.confidence)}</span>
                  </div>
                </div>
                <div className="model-result">
                  <h3>CBAM</h3>
                  <div className="prediction">
                    <span className="disease-name">{result.cbam.predicted_class}</span>
                    <span className="confidence">{formatConfidence(result.cbam.confidence)}</span>
                  </div>
                </div>
                <div className="agreement">
                  {result.agreement ? 'Both models agree' : 'Models disagree'}
                </div>
              </div>
            ) : (
              <>
                <div className="main-prediction">
                  <h3>Predicted disease</h3>
                  <div className="prediction-large">
                    <span className="disease-name-large">{result.predicted_class}</span>
                    <span className="confidence-large">{formatConfidence(result.confidence)}</span>
                  </div>
                </div>

                <div className="top-predictions">
                  <h4>Top predictions</h4>
                  <ul>
                    {result.top_predictions.map((pred, idx) => (
                      <li key={idx} className={idx === 0 ? 'top-prediction' : ''}>
                        <span className="rank">{idx + 1}.</span>
                        <span className="class-name">{pred.class}</span>
                        <span className="pred-confidence">{formatConfidence(pred.confidence)}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <button
                  className="gradcam-button"
                  onClick={handleGradCam}
                  disabled={loading}
                >
                  Show explanation
                </button>
              </>
            )}
          </div>
        )}

        {showGradCam && gradCamImage && (
          <div className="gradcam-section">
            <h3>Explanation</h3>
            <img src={gradCamImage} alt="Grad-CAM heatmap" className="gradcam-image" />
            <button
              className="close-button"
              onClick={() => setShowGradCam(false)}
            >
              Close
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
