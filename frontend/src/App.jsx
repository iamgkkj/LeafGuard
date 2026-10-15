import { useState, useRef, useEffect } from 'react'
import logo from './assets/logo.svg'
import logoPng from './assets/logo.png'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('compare')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showGradCam, setShowGradCam] = useState(false)
  const [gradCamImage, setGradCamImage] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    document.title = 'LeafGuard AI'
    const link = document.querySelector("link[rel*='icon']") || document.createElement('link')
    link.rel = 'icon'
    link.type = 'image/svg+xml'
    link.href = logo
    document.head.appendChild(link)

    const apple = document.querySelector("link[rel='apple-touch-icon']") || document.createElement('link')
    apple.rel = 'apple-touch-icon'
    apple.href = logoPng
    document.head.appendChild(apple)
  }, [])

  const runAnalysis = async (file, selectedModel) => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      if (selectedModel === 'compare') {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch(`${API_URL}/compare`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        setResult({ mode: 'compare', data })
      } else {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('model', selectedModel)

        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        setResult({ mode: 'single', model: selectedModel, data })
      }
    } catch (err) {
      setError(err.message || 'Failed to analyze image')
    } finally {
      setLoading(false)
    }
  }

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
      setGradCamImage(null)
      setShowGradCam(false)
      runAnalysis(file, model)
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
    runAnalysis(image, model)
  }

  const handleGradCam = async () => {
    if (!image) return

    const gradModel = model === 'compare' ? 'cbam' : model

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', image)
    formData.append('model', gradModel)

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

  const renderTopPredictions = (predictions) => (
    <div className="top-predictions">
      <h4>Top predictions</h4>
      <ul>
        {predictions.map((pred, idx) => (
          <li key={idx} className={idx === 0 ? 'top-prediction' : ''}>
            <span className="rank">{idx + 1}.</span>
            <span className="class-name">{pred.class}</span>
            <span className="pred-confidence">{formatConfidence(pred.confidence)}</span>
          </li>
        ))}
      </ul>
    </div>
  )

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <img src={logo} alt="LeafGuard logo" className="header-logo" />
          <div className="header-text">
            <h1>LeafGuard AI</h1>
            <p className="subtitle">Crop leaf disease identification</p>
          </div>
        </div>
        <a
          className="github-link"
          href="https://github.com/iamgkkj/LeafGuard"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="LeafGuard GitHub repository"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true">
            <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.012 24 17.522 24 12.227c0-6.627-5.373-12-12-12-.001 0 0 0 0 0z" fillRule="evenodd" clipRule="evenodd" />
          </svg>
          <span>GitHub</span>
        </a>
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
              onChange={(e) => {
                setModel(e.target.value)
                if (image) runAnalysis(image, e.target.value)
              }}
            >
              <option value="compare">Compare both (CBAM + Baseline)</option>
              <option value="cbam">CBAM only</option>
              <option value="baseline">Baseline only</option>
            </select>
          </div>

          <button
            className="analyze-button"
            onClick={handleAnalyze}
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
            {result.mode === 'compare' ? (
              <div className="comparison-results">
                <div className="model-result">
                  <h3>Baseline MobileNetV3-Small</h3>
                  <div className="prediction">
                    <span className="disease-name">{result.data.baseline.predicted_class}</span>
                    <span className="confidence">{formatConfidence(result.data.baseline.confidence)}</span>
                  </div>
                  {renderTopPredictions(result.data.baseline.top_predictions)}
                </div>
                <div className="model-result">
                  <h3>MobileNetV3-Small + CBAM</h3>
                  <div className="prediction">
                    <span className="disease-name">{result.data.cbam.predicted_class}</span>
                    <span className="confidence">{formatConfidence(result.data.cbam.confidence)}</span>
                  </div>
                  {renderTopPredictions(result.data.cbam.top_predictions)}
                </div>
                <div className="agreement">
                  {result.data.agreement ? 'Both models agree' : 'Models disagree'}
                </div>
              </div>
            ) : (
              <>
                <div className="main-prediction">
                  <h3>Predicted disease</h3>
                  <div className="prediction-large">
                    <span className="disease-name-large">{result.data.predicted_class}</span>
                    <span className="confidence-large">{formatConfidence(result.data.confidence)}</span>
                  </div>
                </div>

                {renderTopPredictions(result.data.top_predictions)}

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