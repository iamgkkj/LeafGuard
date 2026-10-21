import { useState, useRef, useEffect } from 'react'
import logo from './assets/logo.svg'
import logoPng from './assets/logo.png'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MODEL_LABELS = {
  baseline: 'Baseline MobileNetV3-Small',
  cbam: 'MobileNetV3-Small + CBAM',
}

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('compare')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [gradCam, setGradCam] = useState({ loading: false, data: null, model: null })
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
    setGradCam({ loading: false, data: null, model: null })

    try {
      const formData = new FormData()
      formData.append('file', file)

      let endpoint = '/predict'
      if (selectedModel === 'compare') {
        endpoint = '/compare'
      } else {
        formData.append('model', selectedModel)
      }

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => null)
        throw new Error(detail?.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult({ mode: selectedModel === 'compare' ? 'compare' : 'single', model: selectedModel, data })
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
      setGradCam({ loading: false, data: null, model: null })
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

  const handleGradCam = async (targetModel) => {
    if (!image) return

    setGradCam({ loading: true, data: null, model: targetModel })
    setError(null)

    const formData = new FormData()
    formData.append('file', image)
    formData.append('model', targetModel)

    try {
      const response = await fetch(`${API_URL}/gradcam`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => null)
        throw new Error(detail?.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setGradCam({ loading: false, data, model: targetModel })
    } catch (err) {
      setError(err.message || 'Failed to generate Grad-CAM')
      setGradCam({ loading: false, data: null, model: null })
    }
  }

  const formatConfidence = (val) => (val * 100).toFixed(1) + '%'

  const formatClassName = (name) => name.replace(/___/g, ' — ').replace(/_/g, ' ')

  const renderTopPredictions = (predictions) => (
    <div className="top-predictions">
      <h4>Top predictions</h4>
      <ul>
        {predictions.map((pred, idx) => (
          <li key={idx} className={idx === 0 ? 'top-prediction' : ''}>
            <span className="rank">{idx + 1}</span>
            <span className="class-name">{formatClassName(pred.class)}</span>
            <span className="pred-confidence">{formatConfidence(pred.confidence)}</span>
          </li>
        ))}
      </ul>
    </div>
  )

  const renderConfidenceBar = (confidence) => (
    <div className="confidence-bar" aria-hidden="true">
      <div className="confidence-fill" style={{ width: `${Math.min(confidence * 100, 100)}%` }} />
    </div>
  )

  const renderModelCard = (prediction, modelKey) => (
    <div className="model-result">
      <div className="model-card-head">
        <span className={`model-badge ${modelKey}`}>{modelKey === 'cbam' ? 'CBAM' : 'Baseline'}</span>
        <h3>{MODEL_LABELS[modelKey]}</h3>
      </div>

      <div className="prediction">
        <span className="disease-name">{formatClassName(prediction.predicted_class)}</span>
        <span className="confidence">{formatConfidence(prediction.confidence)}</span>
      </div>
      {renderConfidenceBar(prediction.confidence)}

      {renderTopPredictions(prediction.top_predictions)}

      <button
        className="gradcam-button"
        onClick={() => handleGradCam(modelKey)}
        disabled={gradCam.loading}
      >
        {gradCam.loading && gradCam.model === modelKey ? 'Generating...' : 'Show explanation'}
      </button>
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
                <div className="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 16V4" />
                    <path d="m7 9 5-5 5 5" />
                    <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                  </svg>
                </div>
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
                {renderModelCard(result.data.baseline, 'baseline')}
                {renderModelCard(result.data.cbam, 'cbam')}
                <div className={`agreement ${result.data.agreement ? '' : 'disagree'}`}>
                  {result.data.agreement ? (
                    <>Both models agree on the same disease class</>
                  ) : (
                    <>Models disagree — see top predictions for each</>
                  )}
                </div>
              </div>
            ) : (
              <div className="single-result">
                {renderModelCard(result.data, result.model)}
              </div>
            )}
          </div>
        )}

        {gradCam.loading && (
          <div className="gradcam-section">
            <h3>Explanation</h3>
            <p className="gradcam-loading">Generating Grad-CAM heatmap for {MODEL_LABELS[gradCam.model]}...</p>
          </div>
        )}

        {gradCam.data && (
          <div className="gradcam-section">
            <div className="gradcam-head">
              <h3>Explanation — {MODEL_LABELS[gradCam.data.model]}</h3>
              <button className="close-button" onClick={() => setGradCam({ loading: false, data: null, model: null })}>
                Close
              </button>
            </div>

            <p className="gradcam-caption">
              Prediction: <strong>{formatClassName(gradCam.data.prediction.predicted_class)}</strong>
              {' · '}
              Confidence: <strong>{formatConfidence(gradCam.data.prediction.confidence)}</strong>
            </p>

            <div className="gradcam-grid">
              <figure>
                <img src={preview} alt="Original leaf" className="gradcam-image" />
                <figcaption>Original</figcaption>
              </figure>
              <figure>
                <img src={gradCam.data.heatmap_image} alt="Grad-CAM heatmap" className="gradcam-image" />
                <figcaption>Grad-CAM (red = high attention)</figcaption>
              </figure>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
