import { useState, useRef } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

function ConfidenceRing({ value }) {
  const pct = Math.round(value*1000)/10
  const r = 44, c = 2*Math.PI*r, off = c - (pct/100)*c
  let color = pct >= 85 ? '#16a34a' : pct >= 65 ? '#eab308' : '#ef4444'
  return (
    <div className="relative w-28 h-28 flex items-center justify-center">
      <svg width="112" height="112" className="-rotate-90">
        <circle cx="56" cy="56" r={r} fill="none" stroke="#dcfce7" strokeWidth="10" />
        <circle cx="56" cy="56" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={off} style={{transition:'stroke-dashoffset 0.8s ease'}} />
      </svg>
      <div className="absolute text-center">
        <div className="text-xl font-extrabold text-leaf-900 leading-none">{pct.toFixed(1)}%</div>
        <div className="text-[10px] font-semibold tracking-widest text-leaf-600 uppercase">Confidence</div>
      </div>
    </div>
  )
}

export default function PredictView({ api }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [gradcam, setGradcam] = useState(null)
  const [opacity, setOpacity] = useState(0.5)
  const [slider, setSlider] = useState(50)
  const [method, setMethod] = useState('gradcam')
  const [modelType, setModelType] = useState('cbam')
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

  const onFile = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) { setError('Please upload a JPEG/PNG image'); return }
    setError(null); setResult(null); setGradcam(null)
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const pred = await axios.post(`${api}/predict?model_type=${modelType}`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(pred.data)
      const fd2 = new FormData(); fd2.append('file', file)
      const gc = await axios.post(`${api}/gradcam?method=${method}&alpha=${opacity}&model_type=${modelType}`, fd2)
      setGradcam(gc.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Analysis failed')
    } finally { setLoading(false) }
  }

  const handleDrop = (e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; onFile(f) }
  const barData = result ? result.top3.map((t,i) => ({ name: t.label.split('___').pop().replaceAll('_',' ').slice(0,22), conf: Math.round(t.confidence*1000)/10, idx:i })) : []
  const colors = ['#16a34a','#22c55e','#86efac']

  return (
    <div className="space-y-6">
      {/* Step indicator */}
      <div className="flex items-center gap-2 text-xs font-semibold">
        <span className={`px-3 py-1.5 rounded-full border ${!file?'bg-leaf-600 text-white border-leaf-600':'bg-white text-leaf-700 border-leaf-200'}`}>① Upload</span>
        <span className="w-6 h-px bg-leaf-200" />
        <span className={`px-3 py-1.5 rounded-full border ${file && !result ? 'bg-leaf-600 text-white border-leaf-600' : 'bg-white text-gray-500 border-gray-200'}`}>② Analyze</span>
        <span className="w-6 h-px bg-leaf-200" />
        <span className={`px-3 py-1.5 rounded-full border ${result?'bg-leaf-600 text-white border-leaf-600':'bg-white text-gray-500 border-gray-200'}`}>③ Explain</span>
        <span className="ml-auto hidden sm:inline-flex items-center gap-2 bg-white border border-leaf-200 rounded-full px-3 py-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> Live CBAM + Grad-CAM
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1.05fr_0.95fr] gap-6">
        {/* Left: Upload */}
        <div className="glass rounded-[24px] shadow-soft-lg p-5 sm:p-6">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div>
              <h2 className="text-[18px] font-bold text-leaf-900">Upload Leaf Image</h2>
              <p className="text-sm text-sage-700">Drag-and-drop or browse. JPG/PNG, 224×224 optimal.</p>
            </div>
            <span className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium bg-leaf-50 border border-leaf-200 text-leaf-700 px-2.5 py-1 rounded-full">📸 Max 10MB</span>
          </div>

          <div
            onDragOver={e=>{e.preventDefault(); setDragOver(true)}}
            onDragLeave={()=>setDragOver(false)}
            onDrop={handleDrop}
            onClick={()=>!preview && inputRef.current?.click()}
            className={`group relative rounded-[20px] border-2 border-dashed p-2 transition overflow-hidden ${dragOver?'border-leaf-500 bg-leaf-50':'border-leaf-200 bg-white/60 hover:bg-white hover:border-leaf-300 hover:shadow-soft'} ${preview?'cursor-default p-2':'cursor-pointer'}`}
          >
            <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={e=>onFile(e.target.files[0])} />
            {!preview ? (
              <div className="py-10 text-center">
                <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-br from-leaf-500 to-leaf-600 flex items-center justify-center text-white text-xl shadow-md group-hover:scale-105 transition">⬆</div>
                <p className="mt-3 text-sm font-semibold text-leaf-900">Drop leaf image here or click to browse</p>
                <p className="text-xs text-sage-500 mt-1">Tomato, Apple, Grape, Corn — healthy or diseased</p>
                <div className="mt-3 inline-flex gap-1.5 text-[11px]">
                  <span className="bg-leaf-100 text-leaf-700 px-2 py-1 rounded-full">🌿 Auto-crop 224</span>
                  <span className="bg-earth-100 text-earth-700 px-2 py-1 rounded-full">⚡ Instant</span>
                </div>
              </div>
            ) : (
              <div className="relative">
                <img src={preview} alt="preview" className="w-full max-h-[360px] object-contain rounded-[16px] bg-leaf-50" />
                <div className="absolute top-3 left-3 bg-white/90 backdrop-blur px-2.5 py-1 rounded-full text-xs font-medium shadow">📄 {file?.name?.slice(0,22)}</div>
                <button onClick={()=>{setPreview(null); setFile(null); setResult(null); setGradcam(null)}} className="absolute top-3 right-3 w-8 h-8 rounded-full bg-white shadow flex items-center justify-center hover:bg-gray-50">✕</button>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 mt-4">
            <label className="text-xs font-semibold text-leaf-800">Model
              <select value={modelType} onChange={e=>setModelType(e.target.value)} className="mt-1 w-full border border-leaf-200 rounded-xl px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-leaf-200">
                <option value="cbam">🌟 CBAM-MobileNetV3</option>
                <option value="baseline">Baseline MobileNetV3</option>
              </select>
            </label>
            <label className="text-xs font-semibold text-leaf-800">Explainability
              <select value={method} onChange={e=>setMethod(e.target.value)} className="mt-1 w-full border border-leaf-200 rounded-xl px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-leaf-200">
                <option value="gradcam">Grad-CAM</option>
                <option value="gradcam++">Grad-CAM++</option>
              </select>
            </label>
          </div>

          <button
            onClick={analyze}
            disabled={!file || loading}
            className="w-full mt-4 bg-gradient-to-r from-leaf-600 to-leaf-500 hover:from-leaf-700 hover:to-leaf-600 disabled:from-gray-300 disabled:to-gray-300 text-white font-semibold py-3 rounded-xl shadow-soft flex items-center justify-center gap-2 transition"
          >
            {loading ? (<><span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" /> Analyzing…</>) : '🔍 Analyze Leaf'}
          </button>
          {error && <div className="mt-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl px-3 py-2.5 flex gap-2"><span>⚠️</span><span>{error}</span></div>}
          {!result && !loading && (
            <div className="mt-4 rounded-xl bg-gradient-to-br from-leaf-50 to-white border border-leaf-100 p-3 text-xs leading-relaxed text-sage-700">
              <strong className="text-leaf-800">Tip:</strong> Try a Tomato leaf — Grad-CAM will highlight lesions in red/yellow while ignoring soil background. No image? API still demos with realistic predictions.
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="glass rounded-[24px] shadow-soft-lg p-5 sm:p-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-[18px] font-bold text-leaf-900">Diagnosis</h2>
            {result && <span className="text-xs font-semibold bg-leaf-100 text-leaf-700 px-2.5 py-1 rounded-full">#{result.class_idx} • {method}</span>}
          </div>
          {!result ? (
            <div className="h-[420px] flex flex-col items-center justify-center text-center">
              <div className="w-20 h-20 rounded-3xl bg-leaf-50 border border-leaf-100 flex items-center justify-center text-3xl mb-3">🍃</div>
              <p className="text-sm font-semibold text-leaf-900">Awaiting analysis</p>
              <p className="text-xs text-sage-500 mt-1 max-w-[260px]">Upload and tap Analyze to see disease, confidence, and lesion heatmap.</p>
              <div className="mt-4 grid grid-cols-3 gap-2 text-[11px]">
                <span className="bg-white border border-leaf-100 rounded-xl px-3 py-2">Top-3<br/><strong>Ranked</strong></span>
                <span className="bg-white border border-leaf-100 rounded-xl px-3 py-2">Grad-CAM<br/><strong>Overlay</strong></span>
                <span className="bg-white border border-leaf-100 rounded-xl px-3 py-2">Slider<br/><strong>Before/After</strong></span>
              </div>
            </div>
          ) : (
            <div className="space-y-5 animate-slide-up">
              <div className="flex gap-4 items-center bg-white rounded-2xl border border-leaf-100 p-4 shadow-sm">
                <ConfidenceRing value={result.confidence} />
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-semibold tracking-widest text-leaf-600 uppercase">Predicted Disease</p>
                  <p className="text-[16px] font-bold text-leaf-900 leading-tight truncate">{result.predicted_class.replaceAll('___',' — ').replaceAll('_',' ')}</p>
                  <p className="text-xs text-sage-600">Class #{result.class_idx} • Model: {modelType.toUpperCase()} • {gradcam?.method || method}</p>
                  <div className="mt-2 flex gap-1.5 flex-wrap">
                    {result.top3.slice(0,3).map((t,i)=><span key={i} className={`text-[11px] px-2 py-1 rounded-full font-medium ${i===0?'bg-leaf-600 text-white':'bg-leaf-50 text-leaf-700 border border-leaf-200'}`}>{t.label.split('___').pop().replaceAll('_',' ')} {Math.round(t.confidence*100)}%</span>)}
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl border border-leaf-100 p-3">
                <p className="text-xs font-semibold text-leaf-800 mb-2">Top-3 Confidence</p>
                <div className="h-[120px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={barData} layout="vertical" margin={{left:8,right:16,top:4,bottom:4}}>
                      <XAxis type="number" domain={[0,100]} hide />
                      <YAxis type="category" dataKey="name" width={150} tick={{fontSize:11, fill:'#166534', fontWeight:600}} axisLine={false} tickLine={false} />
                      <Tooltip formatter={(v)=>[`${v}%`,'Confidence']} contentStyle={{borderRadius:12, border:'1px solid #dcfce7'}} />
                      <Bar dataKey="conf" radius={[0,12,12,0]} barSize={18}>
                        {barData.map((e,i)=><Cell key={i} fill={colors[i] || '#16a34a'} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-2xl border border-leaf-100 p-3">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <p className="text-xs font-semibold text-leaf-800">Lesion Heatmap — Drag slider to compare</p>
                  <label className="flex items-center gap-2 text-xs font-medium">Opacity <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={e=>setOpacity(parseFloat(e.target.value))} className="w-20" /></label>
                </div>

                {/* Before/After slider */}
                <div className="relative rounded-xl overflow-hidden border border-leaf-200 aspect-square bg-leaf-50 select-none">
                  {gradcam ? (
                    <>
                      <img src={preview} alt="original" className="absolute inset-0 w-full h-full object-cover" />
                      <div className="absolute inset-0 w-full h-full overflow-hidden" style={{clipPath:`inset(0 ${100-slider}% 0 0)`}}>
                        <img src={`data:image/png;base64,${gradcam.overlay_base64}`} alt="heatmap" className="w-full h-full object-cover" style={{opacity: 0.92}} />
                      </div>
                      <div className="absolute top-0 bottom-0 w-0.5 bg-white shadow" style={{left:`${slider}%`}} />
                      <div className="absolute top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white shadow-lg border border-leaf-200 flex items-center justify-center text-leaf-700 text-xs font-bold" style={{left:`calc(${slider}% - 16px)`}}>↔</div>
                      <div className="absolute top-2 left-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded-full">Heatmap</div>
                      <div className="absolute top-2 right-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded-full">Original</div>
                      <input type="range" min="0" max="100" value={slider} onChange={e=>setSlider(parseInt(e.target.value))} className="absolute bottom-3 left-1/2 -translate-x-1/2 w-[60%]" />
                    </>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-sage-500">{loading?'Generating heatmap…':'Heatmap appears after analysis'}</div>
                  )}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <div className="rounded-xl overflow-hidden border border-leaf-200">
                    <img src={preview} alt="original small" className="w-full aspect-square object-cover" />
                    <p className="text-[11px] text-center py-1 bg-leaf-50 font-medium">Original</p>
                  </div>
                  <div className="rounded-xl overflow-hidden border border-leaf-200 bg-leaf-50">
                    {gradcam ? <img src={`data:image/png;base64,${gradcam.overlay_base64}`} alt="overlay" className="w-full aspect-square object-cover" /> : <div className="aspect-square flex items-center justify-center text-xs text-sage-500">—</div>}
                    <p className="text-[11px] text-center py-1 bg-leaf-50 font-medium">Overlay ({gradcam?.method || method})</p>
                  </div>
                </div>
                <p className="text-[11px] text-sage-600 mt-2">🔴 Red/Yellow = attended lesion region. CBAM focuses attention on diseased tissue, ignoring soil/background clutter.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
