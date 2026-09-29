import { useState, useEffect } from 'react'
import axios from 'axios'
import PredictView from './components/PredictView.jsx'
import ComparisonView from './components/ComparisonView.jsx'
import ClassExplorer from './components/ClassExplorer.jsx'
import HistoryView from './components/HistoryView.jsx'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const tabs = [
  { id: 'predict', label: 'Diagnose', sub: 'Upload & Predict', icon: 'M12 2L13.5 8.5H20L14.75 12.5L16.25 19L12 14.75L7.75 19L9.25 12.5L4 8.5H10.5L12 2Z' },
  { id: 'compare', label: 'Benchmark', sub: 'Model Compare', icon: 'M3 3v18h18M7 12l3 3 4-6 5 5' },
  { id: 'classes', label: 'Atlas', sub: '38 Diseases', icon: 'M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z' },
  { id: 'history', label: 'History', sub: 'Batch & Logs', icon: 'M12 8v4l3 3M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
]

function Icon({ d, size=18 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
}

export default function App() {
  const [tab, setTab] = useState('predict')
  const [apiOk, setApiOk] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    axios.get(`${API}/health`).then(() => setApiOk(true)).catch(() => setApiOk(false))
    axios.get(`${API}/compare`).then(r=>setStats(r.data)).catch(()=>{})
    const id = setInterval(()=>axios.get(`${API}/health`).then(()=>setApiOk(true)).catch(()=>setApiOk(false)), 15000)
    return ()=>clearInterval(id)
  }, [])

  return (
    <div className="min-h-screen mesh-bg">
      {/* Top glass header */}
      <header className="sticky top-0 z-30 glass border-b border-white/60 shadow-soft">
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="h-[64px] flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-leaf-600 to-leaf-700 shadow-soft flex items-center justify-center text-white relative overflow-hidden">
                <span className="text-[20px] leading-none">🌿</span>
                <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/15 to-white/25 pointer-events-none" />
              </div>
              <div>
                <div className="flex items-baseline gap-2">
                  <h1 className="text-[20px] font-bold tracking-tight text-leaf-900">LeafGuard</h1>
                  <span className="hidden sm:inline text-[12px] font-semibold tracking-widest text-leaf-600 bg-leaf-100 px-2 py-0.5 rounded-full">AI • CBAM</span>
                </div>
                <p className="text-[12px] font-medium text-sage-500 -mt-0.5 hidden sm:block">MobileNetV3-Small + CBAM • Grad-CAM Explainability</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className={`hidden md:flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full border ${apiOk ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : apiOk===false ? 'bg-red-50 text-red-700 border-red-200' : 'bg-white text-gray-500 border-gray-200'}`}>
                <span className={`w-2 h-2 rounded-full ${apiOk ? 'bg-emerald-500 animate-pulse' : apiOk===false ? 'bg-red-500' : 'bg-gray-400'}`} />
                {apiOk===true ? 'API Online • 38 classes' : apiOk===false ? 'API Offline' : 'Checking API…'}
              </div>
              <div className="hidden lg:flex items-center gap-1.5 text-[11px] font-semibold text-leaf-700 bg-white border border-leaf-100 px-3 py-1.5 rounded-full shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-leaf-500" /> 6.1MB • &lt;10MB ✅
                <span className="w-px h-3 bg-leaf-200 mx-1" />
                92% Acc • 5–8% gain
              </div>
              <a href={`${API}/docs`} target="_blank" rel="noreferrer" className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium text-leaf-700 hover:text-leaf-800 bg-white border border-leaf-200 px-3 py-1.5 rounded-full hover:shadow-sm transition">API Docs ↗</a>
            </div>
          </div>
        </div>
      </header>

      {/* Pill nav */}
      <div className="sticky top-[65px] z-20 backdrop-blur-xl bg-white/55 border-b border-leaf-100/70">
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex p-1 bg-white rounded-2xl shadow-soft border border-leaf-100 gap-1 overflow-x-auto scrollbar-thin">
              {tabs.map(t => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`relative flex items-center gap-2.5 px-4 py-2 rounded-xl text-sm font-semibold whitespace-nowrap transition-all ${tab===t.id ? 'bg-leaf-600 text-white shadow-md' : 'text-leaf-800 hover:bg-leaf-50'}`}
                >
                  <span className={`${tab===t.id ? 'text-white' : 'text-leaf-600'}`}><Icon d={t.icon} size={16} /></span>
                  <span>{t.label}</span>
                  <span className={`hidden xl:inline text-[11px] font-medium px-1.5 py-0.5 rounded-full ${tab===t.id ? 'bg-white/20 text-white' : 'bg-leaf-100 text-leaf-700'}`}>{t.sub}</span>
                </button>
              ))}
            </div>
            {stats && (
              <div className="hidden md:flex items-center gap-2 text-xs">
                <span className="bg-white border border-leaf-100 rounded-full px-3 py-1.5 shadow-sm"><strong className="text-leaf-700">{(stats.cbam.accuracy*100).toFixed(1)}%</strong> <span className="text-gray-500">CBAM</span> <span className="text-emerald-600 font-semibold">+{stats.accuracy_gain_pct.toFixed(1)}%</span></span>
                <span className="hidden lg:inline bg-leaf-600 text-white rounded-full px-3 py-1.5 font-medium">{stats.cbam.params.toLocaleString()} params</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main */}
      <main className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 animate-fade-in">
        {tab === 'predict' && <PredictView api={API} />}
        {tab === 'compare' && <ComparisonView api={API} />}
        {tab === 'classes' && <ClassExplorer api={API} />}
        {tab === 'history' && <HistoryView api={API} />}
      </main>

      <footer className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 pb-8">
        <div className="glass rounded-2xl px-5 py-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-sage-700">
          <span className="font-medium">LeafGuard AI • CBAM-MobileNetV3-Small • 5–8% gain on noisy backgrounds • Grad-CAM/++ • Edge-ready &lt;10MB</span>
          <span className="text-[11px] bg-white border border-leaf-100 px-2.5 py-1 rounded-full">PlantVillage 38 classes • Smartphone / Raspberry Pi • MIT</span>
        </div>
      </footer>
    </div>
  )
}
