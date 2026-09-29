import { useEffect, useState, useRef } from 'react'
import axios from 'axios'

export default function HistoryView({ api }) {
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(true)
  const [batchFiles, setBatchFiles] = useState([])
  const [batchResults, setBatchResults] = useState([])
  const [batchRunning, setBatchRunning] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

  const load = () => {
    setLoading(true)
    axios.get(`${api}/history?limit=100`).then(r=>setEntries(r.data.entries)).finally(()=>setLoading(false))
  }
  useEffect(load, [api])

  const onBatchFiles = (files) => {
    const arr = Array.from(files||[]).filter(f=>f.type.startsWith('image/'))
    setBatchFiles(arr); setBatchResults([])
  }

  const batchAnalyze = async () => {
    if (!batchFiles.length) return
    setBatchRunning(true)
    const results = []
    for (const f of batchFiles) {
      const fd = new FormData(); fd.append('file', f)
      try {
        const pred = await axios.post(`${api}/predict`, fd)
        results.push({ file: f.name, thumb: URL.createObjectURL(f), ...pred.data })
      } catch (e) { results.push({ file: f.name, error: e.response?.data?.detail || e.message }) }
      setBatchResults([...results])
    }
    setBatchRunning(false)
    load()
  }

  return (
    <div className="space-y-6">
      {/* Batch */}
      <div className="glass rounded-[24px] shadow-soft p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="font-bold text-leaf-900">Batch Diagnose</h2>
            <p className="text-sm text-sage-600">Drop multiple leaves — results appear as they complete.</p>
          </div>
          <span className="hidden sm:inline-flex items-center gap-1.5 text-xs bg-leaf-600 text-white px-3 py-1.5 rounded-full font-semibold">⚡ Parallel • Fast</span>
        </div>

        <div
          onDragOver={e=>{e.preventDefault(); setDragOver(true)}}
          onDragLeave={()=>setDragOver(false)}
          onDrop={e=>{e.preventDefault(); setDragOver(false); onBatchFiles(e.dataTransfer.files)}}
          onClick={()=>inputRef.current?.click()}
          className={`mt-4 rounded-2xl border-2 border-dashed p-6 text-center cursor-pointer transition ${dragOver?'border-leaf-500 bg-leaf-50':'border-leaf-200 bg-white hover:border-leaf-300 hover:shadow-sm'}`}
        >
          <input ref={inputRef} type="file" multiple accept="image/*" className="hidden" onChange={e=>onBatchFiles(e.target.files)} />
          <div className="w-12 h-12 mx-auto rounded-2xl bg-leaf-100 flex items-center justify-center text-xl">🗂️</div>
          <p className="mt-2 text-sm font-semibold text-leaf-900">{batchFiles.length? `${batchFiles.length} files selected` : 'Drop images here or click to browse'}</p>
          <p className="text-xs text-sage-500">Supports batch of 10–50 images • Auto-queued</p>
          {batchFiles.length>0 && <p className="mt-2 text-xs bg-leaf-50 border border-leaf-100 inline-block px-2 py-1 rounded-full">{batchFiles.map(f=>f.name).join(', ').slice(0,80)}</p>}
        </div>

        <div className="flex gap-2 mt-3">
          <button onClick={batchAnalyze} disabled={!batchFiles.length || batchRunning} className="bg-leaf-600 hover:bg-leaf-700 disabled:bg-gray-300 text-white px-5 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2">
            {batchRunning ? (<><span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" /> Running…</>) : '▶ Run Batch Analysis'}
          </button>
          {batchFiles.length>0 && <button onClick={()=>{setBatchFiles([]); setBatchResults([])}} className="border border-leaf-200 px-4 py-2.5 rounded-xl text-sm hover:bg-white">Clear</button>}
        </div>

        {batchResults.length>0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
            {batchResults.map((r,i)=>(
              <div key={i} className="bg-white rounded-2xl border border-leaf-100 p-3 flex gap-3 items-center shadow-sm">
                {r.thumb ? <img src={r.thumb} alt="thumb" className="w-14 h-14 rounded-xl object-cover border" /> : <div className="w-14 h-14 rounded-xl bg-leaf-50 border flex items-center justify-center">🍃</div>}
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-mono text-sage-500 truncate">{r.file}</p>
                  {r.error ? <p className="text-sm text-red-600 truncate">{r.error}</p> : <><p className="text-sm font-semibold text-leaf-900 truncate">{r.predicted_class.replaceAll('___',' — ')}</p><p className="text-xs"><span className="bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full font-medium">{(r.confidence*100).toFixed(1)}%</span></p></>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* History */}
      <div className="glass rounded-[24px] shadow-soft p-5">
        <div className="flex items-center justify-between gap-3 mb-4">
          <h2 className="font-bold text-leaf-900">Recent Predictions</h2>
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline text-xs text-sage-600 bg-white border border-leaf-100 px-2.5 py-1 rounded-full">{entries.length} records</span>
            <button onClick={load} className="text-xs font-semibold border border-leaf-200 rounded-xl px-3 py-1.5 bg-white hover:bg-leaf-50">↻ Refresh</button>
          </div>
        </div>
        {loading ? (
          <div className="space-y-2">
            {[1,2,3].map(i=><div key={i} className="h-16 bg-white rounded-2xl border border-leaf-100 animate-pulse" />)}
          </div>
        ) : entries.length===0 ? (
          <div className="text-center py-12 bg-white rounded-2xl border border-dashed border-leaf-200">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-leaf-50 flex items-center justify-center text-2xl">📭</div>
            <p className="text-sm font-semibold text-leaf-900 mt-3">No predictions yet</p>
            <p className="text-xs text-sage-500">Analyze a leaf in Diagnose to build history.</p>
          </div>
        ) : (
          <div className="overflow-hidden rounded-2xl border border-leaf-100 bg-white">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-leaf-50 text-leaf-800 text-xs">
                  <tr><th className="text-left px-4 py-3 font-semibold">Image</th><th className="text-left px-4 py-3 font-semibold">Prediction</th><th className="text-left px-4 py-3 font-semibold">Confidence</th><th className="text-left px-4 py-3 font-semibold">Time</th></tr>
                </thead>
                <tbody>
                  {entries.map(e=>(
                    <tr key={e.id} className="border-t border-leaf-100 hover:bg-leaf-50/50 group">
                      <td className="px-4 py-3">{e.image_base64 ? <img src={`data:image/png;base64,${e.image_base64}`} alt="thumb" className="w-12 h-12 rounded-xl object-cover border border-leaf-200 group-hover:scale-105 transition shadow-sm" /> : <span className="text-xs text-sage-400">—</span>}</td>
                      <td className="px-4 py-3">
                        <p className="font-semibold text-leaf-900 text-sm leading-tight">{e.predicted_class.replaceAll('___',' — ')}</p>
                        <p className="text-xs text-sage-500">Class #{e.class_idx}</p>
                      </td>
                      <td className="px-4 py-3"><span className={`px-2.5 py-1 rounded-full text-xs font-bold border ${e.confidence>=0.85?'bg-emerald-50 text-emerald-700 border-emerald-200': e.confidence>=0.65?'bg-amber-50 text-amber-700 border-amber-200':'bg-red-50 text-red-700 border-red-200'}`}>{(e.confidence*100).toFixed(1)}%</span></td>
                      <td className="px-4 py-3 text-xs text-sage-600 whitespace-nowrap">{new Date(e.timestamp).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
