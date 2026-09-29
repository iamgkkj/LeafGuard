import { useEffect, useState } from 'react'
import axios from 'axios'

const emojiFor = (name) => {
  if (name.includes('healthy')) return '🌿'
  if (name.includes('Tomato')) return '🍅'
  if (name.includes('Apple')) return '🍎'
  if (name.includes('Corn')) return '🌽'
  if (name.includes('Grape')) return '🍇'
  if (name.includes('Potato')) return '🥔'
  if (name.includes('Pepper')) return '🫑'
  if (name.includes('Strawberry')) return '🍓'
  if (name.includes('Peach')) return '🍑'
  if (name.includes('Cherry')) return '🍒'
  if (name.includes('Blueberry')) return '🫐'
  if (name.includes('Orange')) return '🍊'
  if (name.includes('Soybean')) return '🫘'
  if (name.includes('Squash')) return '🎃'
  if (name.includes('Raspberry')) return '🫐'
  return '🍃'
}

const badgeColor = (name) => {
  if (name.includes('healthy')) return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (name.includes('blight')||name.includes('Blight')) return 'bg-red-50 text-red-700 border-red-200'
  if (name.includes('rust')||name.includes('Rust')) return 'bg-amber-50 text-amber-700 border-amber-200'
  if (name.includes('mildew')) return 'bg-sky-50 text-sky-700 border-sky-200'
  if (name.includes('spot')) return 'bg-orange-50 text-orange-700 border-orange-200'
  return 'bg-leaf-50 text-leaf-700 border-leaf-200'
}

export default function ClassExplorer({ api }) {
  const [classes, setClasses] = useState([])
  const [q, setQ] = useState('')
  const [cropFilter, setCropFilter] = useState('All')
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    axios.get(`${api}/classes`).then(r=>setClasses(r.data.classes)).catch(()=>{})
  }, [api])

  const crops = ['All', ...Array.from(new Set(classes.map(c=>c.crop))).sort()]
  const filtered = classes.filter(c=>{
    const matchesQ = !q || c.name.toLowerCase().includes(q.toLowerCase()) || c.disease.toLowerCase().includes(q.toLowerCase()) || c.crop.toLowerCase().includes(q.toLowerCase())
    const matchesCrop = cropFilter==='All' || c.crop===cropFilter
    return matchesQ && matchesCrop
  })

  return (
    <div className="space-y-4">
      <div className="glass rounded-[24px] shadow-soft p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-leaf-900">Disease Atlas</h2>
            <p className="text-sm text-sage-600">38 PlantVillage classes • Search or filter by crop. Click a card for details.</p>
          </div>
          <span className="bg-leaf-600 text-white text-xs font-bold px-3 py-1.5 rounded-full">{filtered.length} / {classes.length} shown</span>
        </div>
        <div className="mt-4 grid grid-cols-1 lg:grid-cols-[1fr_200px] gap-3">
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sage-400">⌕</span>
            <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search disease, crop, e.g. 'blight', 'tomato', 'rust'… " className="w-full border border-leaf-200 rounded-xl pl-9 pr-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-leaf-200" />
          </div>
          <select value={cropFilter} onChange={e=>setCropFilter(e.target.value)} className="border border-leaf-200 rounded-xl px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-leaf-200">
            {crops.map(c=><option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {crops.slice(0,8).map(c=>(
            <button key={c} onClick={()=>setCropFilter(c)} className={`text-xs px-2.5 py-1 rounded-full border font-medium transition ${cropFilter===c?'bg-leaf-600 text-white border-leaf-600':'bg-white text-leaf-700 border-leaf-200 hover:bg-leaf-50'}`}>{c}</button>
          ))}
        </div>
      </div>

      {filtered.length===0 ? (
        <div className="glass rounded-[24px] p-12 text-center">
          <div className="text-3xl mb-2">🔍</div>
          <p className="font-semibold text-leaf-900">No matches</p>
          <p className="text-sm text-sage-600">Try a different crop filter or search term.</p>
          <button onClick={()=>{setQ(''); setCropFilter('All')}} className="mt-3 text-sm bg-leaf-600 text-white px-4 py-2 rounded-xl">Clear filters</button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map(c=>(
            <button key={c.idx} onClick={()=>setSelected(c)} className="text-left group relative bg-white rounded-[20px] border border-leaf-100 p-4 shadow-sm hover:shadow-soft-lg hover:-translate-y-1 hover:border-leaf-200 transition-all">
              <div className="flex items-center justify-between gap-2 mb-3">
                <span className="text-[11px] font-mono bg-leaf-50 border border-leaf-100 px-2 py-1 rounded-full text-leaf-700">#{String(c.idx).padStart(2,'0')}</span>
                <span className={`text-[11px] font-semibold px-2 py-1 rounded-full border ${badgeColor(c.name)}`}>{c.crop}</span>
              </div>
              <div className="w-full h-[88px] rounded-2xl bg-gradient-to-br from-leaf-50 via-white to-earth-50 border border-leaf-100 flex items-center justify-center text-[36px] group-hover:scale-105 transition">{emojiFor(c.name)}</div>
              <h3 className="font-bold text-sm text-leaf-900 leading-tight mt-3 line-clamp-2">{c.disease}</h3>
              <p className="text-[11px] font-mono text-sage-500 truncate mt-1">{c.name}</p>
              <p className="text-xs text-sage-600 mt-2 line-clamp-2 leading-relaxed">{c.description}</p>
              <span className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-leaf-700 group-hover:gap-1.5 transition-all">View details →</span>
            </button>
          ))}
        </div>
      )}

      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-leaf-950/40 backdrop-blur-sm" onClick={()=>setSelected(null)} />
          <div className="relative bg-white rounded-[24px] shadow-2xl max-w-lg w-full overflow-hidden animate-slide-up max-h-[90vh] flex flex-col">
            <div className="h-28 bg-gradient-to-br from-leaf-600 to-emerald-600 flex items-center justify-center text-5xl relative">
              <span>{emojiFor(selected.name)}</span>
              <button onClick={()=>setSelected(null)} className="absolute top-3 right-3 w-8 h-8 rounded-full bg-white/90 flex items-center justify-center hover:bg-white">✕</button>
              <span className="absolute bottom-3 left-4 bg-white/90 backdrop-blur px-2.5 py-1 rounded-full text-xs font-bold text-leaf-800">#{selected.idx} • {selected.crop}</span>
            </div>
            <div className="p-5 overflow-auto">
              <h3 className="text-lg font-extrabold text-leaf-900">{selected.disease}</h3>
              <p className="text-xs font-mono text-sage-500 mt-1 break-all">{selected.name}</p>
              <p className="text-sm text-sage-700 mt-3 leading-relaxed">{selected.description}</p>
              <div className="mt-4 rounded-xl bg-leaf-50 border border-leaf-100 p-3 text-xs">
                <p className="font-semibold text-leaf-800">Agronomist tip</p>
                <p className="text-sage-700 mt-1">Early detection via LeafGuard helps reduce spread. Isolate affected leaves, improve airflow, and apply targeted treatment per disease.</p>
              </div>
              <div className="mt-4 flex gap-2">
                <button onClick={()=>setSelected(null)} className="flex-1 bg-leaf-600 text-white py-2.5 rounded-xl font-semibold">Close</button>
                <button onClick={()=>{setSelected(null); setCropFilter(selected.crop)}} className="px-4 py-2.5 rounded-xl border border-leaf-200 font-medium hover:bg-leaf-50">Filter {selected.crop}</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
