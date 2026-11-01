import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { addObject } from '../store/actions'
import { runtime, useStudio } from '../store/studio'
import { Icon } from './Icon'

/** The source photograph with the instance map on top. Click a region to select its
 *  object; in "add" mode click something the detector missed to segment it (SAM 2.1
 *  point prompt) and rebuild the scene with it as a new object. */
export function SourceView() {
  const m = useStudio((s) => s.manifest)
  const sceneId = useStudio((s) => s.sceneId)
  const set = useStudio((s) => s.set)
  const job = useStudio((s) => s.job)
  const hovered = useStudio((s) => s.hovered)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [labels, setLabels] = useState<ImageData | null>(null)
  const [adding, setAdding] = useState(false)
  const [label, setLabel] = useState('')
  const [text, setText] = useState('')
  const busy = job && (job.status === 'running' || job.status === 'queued')

  useEffect(() => {
    if (!m || !sceneId) return
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = img.width
      c.height = img.height
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      setLabels(ctx.getImageData(0, 0, c.width, c.height))
    }
    img.src = api.fileUrl(sceneId, m.textures.labels) + `?v=${m.created}`
  }, [m, sceneId])

  const maskToNode = new Map<number, string>()
  m?.nodes.forEach((n) => n.detection && maskToNode.set(n.detection.maskIndex, n.id))

  // overlay: tint every instance, emphasise the hovered/selected one
  useEffect(() => {
    const c = canvas.current
    if (!c || !labels) return
    c.width = labels.width
    c.height = labels.height
    const ctx = c.getContext('2d')!
    const out = ctx.createImageData(labels.width, labels.height)
    const sel = new Set(useStudio.getState().selection)
    for (let i = 0; i < labels.data.length; i += 4) {
      const k = labels.data[i]
      if (!k) continue
      const id = maskToNode.get(k)
      const hue = (k * 67) % 360
      const [r, g, b] = hsl(hue, 0.7, 0.6)
      const strong = id && (id === hovered || sel.has(id))
      out.data[i] = r; out.data[i + 1] = g; out.data[i + 2] = b
      out.data[i + 3] = strong ? 150 : 60
    }
    ctx.putImageData(out, 0, 0)
  }, [labels, hovered]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!m || !sceneId) return null
  const pixelAt = (e: React.MouseEvent) => {
    const r = (e.target as HTMLElement).getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * m.source.width, ((e.clientY - r.top) / r.height) * m.source.height] as [number, number]
  }
  const nodeAt = (e: React.MouseEvent) => {
    if (!labels) return null
    const [x, y] = pixelAt(e)
    const k = labels.data[(Math.floor(y) * labels.width + Math.floor(x)) * 4]
    return k ? maskToNode.get(k) ?? null : null
  }

  return (
    <div className="overlay" onClick={() => set({ sourceOpen: false })}>
      <div className="source-view" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Source photo</h2>
          <span className="dim tnum">{m.source.width} × {m.source.height} px, each detected object tinted</span>
          <span className="grow" />
          <button className={adding ? 'primary small' : 'ghost small'} onClick={() => setAdding(!adding)} disabled={!!busy || m.mode !== 'single-view'}>
            <Icon name="plus" size={14} /> {adding ? 'Now click the missed object' : 'Add a missed object'}
          </button>
          <button className="icon-btn" onClick={() => set({ sourceOpen: false })} aria-label="Close"><Icon name="close" /></button>
        </header>
        <div className={`sv-stage ${adding ? 'adding' : ''}`}>
          <img src={api.fileUrl(sceneId, m.textures.image)} alt={`Source photo ${m.source.name}`} />
          <canvas
            ref={canvas}
            onMouseMove={(e) => runtime.engine?.setHovered(nodeAt(e))}
            onMouseLeave={() => runtime.engine?.setHovered(null)}
            onClick={(e) => {
              if (adding) {
                addObject({ points: [pixelAt(e)], label: label || 'object' })
                setAdding(false)
              } else {
                const id = nodeAt(e)
                if (id) runtime.engine?.select([id], e.shiftKey ? 'toggle' : 'set')
              }
            }}
          />
          {busy && <div className="sv-busy">{job?.message}</div>}
        </div>
        <footer>
          {adding && <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Name for the new object (optional)" />}
          <form onSubmit={(e) => { e.preventDefault(); if (text.trim()) addObject({ text: text.trim() }); setText('') }}>
            <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Find more objects by name, such as cable, plug or sticker" disabled={!!busy || m.mode !== 'single-view'} />
            <button disabled={!text.trim() || !!busy}>Find</button>
          </form>
        </footer>
      </div>
    </div>
  )
}

function hsl(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
  }
  return [f(0), f(8), f(4)]
}
