import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { VeronicaEngine, type ViewPreset } from '../engine/VeronicaEngine'
import { GestureController } from '../gesture/GestureController'
import { runtime, useStudio } from '../store/studio'
import { Icon } from './Icon'

const VIEWS: { id: ViewPreset; label: string; key: string }[] = [
  { id: 'photo', label: 'Photo', key: 'C' },
  { id: 'iso', label: 'Iso', key: '' },
  { id: 'front', label: 'Front', key: '' },
  { id: 'left', label: 'Left', key: '' },
  { id: 'right', label: 'Right', key: '' },
  { id: 'top', label: 'Top', key: '' },
]

export function Viewport({ onFiles }: { onFiles: (f: File[]) => void }) {
  const host = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const [drag, setDrag] = useState(false)
  const set = useStudio((s) => s.set)
  const bump = useStudio((s) => s.bump)
  const manifest = useStudio((s) => s.manifest)
  const loading = useStudio((s) => s.loading)
  useStudio((s) => s.rev)

  useEffect(() => {
    const engine = new VeronicaEngine(host.current!)
    runtime.engine = engine
    runtime.gestures = new GestureController(engine)
    const offs = [
      engine.events.on('change', () => bump()),
      engine.events.on('history', () => bump()),
      engine.events.on('selection', (s) => set({ selection: [...s] })),
      engine.events.on('hover', (h) => set({ hovered: h })),
      engine.events.on('stats', (s) => set({ stats: s })),
      engine.events.on('modal', (m) => set({ modal: m })),
      engine.events.on('loading', (l) => set({ loading: l })),
    ]
    return () => {
      offs.forEach((o) => o())
      engine.dispose()
      runtime.engine = null
      runtime.gestures = null
    }
  }, [set, bump])

  // Cmd/Ctrl-drag box selection
  const onPointerDown = (e: React.PointerEvent) => {
    if (!(e.metaKey || e.ctrlKey) || e.button !== 0 || !runtime.engine?.manifest) return
    e.stopPropagation()
    runtime.engine.orbit.enabled = false
    const r = host.current!.getBoundingClientRect()
    setBox({ x0: e.clientX - r.left, y0: e.clientY - r.top, x1: e.clientX - r.left, y1: e.clientY - r.top })
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    if (!box) return
    const r = host.current!.getBoundingClientRect()
    setBox({ ...box, x1: e.clientX - r.left, y1: e.clientY - r.top })
  }
  const onPointerUp = (e: React.PointerEvent) => {
    if (!box) return
    const r = host.current!.getBoundingClientRect()
    const toNdc = (x: number, y: number) => new THREE.Vector2((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1)
    if (Math.hypot(box.x1 - box.x0, box.y1 - box.y0) > 6) runtime.engine!.boxSelect(toNdc(box.x0, box.y0), toNdc(box.x1, box.y1), e.shiftKey ? 'add' : 'set')
    runtime.engine!.orbit.enabled = true
    setBox(null)
  }

  return (
    <div
      className={`viewport ${drag ? 'drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/') || /\.(heic|heif)$/i.test(f.name))
        if (files.length) onFiles(files)
      }}
    >
      <div ref={host} className="viewport-host" onPointerDownCapture={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} />
      {box && (
        <div className="box-select" style={{ left: Math.min(box.x0, box.x1), top: Math.min(box.y0, box.y1), width: Math.abs(box.x1 - box.x0), height: Math.abs(box.y1 - box.y0) }} />
      )}
      {!manifest && !loading && <EmptyState onFiles={onFiles} />}
      {loading && (
        <div className="viewport-loading float" role="status">
          <span>Loading geometry, <span className="tnum">{loading.done} of {loading.total}</span> parts</span>
          <div className="job-bar"><i style={{ width: `${(loading.done / Math.max(1, loading.total)) * 100}%` }} /></div>
        </div>
      )}
      {manifest && <ViewportToolbar />}
      {manifest && <ViewBar />}
      <ModalHud />
      <GestureCursors />
      <Stats />
      {manifest && <Legend />}
    </div>
  )
}

function EmptyState({ onFiles }: { onFiles: (f: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className="empty-state">
      <div className="empty-card">
        <h2>Start with a photo of a room</h2>
        <p>VERONICA measures every object it can see, in metres, and turns each one into a separate 3D object. Anything the photo could not see is estimated and drawn hatched in blue, so you always know which is which.</p>
        <p>Three to eight photos from different angles let it measure the backs as well.</p>
        <div className="actions">
          <input ref={input} type="file" accept="image/*,.heic,.heif" multiple hidden onChange={(e) => { const f = [...(e.target.files ?? [])]; if (f.length) onFiles(f); e.target.value = '' }} />
          <button className="primary" onClick={() => input.current?.click()}><Icon name="upload" size={15} /> Choose photos</button>
          <span className="dim" style={{ alignSelf: 'center' }}>or drop them here</span>
        </div>
      </div>
    </div>
  )
}

function ViewportToolbar() {
  useStudio((s) => s.rev)
  const sel = useStudio((s) => s.selection)
  const e = runtime.engine
  if (!e) return null
  return (
    <div className="vp-toolbar">
      <div className="seg" role="group" aria-label="Transform tool">
        {(['translate', 'rotate', 'scale'] as const).map((m, i) => (
          <button key={m} className={e.gizmoMode === m ? 'on' : ''} title={`${({ translate: 'Move', rotate: 'Rotate', scale: 'Scale' })[m]} (${i + 1})`} aria-label={m} onClick={() => e.setGizmoMode(m)}>
            <Icon name={m} />
          </button>
        ))}
      </div>
      <div className="seg">
        <button className={e.space === 'local' ? 'on' : ''} title="Toggle local / world axes (L)" onClick={() => e.setSpace(e.space === 'world' ? 'local' : 'world')}>
          {e.space === 'world' ? 'World' : 'Local'}
        </button>
        <button className={e.snap.enabled ? 'on' : ''} title="Snapping (Shift+Tab)" aria-label="Snapping" onClick={() => e.setSnap({ enabled: !e.snap.enabled })}>
          <Icon name="magnet" />
        </button>
        <button className={e.precision ? 'on' : ''} title="Precision mode (P)" aria-label="Precision mode" onClick={() => runtime.gestures?.setPrecision(!e.precision)}>
          <Icon name="target" />
        </button>
      </div>
      <div className="seg axis">
        {(['x', 'y', 'z'] as const).map((a) => (
          <button key={a} className={`ax-${a} ${e.axisLock[a] ? 'on' : ''}`} title={`Show ${a.toUpperCase()} handle (Shift+${a.toUpperCase()})`} onClick={() => e.setAxisLock({ ...e.axisLock, [a]: !e.axisLock[a] })}>
            {a.toUpperCase()}
          </button>
        ))}
      </div>
      {sel.length > 1 && <div className="vp-sel float">{sel.length} selected</div>}
    </div>
  )
}

function ViewBar() {
  const manifest = useStudio((s) => s.manifest)
  const cams = manifest?.cameras ?? []
  return (
    <div className="viewbar float" role="group" aria-label="Camera">
      {cams.length > 1 && cams.map((_, i) => (
        <button key={`cam${i}`} onClick={() => runtime.engine!.viewFromCamera(i)} title={`Viewpoint of photo ${i + 1}`}>Photo {i + 1}</button>
      ))}
      {VIEWS.map((v) => (
        <button key={v.id} onClick={() => runtime.engine!.setView(v.id)} title={v.key ? `${v.label} (${v.key})` : v.label}>
          {v.label}
        </button>
      ))}
      <button onClick={() => runtime.engine!.frame()} title="Frame selection (F)">Frame</button>
    </div>
  )
}

function ModalHud() {
  const modal = useStudio((s) => s.modal)
  if (!modal) return null
  const names: Record<string, string> = { translate: 'Grab', rotate: 'Rotate', scale: 'Scale' }
  return (
    <div className="modal-hud float">
      <b>{names[modal.op]}</b>
      {modal.axis && <span className={`ax ax-${modal.axis}`}>{modal.axis.toUpperCase()} axis</span>}
      <span className="tnum">{modal.value}</span>
      <span className="dim">X, Y or Z to constrain, Shift for precision, click or Return to confirm, Esc to cancel</span>
    </div>
  )
}

function GestureCursors() {
  const hud = useStudio((s) => s.gestureHud)
  const on = useStudio((s) => s.gestureOn)
  if (!on || !hud) return null
  return (
    <>
      {hud.hands.map((h) => (
        <div key={h.id} className={`hand-cursor pose-${h.pose}`} style={{ left: `${h.cursor[0] * 100}%`, top: `${h.cursor[1] * 100}%` }}>
          <svg viewBox="-24 -24 48 48" width="48" height="48">
            <circle r={14 - h.pinchStrength * 8} className="halo" />
            <circle r={14 - h.pinchStrength * 8} className="ring" />
            <circle r="2.2" className="dot" />
            <circle r="18" className="progress" strokeDasharray={`${h.pinchStrength * 113} 113`} />
          </svg>
          <span className="hand-tag">{h.pose === 'none' ? '' : h.pose}</span>
        </div>
      ))}
      <div className="gesture-hud">
        <div className="gh-op float">
          <span className={`dot ${hud.op ? 'live' : ''}`} />
          {hud.label}
          {hud.precision && <span className="chip">precision</span>}
        </div>
        <div className="gh-hint">{hud.lastEvent ?? hud.hint}</div>
      </div>
    </>
  )
}

function Stats() {
  const s = useStudio((x) => x.stats)
  const t = useStudio((x) => x.trackerStats)
  const on = useStudio((x) => x.gestureOn)
  return (
    <div className="stats tnum">
      {s.fps} fps, {(s.triangles / 1000).toFixed(0)}k triangles
      {on && t && `, hands ${t.fps} fps at ${t.inferenceMs} ms`}
    </div>
  )
}

function Legend() {
  useStudio((s) => s.rev)
  const e = runtime.engine
  if (!e) return null
  if (e.renderMode !== 'provenance' && !(e.renderMode === 'photo' && e.inferredStyle && e.showInferred)) return null
  return (
    <div className="legend float">
      <span><i className={`sw obs ${e.renderMode === 'provenance' ? 'clay' : ''}`} /> Seen in a photo</span>
      <span><i className={`sw ${e.renderMode === 'provenance' ? 'inf' : 'hatch'}`} /> Estimated, not seen</span>
    </div>
  )
}
