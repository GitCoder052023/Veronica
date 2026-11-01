import { useEffect, useRef, useState } from 'react'
import { calibratedThresholds, DEFAULT_THRESHOLDS, type Thresholds } from '../gesture/classifier'
import type { GestureMode } from '../gesture/GestureController'
import { HandTracker } from '../gesture/HandTracker'
import { runtime, useStudio } from '../store/studio'
import { Icon } from './Icon'

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]
const CALIB_KEY = 'veronica.gesture.thresholds'
const MODE_LABEL: Record<GestureMode, string> = { auto: 'Auto', navigate: 'Camera', object: 'Objects', scene: 'Scene' }
const MODE_HELP: Record<GestureMode, string> = {
  auto: 'Pinch an object to move it, pinch empty space to orbit',
  navigate: 'Hands only move the camera',
  object: 'Hands only move the selected objects',
  scene: 'Hands move the whole scene',
}

export async function startGestures() {
  const st = useStudio.getState()
  if (!runtime.gestures) return
  try {
    if (!runtime.tracker) {
      runtime.tracker = new HandTracker()
      let lastStats = 0
      runtime.tracker.onResult = (hands, t) => runtime.gestures?.process(hands, t)
      runtime.tracker.onStats = (s) => {
        const now = performance.now()
        if (now - lastStats > 500) {
          lastStats = now
          useStudio.getState().set({ trackerStats: s })
        }
      }
      runtime.gestures.onHud = (h) => useStudio.getState().set({ gestureHud: h })
      runtime.gestures.onPrecision = () => useStudio.getState().bump()
      const saved = localStorage.getItem(CALIB_KEY)
      if (saved) runtime.gestures.gestures.setThresholds({ ...DEFAULT_THRESHOLDS, ...(JSON.parse(saved) as Partial<Thresholds>) })
    }
    st.set({ gestureError: null })
    await runtime.tracker.start(localStorage.getItem('veronica.camera') ?? undefined)
    st.set({ gestureOn: true })
  } catch (e) {
    const msg = (e as Error).name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access for this site and try again.' : `Hand tracking failed to start: ${(e as Error).message}`
    st.set({ gestureError: msg, gestureOn: false })
  }
}

export function stopGestures() {
  runtime.tracker?.stop()
  runtime.gestures?.cancelOp()
  runtime.gestures?.gestures.reset()
  useStudio.getState().set({ gestureOn: false, gestureHud: null })
}

export function GesturePanel() {
  const on = useStudio((s) => s.gestureOn)
  const err = useStudio((s) => s.gestureError)
  const hud = useStudio((s) => s.gestureHud)
  const stats = useStudio((s) => s.trackerStats)
  const set = useStudio((s) => s.set)
  useStudio((s) => s.rev)
  const [collapsed, setCollapsed] = useState(false)
  const [mode, setMode] = useState<GestureMode>(runtime.gestures?.mode ?? 'auto')
  const [calib, setCalib] = useState<null | { step: 'open' | 'pinch' | 'done'; samples: number[]; open?: number; pinch?: number }>(null)
  const [cams, setCams] = useState<MediaDeviceInfo[]>([])

  useEffect(() => {
    if (on) HandTracker.devices().then(setCams).catch(() => {})
  }, [on])

  // calibration sampling driven by HUD updates
  useEffect(() => {
    if (!calib || calib.step === 'done' || !hud?.hands.length) return
    const r = hud.hands[0].pinchRatio
    const samples = [...calib.samples, r]
    if (samples.length < 45) return setCalib({ ...calib, samples })
    const sorted = [...samples].sort((a, b) => a - b)
    const med = sorted[Math.floor(sorted.length / 2)]
    if (calib.step === 'open') setCalib({ step: 'pinch', samples: [], open: med })
    else {
      const th = calibratedThresholds(calib.open!, med)
      runtime.gestures?.gestures.setThresholds(th)
      localStorage.setItem(CALIB_KEY, JSON.stringify({ pinchOn: th.pinchOn, pinchOff: th.pinchOff }))
      setCalib({ ...calib, step: 'done', pinch: med, samples: [] })
    }
  }, [hud]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={`gesture-panel float ${collapsed ? 'collapsed' : ''}`}>
      <header>
        <Icon name="hand" />
        <h2>Hand control</h2>
        <span className="grow" />
        {on && <button className="icon-btn" onClick={() => setCollapsed(!collapsed)} title={collapsed ? 'Expand' : 'Collapse'} aria-label={collapsed ? 'Expand' : 'Collapse'}><Icon name="chevron" size={12} /></button>}
        <button role="switch" aria-checked={on} aria-label="Hand control" className={`switch ${on ? 'on' : ''}`} onClick={() => (on ? stopGestures() : startGestures())} />
      </header>
      {err && <div className="note error">{err}</div>}
      {!on && !err && (
        <div className="gp-off">
          <p>Move objects with your hands in front of the webcam. The video never leaves this browser.</p>
          <div><button className="small" onClick={() => set({ onboarding: true })}>Learn the gestures</button></div>
        </div>
      )}
      {on && !collapsed && (
        <>
          <Preview />
          <div className="seg full" role="group" aria-label="What your hands move">
            {(['auto', 'navigate', 'object', 'scene'] as GestureMode[]).map((m) => (
              <button key={m} className={mode === m ? 'on' : ''} title={MODE_HELP[m]} onClick={() => { runtime.gestures?.setMode(m); setMode(m) }}>{MODE_LABEL[m]}</button>
            ))}
          </div>
          <div className="hands-row">
            {(hud?.hands ?? []).map((h) => (
              <div key={h.id} className="hand-chip">
                <b>{h.id} hand</b> <span className={`pose pose-${h.pose}`}>{h.pose === 'none' ? 'no pose' : h.pose}</span>
                <div className="meter"><i style={{ width: `${h.pinchStrength * 100}%` }} /></div>
              </div>
            ))}
            {!hud?.hands.length && <span className="dim">No hands in view yet</span>}
          </div>
          <div className="gp-foot">
            <button className="small" onClick={() => setCalib({ step: 'open', samples: [] })}>Calibrate pinch</button>
            <button className="ghost small" onClick={() => set({ onboarding: true })}>Gestures</button>
            {cams.length > 1 && (
              <select className="small" defaultValue={localStorage.getItem('veronica.camera') ?? ''} onChange={async (e) => { localStorage.setItem('veronica.camera', e.target.value); stopGestures(); await startGestures() }}>
                {cams.map((c) => <option key={c.deviceId} value={c.deviceId}>{c.label || 'Camera'}</option>)}
              </select>
            )}
          </div>
          {stats && <div className="dim small tnum">{stats.resolution}, {stats.fps} fps, {stats.inferenceMs} ms per frame on the {stats.delegate}</div>}
          {calib && (
            <div className="calib">
              {calib.step === 'open' && <><span>Hold one hand open, facing the camera.</span><Progress n={calib.samples.length} /></>}
              {calib.step === 'pinch' && <><span>Now pinch your thumb and index finger together and hold.</span><Progress n={calib.samples.length} /></>}
              {calib.step === 'done' && (
                <span>Pinch calibrated (open {calib.open?.toFixed(2)}, pinched {calib.pinch?.toFixed(2)}). <button className="link" onClick={() => setCalib(null)}>Done</button></span>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Progress({ n }: { n: number }) {
  return <div className="meter"><i style={{ width: `${(n / 45) * 100}%` }} /></div>
}

/** Mirrored webcam preview with the tracked skeleton drawn on top. */
function Preview() {
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const t = runtime.tracker
    if (!t || !box.current) return
    const v = t.video
    v.className = 'gp-video'
    box.current.prepend(v)
    let raf = 0
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const c = canvas.current
      if (!c) return
      const w = (c.width = c.clientWidth * devicePixelRatio)
      const h = (c.height = c.clientHeight * devicePixelRatio)
      const ctx = c.getContext('2d')!
      ctx.clearRect(0, 0, w, h)
      const g = runtime.gestures?.gestures.opts.activeRegion
      if (g) {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'
        ctx.setLineDash([4, 4])
        ctx.strokeRect(g.x0 * w, g.y0 * h, (g.x1 - g.x0) * w, (g.y1 - g.y0) * h)
        ctx.setLineDash([])
      }
      const res = t.lastResult
      if (!res) return
      res.landmarks.forEach((lm, hi) => {
        const X = (i: number) => (1 - lm[i].x) * w
        const Y = (i: number) => lm[i].y * h
        ctx.strokeStyle = hi === 0 ? 'rgba(227,154,18,0.95)' : 'rgba(169,214,234,0.95)'
        ctx.lineWidth = 2 * devicePixelRatio
        ctx.beginPath()
        for (const [a, b] of CONNECTIONS) { ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)) }
        ctx.stroke()
        ctx.fillStyle = '#ffffff'
        for (let i = 0; i < 21; i++) { ctx.beginPath(); ctx.arc(X(i), Y(i), (i === 4 || i === 8 ? 3.5 : 2) * devicePixelRatio, 0, Math.PI * 2); ctx.fill() }
      })
    }
    draw()
    return () => {
      cancelAnimationFrame(raf)
      v.remove()
    }
  }, [])
  return (
    <div className="gp-preview" ref={box}>
      <canvas ref={canvas} />
    </div>
  )
}
