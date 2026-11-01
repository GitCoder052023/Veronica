import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { RENDER_MODES } from '../engine/material'
import { VeronicaEngine, type VeronicaNode } from '../engine/VeronicaEngine'
import { download, exportGLB, exportOBJ, exportPLY } from '../engine/exporters'
import { addAngles, generateHidden, revertHidden } from '../store/actions'
import { runtime, useStudio } from '../store/studio'
import { Icon } from './Icon'

export function RightPanel() {
  const sel = useStudio((s) => s.selection)
  const manifest = useStudio((s) => s.manifest)
  useStudio((s) => s.rev)
  if (!manifest || !runtime.engine) return <aside className="panel right"><About /></aside>
  const e = runtime.engine
  const nodes = sel.map((id) => e.nodes.get(id)).filter(Boolean) as VeronicaNode[]
  return (
    <aside className="panel right">
      {nodes.length === 1 ? <Inspector n={nodes[0]} /> : nodes.length > 1 ? <MultiInspector nodes={nodes} /> : <SceneInfo />}
      <Display />
      <SceneOps />
      <Export />
    </aside>
  )
}

function Vec3Field({ label, value, step, onCommit, disabled, unit }: { label: string; value: number[]; step: number; onCommit: (v: number[]) => void; disabled?: boolean; unit: string }) {
  return (
    <div className="vec3">
      <span className="vec-label">{label}</span>
      {value.map((v, i) => (
        <NumberInput key={i} axis={'xyz'[i]} value={v} step={step} disabled={disabled} onCommit={(nv) => { const c = [...value]; c[i] = nv; onCommit(c) }} />
      ))}
      <span className="unit">{unit}</span>
    </div>
  )
}

/** Number field that also scrubs: drag horizontally on the axis letter. */
function NumberInput({ axis, value, step, onCommit, disabled }: { axis: string; value: number; step: number; onCommit: (v: number) => void; disabled?: boolean }) {
  const [text, setText] = useState(value.toFixed(3))
  useEffect(() => setText(value.toFixed(step >= 1 ? 1 : 3)), [value, step])
  const scrub = (ev: React.PointerEvent) => {
    if (disabled) return
    const x0 = ev.clientX
    const v0 = value
    const move = (m: PointerEvent) => onCommit(v0 + Math.round((m.clientX - x0) / 2) * step * (m.shiftKey ? 0.1 : 1))
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }
  return (
    <label className={`num ax-${axis}`}>
      <span onPointerDown={scrub}>{axis.toUpperCase()}</span>
      <input
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => { const v = parseFloat(text); if (Number.isFinite(v) && v !== value) onCommit(v); else setText(value.toFixed(3)) }}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
      />
    </label>
  )
}

function Inspector({ n }: { n: VeronicaNode }) {
  const e = runtime.engine!
  const o = n.object
  const pos = o.position.toArray()
  const rot = [o.rotation.x, o.rotation.y, o.rotation.z].map(THREE.MathUtils.radToDeg)
  const scl = o.scale.toArray()
  const dims = n.meta ? n.meta.dimensions.map((d, i) => d * Math.abs(scl[i])) : null
  const prov = n.meta?.provenance
  const det = n.meta?.detection
  return (
    <section className="section inspector">
      <header className="section-h">
        <h2>{n.name}</h2>
        {n.kind === 'group' ? <span className="tag">Group</span> : n.label && n.label.toLowerCase() !== n.name.toLowerCase() && <span className="tag">{n.label}</span>}
      </header>
      {n.locked && (
        <div className="note">
          Locked {n.kind === 'object' ? 'object' : 'room structure'}.{' '}
          <button className="link" onClick={() => e.setLocked([n.id], false)}>Unlock to edit</button>
        </div>
      )}
      {dims && (
        <div className="lead">
          <span className="fig">{dims.map((d) => (d * 100).toFixed(1)).join(' × ')} <small>cm</small></span>
          <span className="label">Width × height × depth, measured from the photo</span>
        </div>
      )}
      {prov && <Provenance prov={prov} />}
      {dims && (
        <dl className="kv">
          {n.meta?.relations?.supportedBy && <><dt>Rests on</dt><dd>{e.nodes.get(n.meta.relations.supportedBy)?.name ?? n.meta.relations.supportedBy}</dd></>}
          {prov?.meanDepth !== undefined && <><dt>Distance</dt><dd className="tnum">{prov.meanDepth.toFixed(2)} m from the camera</dd></>}
          {det && <><dt>Detected as</dt><dd>{n.label}, {(det.score * 100).toFixed(0)}% confident, mask {(det.maskQuality * 100).toFixed(0)}%</dd></>}
        </dl>
      )}
      <Vec3Field label="Position" unit="m" value={pos} step={0.01} disabled={n.locked} onCommit={(v) => e.setTransform(n.id, { position: v })} />
      <Vec3Field label="Rotation" unit="°" value={rot} step={1} disabled={n.locked} onCommit={(v) => e.setTransform(n.id, { rotationDeg: v })} />
      <Vec3Field label="Scale" unit="×" value={scl} step={0.01} disabled={n.locked} onCommit={(v) => e.setTransform(n.id, { scale: v })} />
      <UniformScale n={n} value={scl[0]} />
      <div className="actions">
        <button title="Reset transform (Alt+G/R/S for parts)" disabled={n.locked} onClick={() => e.resetTransform([n.id])}><Icon name="reset" size={14} /> Reset</button>
        <button title="Duplicate (Shift+D)" disabled={n.locked} onClick={() => e.duplicate([n.id])}><Icon name="copy" size={14} /> Duplicate</button>
        <button title="Isolate (I)" className={e.isolateId === n.id ? 'on' : ''} onClick={() => e.isolate(n.id)}><Icon name="isolate" size={14} /> Isolate</button>
        <button title="Frame (F)" onClick={() => e.frame([n.id])}><Icon name="focus" size={14} /> Frame</button>
        {n.kind === 'group' && <button onClick={() => e.ungroup(n.id)}><Icon name="group" size={14} /> Ungroup</button>}
        <button className="danger" title="Delete (X / Del)" onClick={() => e.deleteNodes([n.id])}><Icon name="trash" size={14} /> Delete</button>
      </div>
      {n.kind === 'object' && n.meta && <HiddenSide n={n} />}
    </section>
  )
}

/** Slider for uniform scale. One manipulation session per drag, so one undo step. */
function UniformScale({ n, value }: { n: VeronicaNode; value: number }) {
  const e = runtime.engine!
  const s0 = useRef<{ v: number; p: THREE.Vector3 } | null>(null)
  return (
    <div className="uniform">
      <span className="vec-label">Uniform</span>
      <input type="range" min={0.1} max={4} step={0.01} value={value} disabled={n.locked}
        onPointerDown={() => { if (e.beginManipulation([n.id], 'uniform scale')) s0.current = { v: value, p: e.selectionCenter([n.id])! } }}
        onChange={(ev) => {
          if (!s0.current) return
          const f = parseFloat(ev.target.value) / s0.current.v
          e.applyManipulation(VeronicaEngine.aboutPoint(s0.current.p, new THREE.Matrix4().makeScale(f, f, f)))
          useStudio.getState().bump()
        }}
        onPointerUp={() => { s0.current = null; e.endManipulation(true) }} />
      <span className="tnum small">{value.toFixed(2)}×</span>
    </div>
  )
}

function Provenance({ prov }: { prov: NonNullable<VeronicaNode['meta']>['provenance'] }) {
  const pct = Math.round(prov.observedFraction * 100)
  const params = prov.completionParams ?? {}
  const methodText: Record<string, string> = {
    'mirror-symmetry': 'Hidden side mirrored across the object’s vertical symmetry planes',
    'surface-of-revolution': 'Hidden side swept from the visible profile (rotational symmetry)',
    'ray-plane continuation': 'Extended behind objects along the fitted plane',
    generative: `Hidden side generated by ${String(params.model ?? 'a generative model')} from ${String(params.views ?? 1)} view(s), aligned to the measured surface (silhouette IoU ${Number(params.silhouette_iou ?? 0).toFixed(2)}, RMS ${(Number(params.surface_rms_m ?? 0) * 100).toFixed(1)} cm) and cut back to what no photo saw`,
    'telea depth inpainting': 'Holes behind objects filled by depth inpainting',
    none: 'Nothing estimated: only what the photo shows',
  }
  return (
    <div className="provenance">
      <div className="prov-bar"><i style={{ width: `${pct}%` }} /></div>
      <div className="prov-legend tnum"><span>{pct}% seen in a photo</span><span className="est">{100 - pct}% estimated</span></div>
      <p>{methodText[prov.completion] ?? prov.completion}.{'depth_source' in params && ` Depth taken from the ${String(params.depth_source)} (${(Number(params.depth_m) * 100).toFixed(0)} cm).`}</p>
    </div>
  )
}

type Completer = NonNullable<ReturnType<typeof useStudio.getState>['health']>['completer']

/** Short model name for buttons: "Hunyuan3D-2mini", "TRELLIS-image-large". */
function completerName(c: Completer) {
  return c.engine === 'hunyuan3d-mini' ? 'Hunyuan3D-2mini' : String(c.model ?? 'completer').split('/').pop()
}

/** Where the crops go and under which licence, shown before anything is generated. */
function CompleterNote({ c }: { c: Completer }) {
  return (
    <>
      <p>
        {c.local ? `Runs on this machine (${c.device}); nothing leaves it.` : `Sends this object’s cut-out to ${c.url} (${c.device}).`}
        {c.local && !c.loaded && ' The first run loads the model, about 4 GB, downloaded once.'}
      </p>
      {c.license && <p className="licence">Licence: {c.license}.</p>}
    </>
  )
}

/** How the unseen side is produced. With a completer worker running (Hunyuan3D-2mini on
 *  this Mac, or TRELLIS on a CUDA machine) it can be generated; otherwise the way to get
 *  real hidden sides is more photos from other angles. */
function HiddenSide({ n }: { n: VeronicaNode }) {
  const health = useStudio((s) => s.health)
  const job = useStudio((s) => s.job)
  const manifest = useStudio((s) => s.manifest)
  const c = health?.completer
  const busy = !!job && (job.status === 'running' || job.status === 'queued')
  const isGen = n.meta!.provenance.completion === 'generative'
  const inferred = 1 - n.meta!.provenance.observedFraction
  if (!c?.configured) {
    return (
      <div className="hidden-side">
        <h3 className="vec-label">Hidden side</h3>
        {isGen && <button disabled={busy} onClick={() => revertHidden(n.id)}><Icon name="reset" size={14} /> Go back to the symmetry estimate</button>}
        {inferred > 0.05 && (
          <>
            <p>
              {manifest?.mode === 'multi-view'
                ? `${Math.round(inferred * 100)}% of this object was not seen from any of the ${manifest.source.views} photos.`
                : `${Math.round(inferred * 100)}% of this object is estimated from symmetry.`}{' '}
              Photos from other angles turn the estimate into measured geometry.
            </p>
            <AddAnglesButton />
          </>
        )}
      </div>
    )
  }
  return (
    <div className="hidden-side">
      <h3 className="vec-label">Hidden side</h3>
      {isGen ? (
        <button disabled={busy} onClick={() => revertHidden(n.id)}><Icon name="reset" size={14} /> Go back to the symmetry estimate</button>
      ) : (
        <button disabled={busy || !c.ok} title={c.ok ? `${c.model} on ${c.device}` : undefined} onClick={() => generateHidden([n.id])}>
          <Icon name="spark" size={14} /> {c.ok ? `Generate with ${completerName(c)}` : 'Generator offline'}
        </button>
      )}
      {!c.ok && <p>The hidden-side generator at {c.url} is not answering: {c.reason ?? c.error}</p>}
      {c.ok && !isGen && <CompleterNote c={c} />}
      {inferred > 0.05 && <AddAnglesButton />}
    </div>
  )
}

/** Pick extra photos of the same place and rebuild the scene from all of them. */
export function AddAnglesButton({ label = 'Add photos from other angles' }: { label?: string }) {
  const input = useRef<HTMLInputElement>(null)
  const job = useStudio((s) => s.job)
  const manifest = useStudio((s) => s.manifest)
  const busy = !!job && (job.status === 'running' || job.status === 'queued')
  const used = manifest?.source.views ?? 1
  return (
    <>
      <input ref={input} type="file" accept="image/*,.heic,.heif" multiple hidden onChange={(e) => { addAngles([...(e.target.files ?? [])]); e.target.value = '' }} />
      <button disabled={busy || used >= 12} title={used >= 12 ? 'This scene already uses 12 photos' : 'Rebuilds as a new multi-view scene; this one is kept'} onClick={() => input.current?.click()}>
        <Icon name="camera" size={14} /> {label}
      </button>
    </>
  )
}

function MultiInspector({ nodes }: { nodes: VeronicaNode[] }) {
  const e = runtime.engine!
  const ids = nodes.map((n) => n.id)
  return (
    <section className="section inspector">
      <header className="section-h"><h2>{nodes.length} objects selected</h2></header>
      <p className="dim">{nodes.map((n) => n.name).join(', ')}</p>
      <div className="actions">
        <button onClick={() => e.group(ids)}><Icon name="group" size={14} /> Group</button>
        <button onClick={() => e.duplicate(ids)}><Icon name="copy" size={14} /> Duplicate</button>
        <button onClick={() => e.resetTransform(ids)}><Icon name="reset" size={14} /> Reset</button>
        <button onClick={() => e.setVisible(ids, false)}><Icon name="eyeOff" size={14} /> Hide</button>
        <button onClick={() => e.frame(ids)}><Icon name="focus" size={14} /> Frame</button>
        <button className="danger" onClick={() => e.deleteNodes(ids)}><Icon name="trash" size={14} /> Delete</button>
      </div>
    </section>
  )
}

function SceneInfo() {
  const m = useStudio((s) => s.manifest)!
  const objs = m.nodes.filter((n) => n.kind === 'object')
  const obs = objs.reduce((a, n) => a + n.provenance.observedFraction, 0) / Math.max(1, objs.length)
  return (
    <section className="section">
      <header className="section-h"><h2>Scene</h2></header>
      <div className="lead">
        <span className="fig tnum">{(obs * 100).toFixed(0)}% <small>seen in a photo</small></span>
        <span className="label">Average over {objs.length} {objs.length === 1 ? 'object' : 'objects'}; the rest is estimated and drawn hatched in blue.</span>
      </div>
      <dl className="kv">
        <dt>{m.source.views === 1 ? 'Photo' : 'Photos'}</dt><dd className="tnum">{m.source.views === 1 ? 'One' : m.source.views}, {m.source.width} × {m.source.height} px</dd>
        <dt>Camera</dt><dd className="tnum">{m.camera.hfov.toFixed(1)}° wide, focal length from the {m.camera.focalSource}</dd>
        <dt>Up direction</dt><dd>From the {m.world.upSource}</dd>
        <dt>Took</dt><dd className="tnum">{Object.values(m.pipeline.timings).reduce((a, b) => a + b, 0).toFixed(1)} s ({Object.entries(m.pipeline.timings).filter(([, v]) => v >= 0.5).map(([k, v]) => `${k} ${v.toFixed(1)} s`).join(', ')})</dd>
      </dl>
      <ul className="uncertainty">
        <li>{m.world.scaleNote}</li>
        {m.uncertainty.notes.map((n, i) => <li key={i}>{n}</li>)}
      </ul>
      <GenerateAll />
      <details className="models">
        <summary><Icon name="chevron" size={12} /> Models and licences</summary>
        {m.pipeline.models.map((x) => (
          <div key={x.key}><b>{x.id}</b><span>{x.role[0].toUpperCase() + x.role.slice(1)}, {x.license}</span></div>
        ))}
      </details>
      <p className="dim">Click an object or pinch it to select it; ⌘-drag selects several.</p>
    </section>
  )
}

function GenerateAll() {
  const c = useStudio((s) => s.health?.completer)
  const job = useStudio((s) => s.job)
  const manifest = useStudio((s) => s.manifest)
  const busy = !!job && (job.status === 'running' || job.status === 'queued')
  if (!c?.ok) {
    const objs = manifest?.nodes.filter((n) => n.kind === 'object') ?? []
    const inferred = objs.reduce((a, n) => a + 1 - n.provenance.observedFraction, 0) / Math.max(1, objs.length)
    if (inferred < 0.1) return null
    return <AddAnglesButton label={manifest?.mode === 'multi-view' ? 'Add more angles' : 'Add photos from other angles'} />
  }
  return (
    <button disabled={busy} onClick={() => generateHidden()} title={`${c.model} on ${c.device}`}>
      <Icon name="spark" size={14} /> Generate every hidden side
    </button>
  )
}

const MODE_LABEL: Record<string, string> = { photo: 'Photo', clay: 'Clay', provenance: 'Coverage', normals: 'Normals' }

function Display() {
  const e = runtime.engine!
  return (
    <section className="section">
      <header className="section-h"><h2>Display</h2></header>
      <div className="seg full" role="group" aria-label="Shading">
        {RENDER_MODES.map((m) => <button key={m} className={e.renderMode === m ? 'on' : ''} onClick={() => e.setRenderMode(m)}>{MODE_LABEL[m] ?? m}</button>)}
      </div>
      <div className="seg full" role="group" aria-label="Geometry">
        {(['hybrid', 'observed', 'points'] as const).map((m) => <button key={m} className={e.repMode === m ? 'on' : ''} onClick={() => e.setRepMode(m)}>{m === 'hybrid' ? 'Everything' : m === 'observed' ? 'Seen only' : 'Points'}</button>)}
      </div>
      <label className="check"><input type="checkbox" checked={e.showInferred} onChange={(ev) => e.setShowInferred(ev.target.checked)} /> Show estimated surfaces</label>
      <label className="check"><input type="checkbox" checked={e.inferredStyle} onChange={(ev) => e.setInferredStyle(ev.target.checked)} /> Hatch estimated surfaces in blue</label>
      <label className="check"><input type="checkbox" checked={e.gridVisible} onChange={(ev) => e.setGridVisible(ev.target.checked)} /> Floor grid</label>
      <label className="check"><input type="checkbox" checked={e.showCameras} onChange={(ev) => e.setShowCameras(ev.target.checked)} /> Photo cameras</label>
    </section>
  )
}

function SceneOps() {
  const e = runtime.engine!
  const rot = (deg: number) => e.transformScene(VeronicaEngine.aboutPoint(e.sceneCenter(), new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(deg))), 'rotate scene')
  const scl = (f: number) => e.transformScene(VeronicaEngine.aboutPoint(e.sceneCenter(), new THREE.Matrix4().makeScale(f, f, f)), 'scale scene')
  return (
    <section className="section">
      <header className="section-h"><h2>Whole scene</h2></header>
      <div className="actions">
        <button onClick={() => rot(-15)} title="Turn the scene 15° left"><Icon name="turnLeft" size={14} /> 15°</button>
        <button onClick={() => rot(15)} title="Turn the scene 15° right"><Icon name="turnRight" size={14} /> 15°</button>
        <button onClick={() => scl(1 / 1.25)} title="Shrink the scene by 20%"><Icon name="shrink" size={14} /> Smaller</button>
        <button onClick={() => scl(1.25)} title="Enlarge the scene by 25%"><Icon name="grow" size={14} /> Larger</button>
        <button onClick={() => e.showAll()}><Icon name="eye" size={14} /> Show all</button>
        <button className="danger" onClick={() => e.restoreOriginal()} title="Undo every edit and return to the reconstruction"><Icon name="reset" size={14} /> Restore original</button>
      </div>
      <div className="history">
        <div className="history-h">
          <span>History</span>
          <button className="icon-btn" disabled={!e.history.canUndo} title="Undo (⌘Z)" onClick={() => e.undo()}><Icon name="undo" size={14} /></button>
          <button className="icon-btn" disabled={!e.history.canRedo} title="Redo (⇧⌘Z)" onClick={() => e.redo()}><Icon name="redo" size={14} /></button>
        </div>
        <ol>
          {[...e.history.entries].slice(-6).reverse().map((h, i) => <li key={h.time + ':' + i}>{h.label}</li>)}
          {!e.history.entries.length && <li className="dim">No edits yet</li>}
        </ol>
      </div>
    </section>
  )
}

function Export() {
  const e = runtime.engine!
  const name = (useStudio.getState().manifest?.source.name ?? 'scene').replace(/\.[^.]+$/, '')
  const [opts, setOpts] = useState({ includeInferred: true, includeStructure: true, selectionOnly: false })
  const [busy, setBusy] = useState<string | null>(null)
  const notify = useStudio((s) => s.notify)
  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label)
    try { await fn() } catch (err) { notify(`${label} failed: ${(err as Error).message}`, 'error') } finally { setBusy(null) }
  }
  return (
    <section className="section">
      <header className="section-h"><h2>Export</h2></header>
      <label className="check"><input type="checkbox" checked={opts.includeInferred} onChange={(ev) => setOpts({ ...opts, includeInferred: ev.target.checked })} /> Include estimated surfaces</label>
      <label className="check"><input type="checkbox" checked={opts.includeStructure} onChange={(ev) => setOpts({ ...opts, includeStructure: ev.target.checked })} /> Include room structure</label>
      <label className="check"><input type="checkbox" checked={opts.selectionOnly} onChange={(ev) => setOpts({ ...opts, selectionOnly: ev.target.checked })} /> Selection only</label>
      <div className="actions">
        <button disabled={!!busy} title="glTF binary: textured meshes, current transforms & hierarchy" onClick={() => run('GLB', async () => download(await exportGLB(e, opts), `${name}.glb`))}>GLB</button>
        <button disabled={!!busy} title="Wavefront OBJ: geometry + UVs only (no embedded texture)" onClick={() => run('OBJ', async () => download(exportOBJ(e, opts), `${name}.obj`))}>OBJ</button>
        <button disabled={!!busy} title="PLY point cloud of observed points with colour" onClick={() => run('PLY', async () => download(await exportPLY(e, opts), `${name}.ply`))}>PLY points</button>
        <button disabled={!!busy} onClick={() => run('Screenshot', async () => download(await e.screenshot(), `${name}.png`))}><Icon name="image" size={14} /> PNG</button>
        {!e.isRecording ? (
          <>
            <button onClick={() => e.startRecording(false)}><Icon name="record" size={14} /> Record</button>
            <button onClick={() => e.startRecording(true)}><Icon name="record" size={14} /> Turntable</button>
          </>
        ) : (
          <button className="danger" onClick={() => run('Video', async () => { const b = await e.stopRecording(); if (b) download(b, `${name}.webm`) })}><Icon name="stop" size={14} /> Stop & save</button>
        )}
      </div>
      {busy && <p className="dim">Exporting {busy}…</p>}
      <p className="dim small">GLB keeps textures and the object hierarchy. OBJ has geometry only; PLY has only the points seen in a photo.</p>
    </section>
  )
}

function About() {
  const health = useStudio((s) => s.health)
  return (
    <section className="section">
      <header className="section-h"><h2>VERONICA, the spatial intelligence system</h2></header>
      <p className="dim">Turns photos of a room into a measured 3D scene of separate objects you can move with a mouse or your hands. What no photo saw is estimated and always marked.</p>
      {health && (
        <dl className="kv">
          <dt>Runs on</dt><dd>{({ mps: 'Apple GPU (Metal)', cuda: 'NVIDIA GPU', cpu: 'CPU' } as Record<string, string>)[health.device] ?? health.device}</dd>
          <dt>Several photos</dt><dd>{health.multiView.available ? `Supported (${health.multiView.model})` : `Unavailable: ${health.multiView.reason}`}</dd>
        </dl>
      )}
      <p className="dim small">Open source under the Apache License 2.0. Each model keeps its own licence (see NOTICE).</p>
    </section>
  )
}
