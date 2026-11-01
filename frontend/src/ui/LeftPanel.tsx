import { useRef, useState } from 'react'
import { api } from '../api/client'
import type { VeronicaNode } from '../engine/VeronicaEngine'
import { openScene, reconstruct, refreshScenes } from '../store/actions'
import { runtime, useStudio } from '../store/studio'
import { Icon } from './Icon'

const STAGES = [
  { id: 'load', label: 'Decode' },
  { id: 'depth', label: 'Metric depth' },
  { id: 'pose', label: 'Camera poses' },
  { id: 'detect', label: 'Detect objects' },
  { id: 'segment', label: 'Segment' },
  { id: 'geometry', label: 'Build 3D scene' },
]
const GEN_STAGES = [
  { id: 'generate', label: 'Generate the hidden side' },
  { id: 'align', label: 'Align to surface' },
  { id: 'carve', label: 'Keep only what no photo saw' },
]

export function LeftPanel() {
  return (
    <aside className="panel left">
      <ReconstructCard />
      <Hierarchy />
      <ScenesCard />
    </aside>
  )
}

function ReconstructCard() {
  const job = useStudio((s) => s.job)
  const health = useStudio((s) => s.health)
  const input = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [vocab, setVocab] = useState('')
  const [depthModel, setDepthModel] = useState('')
  const busy = job && (job.status === 'queued' || job.status === 'running')
  const previews = files.map((f) => URL.createObjectURL(f))
  const multi = files.length > 1

  const run = () => {
    if (!files.length) return input.current?.click()
    reconstruct(files, { vocabulary: vocab ? vocab.split(',').map((s) => s.trim()).filter(Boolean) : undefined, depthModel: depthModel || undefined })
    setFiles([])
  }

  return (
    <section className="section">
      <header className="section-h"><h2>Reconstruct</h2></header>
      <input ref={input} type="file" accept="image/*,.heic,.heif" multiple hidden onChange={(e) => setFiles([...(e.target.files ?? [])])} />
      {!busy && (
        <>
          <button className="dropzone" onClick={() => input.current?.click()}>
            {files.length ? (
              <div className="thumbs">
                {previews.slice(0, 6).map((p, i) => <img key={i} src={p} alt="" />)}
                {files.length > 6 && <span>+{files.length - 6}</span>}
              </div>
            ) : (
              <>
                <b>Choose photos</b>
                <span>or drop them on the viewport. One photo works; three to eight from different angles measure the backs too.</span>
              </>
            )}
          </button>
          {files.length > 0 && (
            <p className="pipeline-note">
              {multi
                ? health?.multiView.available
                  ? `${files.length} photos: VERONICA finds where each was taken and fuses them, so object backs seen in any photo are measured.`
                  : `Several photos need multi-view support, which is unavailable: ${health?.multiView.reason ?? 'the backend is offline'}.`
                : 'One photo: every object is measured from the front, and its hidden side is estimated and marked as such.'}
            </p>
          )}
          <details className="adv capture">
            <summary><Icon name="chevron" size={12} /> How to shoot several angles</summary>
            <ul>
              <li>3–8 photos of the same scene, walking around it in steps of roughly 20–40°.</li>
              <li>Keep ~60 % overlap between neighbouring shots; same lighting, nothing moving.</li>
              <li>Include the first photo’s viewpoint; aim at the objects, not the empty wall.</li>
              <li>Backs of objects are only measured if some photo actually sees them.</li>
            </ul>
          </details>
          <details className="adv">
            <summary><Icon name="chevron" size={12} /> Options</summary>
            <label className="field">
              Extra object names
              <input value={vocab} onChange={(e) => setVocab(e.target.value)} placeholder="e.g. guitar, kettlebell (default list covers ~75 indoor classes)" />
            </label>
            <label className="field">
              Depth model
              <select value={depthModel} onChange={(e) => setDepthModel(e.target.value)}>
                <option value="">Default ({health?.depthModel ?? 'depth-pro'})</option>
                {health && Object.entries(health.models).filter(([, m]) => m.role === 'depth').map(([k, m]) => (
                  <option key={k} value={k}>{k} — {m.license.split(' ')[0]}</option>
                ))}
              </select>
            </label>
          </details>
          <button className="primary wide" disabled={multi && !health?.multiView.available} onClick={run}>
            {files.length ? `Reconstruct ${files.length > 1 ? `${files.length} photos` : 'the photo'}` : 'Choose photos to reconstruct'}
          </button>
        </>
      )}
      {job && <JobProgress />}
    </section>
  )
}

function JobProgress() {
  const job = useStudio((s) => s.job)!
  const all = job.kind === 'generate' ? GEN_STAGES : STAGES
  const active = all.findIndex((s) => s.id === job.stage)
  const stages = job.kind === 'generate' ? GEN_STAGES : job.kind === 'multi-view' ? STAGES : STAGES.filter((s) => s.id !== 'pose')
  return (
    <div className={`job ${job.status}`}>
      <div className="job-bar"><i style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
      <ol className="stages">
        {stages.map((s) => {
          const idx = all.findIndex((x) => x.id === s.id)
          const state = job.status === 'done' || idx < active ? 'done' : s.id === job.stage ? 'live' : 'todo'
          return <li key={s.id} className={state}>{state === 'done' ? <Icon name="check" size={14} /> : <span className="stage-dot" />}{s.label}</li>
        })}
      </ol>
      <div className="job-msg">{job.status === 'error' ? job.error : job.message}</div>
      {(job.status === 'running' || job.status === 'queued') && (
        <button className="ghost small" onClick={() => api.cancel(job.id)}>Cancel</button>
      )}
    </div>
  )
}

function when(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const sameDay = d.toDateString() === new Date().toDateString()
  return sameDay ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString([], { day: 'numeric', month: 'short' })
}

function ScenesCard() {
  const scenes = useStudio((s) => s.scenes)
  const current = useStudio((s) => s.sceneId)
  if (!scenes.length) return null
  return (
    <section className="section scenes">
      <header className="section-h"><h2>Scenes</h2><span className="count tnum">{scenes.length}</span></header>
      <ul>
        {scenes.slice(0, 12).map((s) => (
          <li key={s.id} className={s.id === current ? 'on' : ''}>
            <button onClick={() => openScene(s.id)} aria-current={s.id === current}>
              <span>{s.name || s.id}</span>
              <span className="sub">{s.objects} {s.objects === 1 ? 'object' : 'objects'}, {s.mode === 'multi-view' ? 'several photos' : 'one photo'}, {when(s.created)}</span>
            </button>
            <button className="icon-btn" title={`Delete ${s.name || 'this scene'}`} aria-label="Delete scene" onClick={async () => { if (confirm(`Delete the reconstruction of ${s.name || 'this scene'}? This cannot be undone.`)) { await api.deleteScene(s.id); refreshScenes() } }}>
              <Icon name="trash" size={14} />
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Hierarchy() {
  useStudio((s) => s.rev)
  const sel = useStudio((s) => s.selection)
  const hovered = useStudio((s) => s.hovered)
  const manifest = useStudio((s) => s.manifest)
  const set = useStudio((s) => s.set)
  const e = runtime.engine
  if (!e || !manifest) return null
  const top = e.children(null)
  const objects = top.filter((n) => n.kind === 'object' || n.kind === 'group')
  const structure = top.filter((n) => n.kind === 'structure' || n.kind === 'background')
  return (
    <section className="section hierarchy">
      <header className="section-h">
        <h2>Objects</h2><span className="count tnum">{objects.length}</span>
        <span className="grow" />
        <button className="ghost small" title="See the source photo and add objects the detector missed" onClick={() => set({ sourceOpen: true })}>
          <Icon name="photo" size={14} /> Photo
        </button>
      </header>
      <div className="tree">
        <div className="tree-head"><span>Name</span><span title="Share of each object's surface that no photo saw">Seen / estimated</span></div>
        {objects.map((n) => <TreeRow key={n.id} n={n} depth={0} sel={sel} hovered={hovered} />)}
        <div className="tree-head"><span>Room</span></div>
        {structure.map((n) => <TreeRow key={n.id} n={n} depth={0} sel={sel} hovered={hovered} />)}
      </div>
    </section>
  )
}

function TreeRow({ n, depth, sel, hovered }: { n: VeronicaNode; depth: number; sel: string[]; hovered: string | null }) {
  const e = runtime.engine!
  const [open, setOpen] = useState(true)
  const [editing, setEditing] = useState(false)
  const kids = e.children(n.id)
  const on = sel.includes(n.id)
  const inferredPct = n.meta ? Math.round((1 - n.meta.provenance.observedFraction) * 100) : null
  return (
    <>
      <div
        className={`row ${on ? 'on' : ''} ${hovered === n.id ? 'hover' : ''} ${n.visible ? '' : 'hidden'}`}
        style={{ paddingLeft: 8 + depth * 14 }}
        onClick={(ev) => e.select([n.id], ev.shiftKey || ev.metaKey ? 'toggle' : 'set')}
        onDoubleClick={() => (n.kind === 'object' || n.kind === 'group' ? setEditing(true) : e.frame([n.id]))}
        onMouseEnter={() => e.setHovered(n.id)}
        onMouseLeave={() => e.setHovered(null)}
      >
        {kids.length > 0 ? (
          <button className={`twisty ${open ? 'open' : ''}`} onClick={(ev) => { ev.stopPropagation(); setOpen(!open) }}><Icon name="chevron" size={12} /></button>
        ) : <span className="twisty" />}
        <Icon name={n.kind === 'group' ? 'group' : n.kind === 'object' ? 'cube' : 'plane'} size={14} />
        {editing ? (
          <input
            autoFocus
            defaultValue={n.name}
            onClick={(ev) => ev.stopPropagation()}
            onBlur={(ev) => { e.rename(n.id, ev.target.value || n.name); setEditing(false) }}
            onKeyDown={(ev) => { if (ev.key === 'Enter') (ev.target as HTMLInputElement).blur(); if (ev.key === 'Escape') setEditing(false) }}
          />
        ) : (
          <span className="name">{n.name}{n.duplicateOf && <em> copy</em>}</span>
        )}
        {inferredPct !== null && n.kind === 'object' && (
          <span className="split" title={`${100 - inferredPct}% seen in a photo, ${inferredPct}% estimated`}>
            <span className="split-bar"><i style={{ width: `${100 - inferredPct}%` }} /></span>
            <span className="fig">{inferredPct}%</span>
          </span>
        )}
        <button className="icon-btn" title={n.locked ? 'Unlock' : 'Lock'} aria-label={n.locked ? 'Unlock' : 'Lock'} onClick={(ev) => { ev.stopPropagation(); e.setLocked([n.id], !n.locked) }}>
          <Icon name={n.locked ? 'lock' : 'unlock'} size={13} />
        </button>
        <button className="icon-btn" title={n.visible ? 'Hide (H)' : 'Show'} aria-label={n.visible ? 'Hide' : 'Show'} onClick={(ev) => { ev.stopPropagation(); e.setVisible([n.id], !n.visible) }}>
          <Icon name={n.visible ? 'eye' : 'eyeOff'} size={13} />
        </button>
      </div>
      {open && kids.map((k) => <TreeRow key={k.id} n={k} depth={depth + 1} sel={sel} hovered={hovered} />)}
    </>
  )
}
