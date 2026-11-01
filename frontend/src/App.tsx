import { useEffect } from 'react'
import { openScene, reconstruct, refreshHealth, refreshScenes } from './store/actions'
import { runtime, useStudio } from './store/studio'
import { GesturePanel } from './ui/GesturePanel'
import { BrandMark, Icon } from './ui/Icon'
import { LeftPanel } from './ui/LeftPanel'
import { Onboarding } from './ui/Onboarding'
import { RightPanel } from './ui/RightPanel'
import { bindShortcuts } from './ui/shortcuts'
import { SourceView } from './ui/SourceView'
import { Viewport } from './ui/Viewport'

export default function App() {
  const onboarding = useStudio((s) => s.onboarding)
  const sourceOpen = useStudio((s) => s.sourceOpen)
  const healthError = useStudio((s) => s.healthError)
  const set = useStudio((s) => s.set)

  useEffect(() => {
    const off = bindShortcuts()
    ;(async () => {
      await refreshHealth()
      await refreshScenes()
      const last = localStorage.getItem('veronica.lastScene')
      const scenes = useStudio.getState().scenes
      const target = scenes.find((s) => s.id === last)?.id ?? scenes[0]?.id
      if (target) await openScene(target).catch(() => {})
    })()
    if (!localStorage.getItem('veronica.onboarded')) set({ onboarding: true })
    const iv = setInterval(refreshHealth, 15000)
    return () => {
      off()
      clearInterval(iv)
      runtime.tracker?.dispose()
    }
  }, [set])

  return (
    <div className="app">
      <TopBar />
      {healthError && <div className="banner">{healthError}</div>}
      <main className="workspace">
        <LeftPanel />
        <div className="center">
          <div className="sheet">
            <Viewport onFiles={(f) => reconstruct(f)} />
            <GesturePanel />
          </div>
        </div>
        <RightPanel />
      </main>
      {onboarding && <Onboarding />}
      {sourceOpen && <SourceView />}
      <Toast />
    </div>
  )
}

const DEVICE_NAME: Record<string, string> = { mps: 'Apple GPU', cuda: 'NVIDIA GPU', cpu: 'CPU' }

function TopBar() {
  const health = useStudio((s) => s.health)
  const set = useStudio((s) => s.set)
  const manifest = useStudio((s) => s.manifest)
  useStudio((s) => s.rev)
  const e = runtime.engine
  const views = manifest?.source.views ?? 1
  const objects = manifest?.nodes.filter((n) => n.kind === 'object').length ?? 0
  return (
    <header className="topbar">
      <div className="brand">
        <BrandMark size={22} />
        <b>VERONICA</b>
      </div>
      {manifest ? (
        <div className="title">
          <h1>{manifest.source.name || 'Untitled scene'}</h1>
          <span>{views === 1 ? 'One photo' : `${views} photos`}, {objects} {objects === 1 ? 'object' : 'objects'}</span>
        </div>
      ) : (
        <div className="title"><h1>New scene</h1></div>
      )}
      <span className="grow" />
      {e && manifest && (
        <>
          <button className="icon-btn" disabled={!e.history.canUndo} onClick={() => e.undo()} title="Undo (⌘Z)" aria-label="Undo"><Icon name="undo" size={16} /></button>
          <button className="icon-btn" disabled={!e.history.canRedo} onClick={() => e.redo()} title="Redo (⇧⌘Z)" aria-label="Redo"><Icon name="redo" size={16} /></button>
        </>
      )}
      <button className="ghost small" onClick={() => set({ onboarding: true })}>Guide</button>
      <div className={`backend ${health ? 'ok' : 'down'}`} title={health ? `Models loaded: ${health.loaded.join(', ') || 'none yet'}` : 'The VERONICA API is not answering'}>
        <span className={`status-dot ${health ? 'live' : ''}`} /> <span className="label">{health ? `Running on ${DEVICE_NAME[health.device] ?? health.device}` : 'Backend offline'}</span>
      </div>
    </header>
  )
}

function Toast() {
  const toast = useStudio((s) => s.toast)
  const set = useStudio((s) => s.set)
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => set({ toast: null }), toast.kind === 'error' ? 7000 : 3000)
    return () => clearTimeout(t)
  }, [toast, set])
  if (!toast) return null
  return <div className={`toast ${toast.kind}`}>{toast.text}</div>
}
