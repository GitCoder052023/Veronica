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
          <Viewport onFiles={(f) => reconstruct(f)} />
          <GesturePanel />
        </div>
        <RightPanel />
      </main>
      {onboarding && <Onboarding />}
      {sourceOpen && <SourceView />}
      <Toast />
    </div>
  )
}

function TopBar() {
  const health = useStudio((s) => s.health)
  const set = useStudio((s) => s.set)
  const manifest = useStudio((s) => s.manifest)
  useStudio((s) => s.rev)
  const e = runtime.engine
  return (
    <header className="topbar">
      <div className="brand">
        <BrandMark size={22} />
        <b>VERONICA</b>
        <span className="dim">The Spatial Intelligence System</span>
      </div>
      {manifest && <div className="crumb mono">{manifest.source.name} · {manifest.id}</div>}
      <span className="grow" />
      {e && manifest && (
        <div className="seg">
          <button disabled={!e.history.canUndo} onClick={() => e.undo()} title="Undo (⌘Z)"><Icon name="undo" size={15} /></button>
          <button disabled={!e.history.canRedo} onClick={() => e.redo()} title="Redo (⇧⌘Z)"><Icon name="redo" size={15} /></button>
        </div>
      )}
      <button className="ghost small" onClick={() => set({ onboarding: true })}>Guide</button>
      <div className={`backend ${health ? 'ok' : 'down'}`} title={health ? `Models loaded: ${health.loaded.join(', ') || 'none yet'}` : 'Backend offline'}>
        <span className="status-dot live" /> {health ? `inference · ${health.device}` : 'offline'}
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
