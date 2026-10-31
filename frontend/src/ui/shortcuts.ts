import { RENDER_MODES } from '../engine/material'
import { runtime } from '../store/studio'

/** Blender-flavoured keyboard map. Uses KeyboardEvent.code so Option-modified keys
 *  on macOS still resolve to their letter. */
export function bindShortcuts(): () => void {
  const onKey = (ev: KeyboardEvent) => {
    const t = ev.target as HTMLElement
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
    const e = runtime.engine
    if (!e || !e.manifest) return
    const mod = ev.metaKey || ev.ctrlKey
    const sel = e.selection
    const k = ev.code

    if (e.inModal) {
      if (k === 'KeyX' || k === 'KeyY' || k === 'KeyZ') e.setModalAxis(k.slice(3).toLowerCase() as 'x' | 'y' | 'z')
      else if (k === 'Escape') e.endModal(false)
      else if (k === 'Enter' || k === 'NumpadEnter' || k === 'Space') e.endModal(true)
      else if (k === 'ShiftLeft' || k === 'ShiftRight') e.setPrecision(true)
      else return
      ev.preventDefault()
      return
    }
    let handled = true
    if (mod && k === 'KeyZ') ev.shiftKey ? e.redo() : e.undo()
    else if (mod && k === 'KeyY') e.redo()
    else if (mod && k === 'KeyA') e.selectAll()
    else if (mod && k === 'KeyG') ev.shiftKey ? sel.forEach((id) => e.ungroup(id)) : e.group(sel)
    else if (mod && k === 'KeyD') e.duplicate(sel)
    else if (mod) handled = false
    else if (ev.altKey && k === 'KeyG') e.resetTransform(sel, 'position')
    else if (ev.altKey && k === 'KeyR') e.resetTransform(sel, 'rotation')
    else if (ev.altKey && k === 'KeyS') e.resetTransform(sel, 'scale')
    else if (ev.altKey && k === 'KeyH') e.showAll()
    else if (ev.shiftKey && k === 'KeyD') e.duplicate(sel)
    else if (ev.shiftKey && k === 'Tab') e.setSnap({ enabled: !e.snap.enabled })
    else if (ev.shiftKey && (k === 'KeyX' || k === 'KeyY' || k === 'KeyZ')) {
      const a = k.slice(3).toLowerCase() as 'x' | 'y' | 'z'
      e.setAxisLock({ ...e.axisLock, [a]: !e.axisLock[a] })
    } else if (k === 'KeyG') e.startModal('translate')
    else if (k === 'KeyR') e.startModal('rotate')
    else if (k === 'KeyS') e.startModal('scale')
    else if (k === 'Digit1') e.setGizmoMode('translate')
    else if (k === 'Digit2') e.setGizmoMode('rotate')
    else if (k === 'Digit3') e.setGizmoMode('scale')
    else if (k === 'KeyL') e.setSpace(e.space === 'world' ? 'local' : 'world')
    else if (k === 'KeyX' || k === 'Delete' || k === 'Backspace') e.deleteNodes(sel)
    else if (k === 'KeyH') e.setVisible(sel, false)
    else if (k === 'KeyI') e.isolate(sel[0] ?? null)
    else if (k === 'KeyF') sel.length ? e.frame() : e.setView('iso')
    else if (k === 'KeyC') e.setView('photo')
    else if (k === 'KeyM') e.setRenderMode(RENDER_MODES[(RENDER_MODES.indexOf(e.renderMode) + 1) % RENDER_MODES.length])
    else if (k === 'KeyP') runtime.gestures ? runtime.gestures.setPrecision(!e.precision) : e.setPrecision(!e.precision)
    else if (k === 'Escape') e.isolateId ? e.isolate(null) : e.select([])
    else handled = false
    if (handled) ev.preventDefault()
  }
  const onKeyUp = (ev: KeyboardEvent) => {
    const e = runtime.engine
    if (e?.inModal && (ev.code === 'ShiftLeft' || ev.code === 'ShiftRight')) e.setPrecision(false)
  }
  window.addEventListener('keydown', onKey)
  window.addEventListener('keyup', onKeyUp)
  return () => {
    window.removeEventListener('keydown', onKey)
    window.removeEventListener('keyup', onKeyUp)
  }
}
