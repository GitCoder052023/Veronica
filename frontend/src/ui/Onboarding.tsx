import { useEffect, useState } from 'react'
import { useStudio } from '../store/studio'
import { startGestures } from './GesturePanel'
import { BrandMark, Icon } from './Icon'

const PRACTICE = [
  { id: 'point', title: 'Point', how: 'Index finger out, others curled', does: 'Aims the cursor and highlights what is under it' },
  { id: 'pinch', title: 'Pinch', how: 'Touch thumb and index tips', does: 'Tap = select · hold & move = drag the object (or orbit empty space)' },
  { id: 'open', title: 'Open palm', how: 'All fingers spread, hold still briefly', does: 'Navigate: move to orbit, closer/further to zoom, flick to switch view' },
  { id: 'fist', title: 'Fist', how: 'Close the hand', does: 'Twist the selection with your wrist (or the whole scene)' },
  { id: 'two', title: 'Two-hand pinch', how: 'Pinch with both hands', does: 'Spread to scale / zoom, turn like a wheel to rotate' },
  { id: 'victory', title: 'Victory ✌', how: 'Index + middle out, hold ~0.6 s', does: 'Toggles precision mode (slower, finer control)' },
] as const

export function Onboarding() {
  const set = useStudio((s) => s.set)
  const on = useStudio((s) => s.gestureOn)
  const hud = useStudio((s) => s.gestureHud)
  const [step, setStep] = useState(0)
  const [done, setDone] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!hud) return
    const seen = new Set(done)
    for (const h of hud.hands) seen.add(h.pose)
    if (hud.hands.length === 2 && hud.hands.every((h) => h.pose === 'pinch')) seen.add('two')
    if (seen.size !== done.size) setDone(seen)
  }, [hud]) // eslint-disable-line react-hooks/exhaustive-deps

  const close = () => {
    localStorage.setItem('veronica.onboarded', '1')
    set({ onboarding: false })
  }

  return (
    <div className="overlay" onClick={close}>
      <div className="onboard" onClick={(e) => e.stopPropagation()}>
        <button className="icon-btn close" onClick={close}><Icon name="close" /></button>
        <div className="ob-steps">{['Welcome', 'Mouse & keys', 'Hands'].map((s, i) => <span key={s} className={i === step ? 'on' : i < step ? 'done' : ''}>{s}</span>)}</div>
        {step === 0 && (
          <div className="ob-body">
            <div className="ob-brand"><BrandMark size={28} /><div><b>VERONICA</b><span>The Spatial Intelligence System</span></div></div>
            <h2>Reconstructed reality, in your hands</h2>
            <p>VERONICA turns a photograph into a metric 3D scene made of separate objects. Every object can be selected, moved, rotated, scaled, duplicated, hidden or isolated, with a mouse or with your hands in the air.</p>
            <p className="dim">A single photo only shows one side of each object. VERONICA keeps that distinction visible: <b className="obs-t">observed</b> surfaces come straight from the photo, while <b className="inf-t">inferred</b> surfaces (hidden sides, background behind objects) are estimated and drawn with a hatch.</p>
            <div className="ob-nav"><span /><button className="primary" onClick={() => setStep(1)}>Next</button></div>
          </div>
        )}
        {step === 1 && (
          <div className="ob-body">
            <h2>Mouse, trackpad & keyboard</h2>
            <div className="keys">
              <div><kbd>drag</kbd> orbit · <kbd>right-drag</kbd>/<kbd>⇧ drag</kbd> pan · <kbd>scroll</kbd> zoom</div>
              <div><kbd>click</kbd> select · <kbd>⇧ click</kbd> add · <kbd>⌘ drag</kbd> box select · <kbd>⌥ click</kbd> pick inside group</div>
              <div><kbd>G</kbd> grab · <kbd>R</kbd> rotate · <kbd>S</kbd> scale, then <kbd>X</kbd><kbd>Y</kbd><kbd>Z</kbd> to constrain</div>
              <div><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> gizmo move / rotate / scale · <kbd>L</kbd> local/world · <kbd>⇧ Tab</kbd> snapping</div>
              <div><kbd>⇧ D</kbd> duplicate · <kbd>X</kbd>/<kbd>Del</kbd> delete · <kbd>H</kbd> hide · <kbd>⌥ H</kbd> show all · <kbd>I</kbd> isolate</div>
              <div><kbd>⌘ G</kbd> group · <kbd>⇧⌘ G</kbd> ungroup · <kbd>⌥ G</kbd>/<kbd>⌥ R</kbd>/<kbd>⌥ S</kbd> clear move/rot/scale</div>
              <div><kbd>F</kbd> frame · <kbd>C</kbd> photo viewpoint · <kbd>M</kbd> render mode · <kbd>P</kbd> precision · <kbd>⌘ Z</kbd> undo</div>
            </div>
            <div className="ob-nav"><button className="ghost" onClick={() => setStep(0)}>Back</button><button className="primary" onClick={() => setStep(2)}>Next</button></div>
          </div>
        )}
        {step === 2 && (
          <div className="ob-body">
            <h2>Hand control</h2>
            {!on ? (
              <p>Turn on your webcam. Hand landmarks are computed locally in this browser; no video leaves your machine. <button className="primary" onClick={() => startGestures()}>Enable camera</button></p>
            ) : (
              <p className="dim">Try each gesture. They tick off as VERONICA recognises them. Keep your hand roughly 40–80 cm from the camera.</p>
            )}
            <div className="practice">
              {PRACTICE.map((g) => (
                <div key={g.id} className={`pg ${done.has(g.id) ? 'ok' : ''}`}>
                  <span className="check">{done.has(g.id) ? '✓' : ''}</span>
                  <div><b>{g.title}</b> <span className="dim">{g.how}</span><br /><span>{g.does}</span></div>
                </div>
              ))}
            </div>
            <p className="dim small">Both palms open for ~0.6 s cancels the current gesture or undoes the last edit. Use “Calibrate pinch” in the hand panel if pinches fire too early or too late.</p>
            <div className="ob-nav"><button className="ghost" onClick={() => setStep(1)}>Back</button><button className="primary" onClick={close}>Start</button></div>
          </div>
        )}
      </div>
    </div>
  )
}
