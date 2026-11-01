import { useEffect, useState } from 'react'
import { useStudio } from '../store/studio'
import { startGestures } from './GesturePanel'
import { BrandMark, Icon } from './Icon'

const KEYS: [string, [string, string][]][] = [
  ['Look around', [['Drag', 'orbit'], ['Right-drag', 'pan'], ['Scroll', 'zoom']]],
  ['Select', [['Click', 'one'], ['⇧ Click', 'add'], ['⌘ Drag', 'box'], ['⌥ Click', 'inside a group']]],
  ['Transform', [['G', 'move'], ['R', 'rotate'], ['S', 'scale'], ['X Y Z', 'lock to an axis']]],
  ['Handles', [['1 2 3', 'move, rotate, scale'], ['L', 'local or world'], ['⇧ Tab', 'snapping']]],
  ['Edit', [['⇧ D', 'duplicate'], ['X', 'delete'], ['H', 'hide'], ['⌥ H', 'show all'], ['I', 'isolate']]],
  ['Group', [['⌘ G', 'group'], ['⇧ ⌘ G', 'ungroup'], ['⌥ G R S', 'clear move, rotation, scale']]],
  ['View', [['F', 'frame'], ['C', 'photo viewpoint'], ['M', 'shading'], ['P', 'precision'], ['⌘ Z', 'undo']]],
]

const PRACTICE = [
  { id: 'point', title: 'Point', how: 'Index finger out, others curled', does: 'Aims the cursor and highlights what is under it' },
  { id: 'pinch', title: 'Pinch', how: 'Touch thumb and index tips', does: 'Tap to select; hold and move to drag the object, or to orbit over empty space' },
  { id: 'open', title: 'Open palm', how: 'All fingers spread, hold still briefly', does: 'Navigate: move to orbit, closer/further to zoom, flick to switch view' },
  { id: 'fist', title: 'Fist', how: 'Close the hand', does: 'Twist the selection with your wrist (or the whole scene)' },
  { id: 'two', title: 'Two-hand pinch', how: 'Pinch with both hands', does: 'Spread to scale / zoom, turn like a wheel to rotate' },
  { id: 'victory', title: 'Two fingers', how: 'Index and middle finger out, hold for about half a second', does: 'Switches precision mode on or off (slower, finer control)' },
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
        <button className="icon-btn close" onClick={close} aria-label="Close"><Icon name="close" /></button>
        <p className="ob-step tnum">Step {step + 1} of 3</p>
        {step === 0 && (
          <div className="ob-body">
            <div className="ob-brand"><BrandMark size={30} /><div><b>VERONICA</b><span>The spatial intelligence system</span></div></div>
            <h2>A photo of a room, as a room you can rearrange</h2>
            <p>VERONICA measures a photograph in metres and turns every object in it into a separate 3D object. Select, move, turn, scale, copy, hide or isolate any of them, with a mouse or with your hands in the air.</p>
            <p className="dim">A photo only shows one side of each object, and VERONICA never hides that. Surfaces <b className="obs-t">seen in a photo</b> come straight from it; surfaces it had to <b className="inf-t">estimate</b>, such as the backs of things, are hatched in blue, the way a draftsman draws hidden edges.</p>
            <div className="ob-nav"><span /><button className="primary" onClick={() => setStep(1)}>Next</button></div>
          </div>
        )}
        {step === 1 && (
          <div className="ob-body">
            <h2>Mouse, trackpad and keyboard</h2>
            <dl className="keys">
              {KEYS.map(([what, how]) => (
                <div key={what} style={{ display: 'contents' }}>
                  <dt>{what}</dt>
                  <dd>{how.map(([k, does]) => <span key={k}><kbd>{k}</kbd> {does}</span>)}</dd>
                </div>
              ))}
            </dl>
            <div className="ob-nav"><button className="ghost" onClick={() => setStep(0)}>Back</button><button className="primary" onClick={() => setStep(2)}>Next</button></div>
          </div>
        )}
        {step === 2 && (
          <div className="ob-body">
            <h2>Hand control</h2>
            {!on ? (
              <>
                <p>Turn on your webcam to try them. Your hands are tracked inside this browser; no video leaves your computer.</p>
                <div><button className="primary" onClick={() => startGestures()}>Turn on the camera</button></div>
              </>
            ) : (
              <p className="dim">Try each gesture; each one ticks off when VERONICA recognises it. Keep your hand about 40 to 80 cm from the camera.</p>
            )}
            <ul className="practice">
              {PRACTICE.map((g) => (
                <li key={g.id} className={`pg ${done.has(g.id) ? 'ok' : ''}`}>
                  <span className="tick">{done.has(g.id) && <Icon name="check" size={12} />}</span>
                  <div><b>{g.title}</b><br /><span>{g.how}.</span><br />{g.does}.</div>
                </li>
              ))}
            </ul>
            <p className="dim">Hold both palms open for about half a second to cancel a gesture or undo the last edit. If pinches fire too early or too late, use Calibrate pinch in the hand panel.</p>
            <div className="ob-nav"><button className="ghost" onClick={() => setStep(1)}>Back</button><button className="primary" onClick={close}>Start</button></div>
          </div>
        )}
      </div>
    </div>
  )
}
