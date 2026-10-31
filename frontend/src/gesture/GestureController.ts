import * as THREE from 'three'
import { VeronicaEngine, type ViewPreset } from '../engine/VeronicaEngine'
import type { Pose } from './classifier'
import { GestureEngine, type GestureEvent, type GestureFrame, type RawHand, type TrackedHand } from './GestureEngine'

/** What the hands control.
 *  auto     — contextual: pinch an object to move it, pinch empty space to orbit,
 *             fist twists the selection (or the scene), two-hand pinch scales/rotates
 *  navigate — hands only move the camera
 *  object   — hands only act on the selection
 *  scene    — hands transform the whole reconstructed scene */
export type GestureMode = 'auto' | 'navigate' | 'object' | 'scene'

export type ActiveOp =
  | { kind: 'drag-object'; hand: string; plane: THREE.Plane; hit0: THREE.Vector3; depth0: number; dist0: number }
  | { kind: 'orbit'; hand: string; last: [number, number] }
  | { kind: 'rotate-object'; hand: string; q0: THREE.Quaternion; pivot: THREE.Vector3 }
  | { kind: 'rotate-scene'; hand: string; q0: THREE.Quaternion; pivot: THREE.Vector3 }
  | { kind: 'navigate'; hand: string; last: [number, number]; depth0: number; lastDepthApplied: number }
  | { kind: 'two-hand'; target: 'object' | 'scene' | 'camera'; d0: number; a0: number; m0: [number, number]; pivot: THREE.Vector3; plane: THREE.Plane; hit0: THREE.Vector3; lastD: number; lastA: number; lastM: [number, number] }

export interface HandHud {
  id: string
  cursor: [number, number]
  pose: Pose
  pinchStrength: number
  pinchRatio: number
  confidence: number
}

export interface GestureHud {
  hands: HandHud[]
  op: ActiveOp['kind'] | null
  label: string
  hint: string
  precision: boolean
  lastEvent: string | null
}

const VIEW_CYCLE: ViewPreset[] = ['photo', 'iso', 'front', 'right', 'back', 'left', 'top']
const NAV_DEADZONE = 0.004

export class GestureController {
  readonly gestures = new GestureEngine()
  mode: GestureMode = 'auto'
  enabled = true
  private op: ActiveOp | null = null
  private viewIndex = 0
  private lastEvent: { text: string; t: number } | null = null
  private hoverFrame = 0
  onHud: ((h: GestureHud) => void) | null = null
  onPrecision: ((p: boolean) => void) | null = null

  constructor(private engine: VeronicaEngine) {}

  setMode(m: GestureMode) {
    this.cancelOp()
    this.mode = m
  }

  setPrecision(p: boolean) {
    this.gestures.setPrecision(p)
    this.engine.setPrecision(p)
    this.onPrecision?.(p)
  }

  get precision() {
    return this.gestures.opts.precision
  }

  private gain(base: number) {
    return this.precision ? base * 0.3 : base
  }

  /** Called with every new tracker result. */
  process(raw: RawHand[], tMs: number) {
    const frame = this.gestures.update(raw, tMs)
    if (!this.enabled || !this.engine.manifest) {
      this.emitHud(frame)
      return
    }
    for (const e of frame.events) this.handleEvent(e, frame)
    this.continuous(frame)
    this.emitHud(frame)
  }

  // ------------------------------------------------------------------ discrete events
  private handleEvent(e: GestureEvent, frame: GestureFrame) {
    const hand = (id: string) => frame.hands.find((h) => h.id === id)
    switch (e.type) {
      case 'pinchStart': {
        const other = frame.hands.find((h) => h.id !== e.hand && h.pose.pinching)
        if (other) {
          this.endOp(true)
          this.startTwoHand(frame)
        } else if (!this.op) this.startPinchOp(e.hand, e.at, hand(e.hand))
        break
      }
      case 'pinchEnd':
        if (this.op && (this.op.kind === 'two-hand' || ('hand' in this.op && this.op.hand === e.hand))) this.endOp(true)
        break
      case 'tap': {
        if (this.mode === 'navigate' || this.mode === 'scene') break
        const h = this.engine.pick(this.toNdc(e.at))
        if (h) {
          const id = this.engine.selectable(h.id)
          this.engine.select([id])
          this.flash(`Selected ${this.engine.nodes.get(id)?.name ?? id}`)
        } else {
          this.engine.select([])
          this.flash('Selection cleared')
        }
        break
      }
      case 'swipe':
        if (this.op && this.op.kind !== 'navigate') break
        if (e.direction === 'left' || e.direction === 'right') {
          this.cancelOp()
          this.viewIndex = (this.viewIndex + (e.direction === 'right' ? 1 : VIEW_CYCLE.length - 1)) % VIEW_CYCLE.length
          this.engine.setView(VIEW_CYCLE[this.viewIndex])
          this.flash(`View: ${VIEW_CYCLE[this.viewIndex]}`)
        }
        break
      case 'poseHold':
        if (e.pose === 'victory') {
          this.setPrecision(!this.precision)
          this.flash(this.precision ? 'Precision mode on' : 'Precision mode off')
        }
        break
      case 'twoHandHold':
        if (e.pose === 'open') {
          if (this.op) {
            this.cancelOp()
            this.flash('Operation cancelled')
          } else {
            this.engine.undo()
            this.flash('Undo')
          }
        }
        break
      case 'handLost':
        if (this.op && 'hand' in this.op && this.op.hand === e.hand) this.endOp(true)
        if (this.op?.kind === 'two-hand') this.endOp(true)
        break
    }
  }

  private startPinchOp(handId: string, at: [number, number], h?: TrackedHand) {
    const ndc = this.toNdc(at)
    const hit = this.mode === 'navigate' || this.mode === 'scene' ? null : this.engine.pick(ndc)
    const target = hit ? this.engine.selectable(hit.id) : null
    const node = target ? this.engine.nodes.get(target) : null
    if (node && !node.locked) {
      if (!this.engine.selection.includes(target!)) this.engine.select([target!])
      if (this.engine.beginManipulation(this.engine.selection, 'move (hand)')) {
        const n = this.engine.camera.getWorldDirection(new THREE.Vector3())
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, hit!.point)
        this.op = { kind: 'drag-object', hand: handId, plane, hit0: hit!.point.clone(), depth0: h?.depth ?? 0.1, dist0: hit!.distance }
        return
      }
    }
    if (this.mode === 'object') return
    if (this.mode === 'scene') {
      // pinch-drag in scene mode turns the whole scene like a turntable
      this.engine.beginSceneManipulation()
      this.op = { kind: 'rotate-scene', hand: handId, q0: new THREE.Quaternion(), pivot: this.engine.sceneCenter() }
      ;(this.op as ActiveOp & { turntable?: [number, number] }).turntable = at
      return
    }
    this.op = { kind: 'orbit', hand: handId, last: at }
  }

  private startTwoHand(frame: GestureFrame) {
    const [a, b] = frame.hands.filter((h) => h.pose.pinching)
    if (!a || !b) return
    const { d, ang, mid } = twoHandGeom(a.cursor, b.cursor, this.engine.camera.aspect)
    const sel = this.engine.manipulable()
    let target: 'object' | 'scene' | 'camera' = 'camera'
    if (this.mode === 'scene') target = 'scene'
    else if (this.mode !== 'navigate' && sel.length) target = 'object'
    let pivot = this.engine.sceneCenter()
    if (target === 'object') {
      if (!this.engine.beginManipulation(sel, 'scale/rotate (two hands)')) target = 'camera'
      else pivot = this.engine.manipulationPivot.clone()
    }
    if (target === 'scene') this.engine.beginSceneManipulation()
    const n = this.engine.camera.getWorldDirection(new THREE.Vector3())
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, pivot)
    const hit0 = this.engine.rayFrom(this.toNdc(mid)).intersectPlane(plane, new THREE.Vector3()) ?? pivot.clone()
    this.op = { kind: 'two-hand', target, d0: d, a0: ang, m0: mid, pivot, plane, hit0, lastD: d, lastA: ang, lastM: mid }
  }

  // ------------------------------------------------------------------ continuous control
  private continuous(frame: GestureFrame) {
    const op = this.op
    const hand = op && 'hand' in op ? frame.hands.find((h) => h.id === op.hand) : undefined

    // Start pose-driven ops (fist twist, open-palm navigation) when idle.
    if (!op) {
      for (const h of frame.hands) {
        const held = performance.now() - h.pose.since
        if (h.pose.pose === 'fist' && held > 120) {
          this.startFist(h)
          break
        }
        if (h.pose.pose === 'open' && held > 250 && this.mode !== 'object' && frame.hands.length === 1) {
          this.op = { kind: 'navigate', hand: h.id, last: h.features.palmCenter, depth0: h.depth, lastDepthApplied: h.depth }
          break
        }
      }
    }

    const cur = this.op
    if (!cur) {
      this.hover(frame)
      return
    }
    switch (cur.kind) {
      case 'drag-object': {
        if (!hand) return
        const ray = this.engine.rayFrom(this.toNdc(hand.cursor))
        const hit = ray.intersectPlane(cur.plane, new THREE.Vector3())
        if (!hit) return
        const d = hit.sub(cur.hit0).multiplyScalar(this.precision ? 0.3 : 1)
        // push/pull: hand apparent size grows when it moves towards the camera
        let ratio = hand.depth / cur.depth0 - 1
        ratio = Math.abs(ratio) < 0.07 ? 0 : ratio - Math.sign(ratio) * 0.07
        const fwd = this.engine.camera.getWorldDirection(new THREE.Vector3())
        d.add(fwd.multiplyScalar(-ratio * cur.dist0 * this.gain(1.6)))
        this.engine.applyManipulation(new THREE.Matrix4().makeTranslation(d.x, d.y, d.z))
        break
      }
      case 'orbit': {
        if (!hand) return
        const dx = hand.cursor[0] - cur.last[0]
        const dy = hand.cursor[1] - cur.last[1]
        cur.last = hand.cursor
        this.engine.orbitBy(dx * Math.PI * this.gain(1.6), dy * Math.PI * this.gain(1.0))
        break
      }
      case 'rotate-object':
      case 'rotate-scene': {
        if (!hand) return
        const turntable = (cur as ActiveOp & { turntable?: [number, number] }).turntable
        let delta: THREE.Matrix4
        if (turntable) {
          // pinch-drag scene rotation: horizontal → yaw about world up, vertical → pitch about camera right
          const yaw = (hand.cursor[0] - turntable[0]) * Math.PI * this.gain(1.8)
          const pitch = (hand.cursor[1] - turntable[1]) * Math.PI * this.gain(1.0)
          const right = new THREE.Vector3().setFromMatrixColumn(this.engine.camera.matrixWorld, 0)
          const R = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(0, 1, 0), yaw).premultiply(new THREE.Matrix4().makeRotationAxis(right, pitch))
          delta = VeronicaEngine.aboutPoint(cur.pivot, R)
        } else {
          // the fist is opening: freeze so the release motion does not twist the object back
          if (hand.pose.candidate !== 'fist') {
            if (hand.pose.pose !== 'fist') this.endOp(true)
            break
          }
          const qHand = new THREE.Quaternion(...hand.rotation)
          const dq = qHand.clone().multiply(cur.q0.clone().invert())
          const qc = this.engine.camera.quaternion
          const dqW = qc.clone().multiply(dq).multiply(qc.clone().invert())
          const g = new THREE.Quaternion().slerp(dqW, this.precision ? 0.35 : 1)
          delta = VeronicaEngine.aboutPoint(cur.pivot, new THREE.Matrix4().makeRotationFromQuaternion(g))
        }
        if (cur.kind === 'rotate-object') this.engine.applyManipulation(delta)
        else this.engine.applySceneManipulation(delta)
        if (!turntable && hand.pose.pose !== 'fist') this.endOp(true)
        break
      }
      case 'navigate': {
        if (!hand || hand.pose.pose !== 'open') {
          this.op = null
          return
        }
        const p = hand.features.palmCenter
        const dx = p[0] - cur.last[0]
        const dy = p[1] - cur.last[1]
        cur.last = p
        // fast flicks are swipes, not orbits
        if (Math.hypot(dx, dy) < 0.05 && Math.hypot(dx, dy) > NAV_DEADZONE) this.engine.orbitBy(dx * Math.PI * this.gain(1.2), dy * Math.PI * this.gain(0.8))
        const rel = hand.depth / cur.lastDepthApplied
        if (Math.abs(rel - 1) > 0.04) {
          this.engine.dollyBy(Math.pow(rel, -this.gain(1.5)))
          cur.lastDepthApplied = hand.depth
        }
        break
      }
      case 'two-hand': {
        const pins = frame.hands.filter((h) => h.pose.pinching)
        if (pins.length < 2) return
        const { d, ang, mid } = twoHandGeom(pins[0].cursor, pins[1].cursor, this.engine.camera.aspect)
        let s = d / cur.d0
        let a = ang - cur.a0
        if (this.precision) {
          s = 1 + (s - 1) * 0.3
          a *= 0.3
        }
        if (cur.target === 'camera') {
          this.engine.dollyBy(cur.lastD / d)
          this.engine.orbitBy(-(ang - cur.lastA) * this.gain(1), 0)
          this.engine.panBy(-(mid[0] - cur.lastM[0]) * this.gain(1), (mid[1] - cur.lastM[1]) * this.gain(1))
          cur.lastD = d
          cur.lastA = ang
          cur.lastM = mid
          break
        }
        const viewAxis = this.engine.camera.getWorldDirection(new THREE.Vector3()).negate()
        const hit = this.engine.rayFrom(this.toNdc(mid)).intersectPlane(cur.plane, new THREE.Vector3())
        const tr = hit ? hit.sub(cur.hit0).multiplyScalar(this.precision ? 0.3 : 1) : new THREE.Vector3()
        if (cur.target === 'object') {
          const RS = new THREE.Matrix4().makeRotationAxis(viewAxis, -a).multiply(new THREE.Matrix4().makeScale(s, s, s))
          const delta = new THREE.Matrix4().makeTranslation(tr.x, tr.y, tr.z).multiply(VeronicaEngine.aboutPoint(cur.pivot, RS))
          this.engine.applyManipulation(delta)
        } else {
          const RS = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(0, 1, 0), -a).multiply(new THREE.Matrix4().makeScale(s, s, s))
          this.engine.applySceneManipulation(VeronicaEngine.aboutPoint(cur.pivot, RS))
        }
        break
      }
    }
  }

  private startFist(h: TrackedHand) {
    if (this.mode === 'navigate') return
    const sel = this.engine.manipulable()
    const q0 = new THREE.Quaternion(...h.rotation)
    if (sel.length && this.mode !== 'scene') {
      if (this.engine.beginManipulation(sel, 'rotate (hand)')) this.op = { kind: 'rotate-object', hand: h.id, q0, pivot: this.engine.manipulationPivot.clone() }
    } else if (this.mode !== 'object') {
      this.engine.beginSceneManipulation()
      this.op = { kind: 'rotate-scene', hand: h.id, q0, pivot: this.engine.sceneCenter() }
    }
  }

  private hover(frame: GestureFrame) {
    if (++this.hoverFrame % 2) return
    const h = frame.hands.find((x) => x.pose.pose === 'point' || x.pose.pose === 'pinch' || x.pose.pose === 'none') ?? frame.hands[0]
    if (!h) return
    const hit = this.engine.pick(this.toNdc(h.cursor))
    this.engine.setHovered(hit ? this.engine.selectable(hit.id) : null)
  }

  endOp(commit: boolean) {
    const op = this.op
    this.op = null
    if (!op) return
    if (op.kind === 'drag-object' || op.kind === 'rotate-object' || (op.kind === 'two-hand' && op.target === 'object')) this.engine.endManipulation(commit)
    if (op.kind === 'rotate-scene' || (op.kind === 'two-hand' && op.target === 'scene')) this.engine.endSceneManipulation(commit)
  }

  cancelOp() {
    this.endOp(false)
  }

  private toNdc([x, y]: [number, number]) {
    return new THREE.Vector2(x * 2 - 1, -(y * 2 - 1))
  }

  private flash(text: string) {
    this.lastEvent = { text, t: performance.now() }
  }

  private emitHud(frame: GestureFrame) {
    if (!this.onHud) return
    const op = this.op
    this.onHud({
      hands: frame.hands.map((h) => ({ id: h.id, cursor: h.cursor, pose: h.pose.pose, pinchStrength: h.pose.pinchStrength, pinchRatio: h.features.pinchRatio, confidence: h.pose.confidence })),
      op: op?.kind ?? null,
      label: opLabel(op),
      hint: hintFor(op, frame, this.engine.selection.length > 0, this.mode),
      precision: this.precision,
      lastEvent: this.lastEvent && performance.now() - this.lastEvent.t < 1600 ? this.lastEvent.text : null,
    })
  }
}

function twoHandGeom(a: [number, number], b: [number, number], aspect: number) {
  const dx = (b[0] - a[0]) * aspect
  const dy = b[1] - a[1]
  // order-independent angle: always measure from the left-most hand
  const sgn = dx >= 0 ? 1 : -1
  return { d: Math.max(0.02, Math.hypot(dx, dy)), ang: Math.atan2(dy * sgn, dx * sgn), mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] as [number, number] }
}

function opLabel(op: ActiveOp | null): string {
  if (!op) return 'Idle'
  switch (op.kind) {
    case 'drag-object': return 'Moving selection'
    case 'orbit': return 'Orbiting camera'
    case 'rotate-object': return 'Twisting selection'
    case 'rotate-scene': return 'Rotating scene'
    case 'navigate': return 'Navigating'
    case 'two-hand': return op.target === 'object' ? 'Scale / rotate selection' : op.target === 'scene' ? 'Scale / rotate scene' : 'Zoom / orbit'
  }
}

function hintFor(op: ActiveOp | null, frame: GestureFrame, hasSel: boolean, mode: GestureMode): string {
  if (!frame.hands.length) return 'Raise a hand into the camera view'
  if (op?.kind === 'drag-object') return 'Move hand to drag · push / pull for depth · release to drop'
  if (op?.kind === 'rotate-object' || op?.kind === 'rotate-scene') return 'Turn your wrist · open hand to finish'
  if (op?.kind === 'two-hand') return 'Spread / pinch hands to scale · turn like a wheel to rotate'
  if (op?.kind === 'navigate') return 'Move open palm to orbit · closer / further to zoom · flick to change view'
  if (op?.kind === 'orbit') return 'Drag to orbit · release to stop'
  if (mode === 'navigate') return 'Pinch-drag or open palm to orbit · two-hand pinch to zoom'
  if (mode === 'scene') return 'Pinch-drag to turn the scene · fist to twist · two hands to scale'
  return hasSel
    ? 'Pinch object to move · fist to twist · two-hand pinch to scale · ✌ precision · two open palms = undo'
    : 'Point to aim · pinch to select · pinch empty space to orbit · open palm to navigate'
}
