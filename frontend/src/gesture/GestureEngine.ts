import { PoseClassifier, DEFAULT_THRESHOLDS, type Pose, type PoseState, type Thresholds } from './classifier'
import { computeFeatures, type HandFeatures, type P3 } from './features'
import { OneEuro, OneEuro2 } from './oneEuro'

export interface RawHand {
  handedness: string // "Left" | "Right" as reported by MediaPipe
  score: number
  image: P3[]
  world: P3[]
}

export type Quat = [number, number, number, number]

export interface TrackedHand {
  id: string
  features: HandFeatures
  pose: PoseState
  cursor: [number, number] // viewport, 0..1, y down — stabilised
  depth: number // filtered apparent hand size (depth proxy)
  rotation: Quat // smoothed hand orientation (three.js camera axes, mirrored)
  pinchAnchor: [number, number] | null // where the pinch began (stabilised cursor)
  pinchStartedAt: number | null
  score: number
}

export type GestureEvent =
  | { type: 'pinchStart'; hand: string; at: [number, number] }
  | { type: 'pinchEnd'; hand: string; at: [number, number]; duration: number; travel: number; lost: boolean }
  | { type: 'tap'; hand: string; at: [number, number] }
  | { type: 'swipe'; hand: string; direction: 'left' | 'right' | 'up' | 'down' }
  | { type: 'poseHold'; hand: string; pose: Pose; ms: number }
  | { type: 'twoHandHold'; pose: Pose }
  | { type: 'handLost'; hand: string }

export interface GestureFrame {
  t: number
  hands: TrackedHand[]
  events: GestureEvent[]
}

export interface EngineOptions {
  thresholds: Thresholds
  /** Portion of the camera frame mapped to the full viewport — users should not need
   *  to reach the edge of the webcam's view to reach the edge of the screen. */
  activeRegion: { x0: number; x1: number; y0: number; y1: number }
  precision: boolean
}

const TAP_MAX_MS = 300
const TAP_MAX_TRAVEL = 0.035
const LOST_MS = 260
const RELEASE_SETTLE_MS = 220
const SWIPE_WINDOW_MS = 160
const SWIPE_MIN_DIST = 0.2
const SWIPE_COOLDOWN_MS = 750
const HOLDS: Partial<Record<Pose, number>> = { victory: 650 }
const TWO_HAND_HOLDS: Partial<Record<Pose, number>> = { open: 600 }

interface HandTrack {
  id: string
  classifier: PoseClassifier
  cursorFilter: OneEuro2
  palmFilter: OneEuro2
  depthFilter: OneEuro
  rot: Quat
  lastSeen: number
  history: { t: number; c: [number, number]; palm: [number, number] }[]
  pinchOnsetPalm: [number, number] | null
  pinchAnchor: [number, number] | null
  pinchStartedAt: number | null
  releaseOffset: [number, number]
  releaseAt: number
  lastSwipe: number
  holdFired: Set<Pose>
  last?: TrackedHand
}

export class GestureEngine {
  opts: EngineOptions = {
    thresholds: { ...DEFAULT_THRESHOLDS },
    activeRegion: { x0: 0.12, x1: 0.88, y0: 0.1, y1: 0.75 },
    precision: false,
  }
  private tracks = new Map<string, HandTrack>()
  private twoHandHoldSince: { pose: Pose; t: number; fired: boolean } | null = null

  setThresholds(t: Thresholds) {
    this.opts.thresholds = t
    for (const tr of this.tracks.values()) tr.classifier.th = t
  }

  setPrecision(p: boolean) {
    this.opts.precision = p
    for (const tr of this.tracks.values()) {
      tr.cursorFilter.set(...this.cursorFilterParams())
      tr.palmFilter.set(...this.cursorFilterParams())
    }
  }

  private cursorFilterParams(): [number, number] {
    return this.opts.precision ? [0.5, 0.8] : [1.4, 2.5]
  }

  private mapToViewport([x, y]: [number, number]): [number, number] {
    const r = this.opts.activeRegion
    const cl = (v: number) => Math.min(1, Math.max(0, v))
    return [cl((x - r.x0) / (r.x1 - r.x0)), cl((y - r.y0) / (r.y1 - r.y0))]
  }

  /** Feed one tracker result. `tMs` is the video frame timestamp in milliseconds. */
  update(raw: RawHand[], tMs: number): GestureFrame {
    const events: GestureEvent[] = []
    const t = tMs / 1000
    const seen = new Set<string>()
    // MediaPipe can label two hands identically for a frame; disambiguate by order.
    const used = new Set<string>()
    for (const h of raw) {
      let id = h.handedness || 'Hand'
      if (used.has(id)) id = id === 'Left' ? 'Right' : 'Left'
      used.add(id)
      seen.add(id)
      let tr = this.tracks.get(id)
      if (!tr) {
        tr = {
          id, classifier: new PoseClassifier(this.opts.thresholds), cursorFilter: new OneEuro2(...this.cursorFilterParams()),
          palmFilter: new OneEuro2(...this.cursorFilterParams()), depthFilter: new OneEuro(0.6, 0.3), rot: [0, 0, 0, 1], lastSeen: tMs, history: [], pinchOnsetPalm: null, pinchAnchor: null,
          pinchStartedAt: null, releaseOffset: [0, 0], releaseAt: -1e9, lastSwipe: -1e9, holdFired: new Set(),
        }
        this.tracks.set(id, tr)
      }
      const dtS = tr.last ? Math.max(0.001, (tMs - tr.lastSeen) / 1000) : 1e3
      tr.lastSeen = tMs
      const f = computeFeatures(h.image, h.world)
      const wasPinching = tr.classifier.current.pinching
      const pose = tr.classifier.update(f, tMs)
      const filtered = tr.cursorFilter.filter(...this.mapToViewport(f.cursor), t)
      const palm = tr.palmFilter.filter(...this.mapToViewport(f.palmCenter), t)
      tr.history.push({ t: tMs, c: filtered, palm: f.palmCenter })
      while (tr.history.length && tMs - tr.history[0].t > 400) tr.history.shift()

      // --- pinch onset/release with click-slip compensation. Closing the fingers moves the
      // thumb–index midpoint, so while pinched the cursor is anchored where it was *just
      // before* the fingers started closing and then follows the palm, which stays still.
      let cursor: [number, number] = filtered
      if (pose.pinching && !wasPinching) {
        const before = tr.history.find((s) => tMs - s.t <= 110) ?? tr.history[0]
        tr.pinchAnchor = before.c
        tr.pinchOnsetPalm = palm
        tr.pinchStartedAt = tMs
        events.push({ type: 'pinchStart', hand: id, at: before.c })
      }
      if (pose.pinching && tr.pinchAnchor && tr.pinchOnsetPalm) {
        cursor = [tr.pinchAnchor[0] + palm[0] - tr.pinchOnsetPalm[0], tr.pinchAnchor[1] + palm[1] - tr.pinchOnsetPalm[1]]
      }
      if (!pose.pinching && wasPinching) {
        const prev = tr.last?.cursor ?? filtered
        const duration = tMs - (tr.pinchStartedAt ?? tMs)
        const travel = Math.hypot(prev[0] - (tr.pinchAnchor?.[0] ?? prev[0]), prev[1] - (tr.pinchAnchor?.[1] ?? prev[1]))
        events.push({ type: 'pinchEnd', hand: id, at: prev, duration, travel, lost: false })
        if (duration <= TAP_MAX_MS && travel <= TAP_MAX_TRAVEL) events.push({ type: 'tap', hand: id, at: tr.pinchAnchor ?? prev })
        tr.releaseOffset = [prev[0] - filtered[0], prev[1] - filtered[1]]
        tr.releaseAt = tMs
        tr.pinchAnchor = tr.pinchOnsetPalm = null
        tr.pinchStartedAt = null
      }
      if (!pose.pinching) {
        // fingers opening also drags the midpoint; ease that offset out
        const k = Math.max(0, 1 - (tMs - tr.releaseAt) / RELEASE_SETTLE_MS)
        cursor = [filtered[0] + tr.releaseOffset[0] * k, filtered[1] + tr.releaseOffset[1] * k]
      }

      // --- orientation smoothing (exponential slerp, frame-rate independent)
      const q = basisToQuat(f.basis)
      tr.rot = slerp(tr.rot, alignSign(tr.rot, q), 1 - Math.exp(-dtS / (this.opts.precision ? 0.1 : 0.04)))

      // --- swipe: fast open-palm flick
      if (pose.pose === 'open' && tMs - tr.lastSwipe > SWIPE_COOLDOWN_MS) {
        const old = tr.history.find((s) => tMs - s.t <= SWIPE_WINDOW_MS)
        if (old) {
          const dx = f.palmCenter[0] - old.palm[0]
          const dy = f.palmCenter[1] - old.palm[1]
          if (Math.abs(dx) > SWIPE_MIN_DIST && Math.abs(dy) < 0.6 * Math.abs(dx)) {
            events.push({ type: 'swipe', hand: id, direction: dx > 0 ? 'right' : 'left' })
            tr.lastSwipe = tMs
          } else if (Math.abs(dy) > SWIPE_MIN_DIST && Math.abs(dx) < 0.6 * Math.abs(dy)) {
            events.push({ type: 'swipe', hand: id, direction: dy > 0 ? 'down' : 'up' })
            tr.lastSwipe = tMs
          }
        }
      }

      // --- pose holds
      const holdMs = HOLDS[pose.pose]
      if (holdMs && tMs - pose.since >= holdMs && !tr.holdFired.has(pose.pose)) {
        tr.holdFired.add(pose.pose)
        events.push({ type: 'poseHold', hand: id, pose: pose.pose, ms: holdMs })
      }
      for (const p of [...tr.holdFired]) if (p !== pose.pose) tr.holdFired.delete(p)

      tr.last = {
        id, features: f, pose, cursor: [Math.min(1, Math.max(0, cursor[0])), Math.min(1, Math.max(0, cursor[1]))],
        depth: tr.depthFilter.filter(f.apparentSize, t), rotation: tr.rot, pinchAnchor: tr.pinchAnchor,
        pinchStartedAt: tr.pinchStartedAt, score: h.score,
      }
    }

    // --- lost hands: brief dropouts keep state; longer ones end any pinch
    for (const [id, tr] of this.tracks) {
      if (seen.has(id)) continue
      if (tMs - tr.lastSeen > LOST_MS) {
        if (tr.classifier.current.pinching && tr.last) {
          events.push({ type: 'pinchEnd', hand: id, at: tr.last.cursor, duration: tMs - (tr.pinchStartedAt ?? tMs), travel: 0, lost: true })
        }
        events.push({ type: 'handLost', hand: id })
        this.tracks.delete(id)
      }
    }

    const hands = [...this.tracks.values()].filter((tr) => tr.last).map((tr) => tr.last!)
    // --- two-hand holds (e.g. both palms open = cancel / undo)
    if (hands.length === 2 && hands[0].pose.pose === hands[1].pose.pose && TWO_HAND_HOLDS[hands[0].pose.pose]) {
      const p = hands[0].pose.pose
      if (!this.twoHandHoldSince || this.twoHandHoldSince.pose !== p) this.twoHandHoldSince = { pose: p, t: tMs, fired: false }
      else if (!this.twoHandHoldSince.fired && tMs - this.twoHandHoldSince.t >= TWO_HAND_HOLDS[p]!) {
        this.twoHandHoldSince.fired = true
        events.push({ type: 'twoHandHold', pose: p })
      }
    } else this.twoHandHoldSince = null

    return { t: tMs, hands, events }
  }

  reset() {
    this.tracks.clear()
    this.twoHandHoldSince = null
  }
}

// ---------------------------------------------------------------- quaternion helpers
export function basisToQuat([x, y, z]: [P3, P3, P3]): Quat {
  // rotation matrix with columns x, y, z
  const m00 = x.x, m01 = y.x, m02 = z.x
  const m10 = x.y, m11 = y.y, m12 = z.y
  const m20 = x.z, m21 = y.z, m22 = z.z
  const tr = m00 + m11 + m22
  let qx, qy, qz, qw
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2
    qw = 0.25 * s; qx = (m21 - m12) / s; qy = (m02 - m20) / s; qz = (m10 - m01) / s
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2
    qw = (m21 - m12) / s; qx = 0.25 * s; qy = (m01 + m10) / s; qz = (m02 + m20) / s
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2
    qw = (m02 - m20) / s; qx = (m01 + m10) / s; qy = 0.25 * s; qz = (m12 + m21) / s
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2
    qw = (m10 - m01) / s; qx = (m02 + m20) / s; qy = (m12 + m21) / s; qz = 0.25 * s
  }
  const n = Math.hypot(qx, qy, qz, qw) || 1
  return [qx / n, qy / n, qz / n, qw / n]
}

function alignSign(ref: Quat, q: Quat): Quat {
  const d = ref[0] * q[0] + ref[1] * q[1] + ref[2] * q[2] + ref[3] * q[3]
  return d < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q
}

export function slerp(a: Quat, b: Quat, t: number): Quat {
  let cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
  if (cos > 0.9995) {
    const r: Quat = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t]
    const n = Math.hypot(...r) || 1
    return [r[0] / n, r[1] / n, r[2] / n, r[3] / n]
  }
  cos = Math.min(1, Math.max(-1, cos))
  const th = Math.acos(cos)
  const s = Math.sin(th)
  const wa = Math.sin((1 - t) * th) / s
  const wb = Math.sin(t * th) / s
  return [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb]
}
