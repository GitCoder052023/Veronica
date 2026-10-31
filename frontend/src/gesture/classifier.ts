import type { HandFeatures } from './features'

/** Static hand poses the engine distinguishes. Chosen for being (a) easy to hold,
 *  (b) far apart in feature space so they rarely get confused, (c) reachable from
 *  each other without passing through a third pose that would fire by accident. */
export type Pose = 'none' | 'open' | 'point' | 'pinch' | 'fist' | 'victory'

export interface Thresholds {
  pinchOn: number // pinchRatio below → pinch starts
  pinchOff: number // pinchRatio above → pinch ends (hysteresis gap avoids chatter)
  extendOn: number // finger extension ratio above → extended
  extendOff: number // below → curled
  poseHoldMs: number // a new non-pinch pose must persist this long
  pinchHoldMs: number // pinch onset / release debounce
}

export const DEFAULT_THRESHOLDS: Thresholds = {
  pinchOn: 0.3,
  pinchOff: 0.45,
  extendOn: 1.28,
  extendOff: 1.12,
  poseHoldMs: 110,
  pinchHoldMs: 45,
}

/** Derive pinch thresholds from a user's measured open and pinched ratios. */
export function calibratedThresholds(openRatio: number, pinchedRatio: number, base = DEFAULT_THRESHOLDS): Thresholds {
  const span = Math.max(0.15, openRatio - pinchedRatio)
  return { ...base, pinchOn: pinchedRatio + 0.33 * span, pinchOff: pinchedRatio + 0.55 * span }
}

export interface PoseState {
  pose: Pose
  since: number // ms timestamp the pose became active
  pinching: boolean
  pinchStrength: number // 0 open … 1 fully pinched, for UI meters
  extended: [boolean, boolean, boolean, boolean]
  candidate: Pose
  confidence: number // 0..1 how far inside its region the current pose is
}

/** Per-hand pose state machine with per-finger hysteresis, pinch hysteresis and
 *  time-based debouncing (frame-rate independent). */
export class PoseClassifier {
  private state: PoseState = { pose: 'none', since: 0, pinching: false, pinchStrength: 0, extended: [false, false, false, false], candidate: 'none', confidence: 0 }
  private candidateSince = 0
  private pinchCandidate: boolean | null = null
  private pinchCandidateSince = 0

  constructor(public th: Thresholds = DEFAULT_THRESHOLDS) {}

  get current(): PoseState {
    return this.state
  }

  update(f: HandFeatures, nowMs: number): PoseState {
    const th = this.th
    const ext = f.extension.map((e, i) => (this.state.extended[i] ? e > th.extendOff : e > th.extendOn)) as PoseState['extended']

    // --- pinch with hysteresis + debounce. A curled-in index means a fist, not a pinch.
    const wantPinch = !f.indexCurledIn && (this.state.pinching ? f.pinchRatio < th.pinchOff : f.pinchRatio < th.pinchOn)
    let pinching = this.state.pinching
    if (wantPinch !== pinching) {
      if (this.pinchCandidate !== wantPinch) {
        this.pinchCandidate = wantPinch
        this.pinchCandidateSince = nowMs
      }
      if (nowMs - this.pinchCandidateSince >= th.pinchHoldMs) pinching = wantPinch
    } else this.pinchCandidate = null

    // --- static pose from finger states
    const [i, m, r, p] = ext
    const nExt = ext.filter(Boolean).length
    let raw: Pose = 'none'
    if (pinching) raw = 'pinch'
    else if (nExt === 0 && f.indexCurledIn) raw = 'fist'
    else if (nExt === 4) raw = 'open'
    else if (i && m && !r && !p) raw = 'victory'
    else if (i && !m && !r && !p) raw = 'point'

    let pose = this.state.pose
    if (raw === 'pinch' || (this.state.pose === 'pinch' && !pinching)) {
      pose = raw // pinch has its own debounce above
      this.candidateSince = nowMs
    } else if (raw !== this.state.pose) {
      if (raw !== this.state.candidate) this.candidateSince = nowMs
      if (nowMs - this.candidateSince >= th.poseHoldMs) pose = raw
    }

    const strength = Math.min(1, Math.max(0, (th.pinchOff + 0.25 - f.pinchRatio) / (th.pinchOff + 0.25 - th.pinchOn * 0.6)))
    const margin = pose === 'pinch'
      ? (th.pinchOff - f.pinchRatio) / th.pinchOff
      : Math.min(...f.extension.map((e) => Math.abs(e - (th.extendOn + th.extendOff) / 2))) / 0.3
    this.state = {
      pose,
      since: pose !== this.state.pose ? nowMs : this.state.since,
      pinching,
      pinchStrength: strength,
      extended: ext,
      candidate: raw,
      confidence: Math.max(0, Math.min(1, margin)),
    }
    return this.state
  }

  reset() {
    this.state = { pose: 'none', since: 0, pinching: false, pinchStrength: 0, extended: [false, false, false, false], candidate: 'none', confidence: 0 }
    this.pinchCandidate = null
  }
}
