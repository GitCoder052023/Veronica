import { describe, expect, it } from 'vitest'
import { PoseClassifier } from '../classifier'
import { computeFeatures } from '../features'
import { GestureEngine, type GestureEvent } from '../GestureEngine'
import { OneEuro } from '../oneEuro'
import { synthHand } from './synthHand'

const FIST: [number, number, number, number] = [1, 1, 1, 1]

describe('features', () => {
  it('separates open, pinched and fist hands', () => {
    const open = computeFeatures(synthHand().image, synthHand().world)
    const pin = computeFeatures(synthHand({ pinch: 1 }).image, synthHand({ pinch: 1 }).world)
    const fist = computeFeatures(synthHand({ curl: FIST }).image, synthHand({ curl: FIST }).world)
    expect(open.pinchRatio).toBeGreaterThan(0.6)
    expect(pin.pinchRatio).toBeLessThan(0.15)
    expect(open.extension.every((e) => e > 1.3)).toBe(true)
    expect(fist.extension.every((e) => e < 1.1)).toBe(true)
    expect(fist.indexCurledIn).toBe(true)
    expect(open.indexCurledIn).toBe(false)
  })

  it('mirrors the cursor like a selfie view', () => {
    const f = computeFeatures(synthHand({ at: [0.2, 0.5] }).image, synthHand().world)
    expect(f.cursor[0]).toBeGreaterThan(0.7)
  })

  it('is invariant to apparent hand size', () => {
    const a = computeFeatures(synthHand({ scale: 0.1 }).image, synthHand().world)
    const b = computeFeatures(synthHand({ scale: 0.3 }).image, synthHand().world)
    expect(a.pinchRatio).toBeCloseTo(b.pinchRatio, 6)
    expect(b.apparentSize / a.apparentSize).toBeCloseTo(3, 1)
  })
})

describe('PoseClassifier', () => {
  const feed = (c: PoseClassifier, spec: Parameters<typeof synthHand>[0], t0: number, ms: number) => {
    let s = c.current
    for (let t = t0; t <= t0 + ms; t += 16) {
      const h = synthHand(spec)
      s = c.update(computeFeatures(h.image, h.world), t)
    }
    return s
  }

  it('debounces: a single-frame blip does not change pose', () => {
    const c = new PoseClassifier()
    feed(c, {}, 0, 300)
    expect(c.current.pose).toBe('open')
    const h = synthHand({ curl: FIST })
    c.update(computeFeatures(h.image, h.world), 316)
    expect(c.current.pose).toBe('open')
    expect(feed(c, { curl: FIST }, 332, 200).pose).toBe('fist')
  })

  it('pinch has hysteresis: half-open does not release', () => {
    const c = new PoseClassifier()
    feed(c, { pinch: 1 }, 0, 200)
    expect(c.current.pinching).toBe(true)
    // find a half-open hand whose ratio lies inside the hysteresis band
    const ratio = (p: number) => { const h = synthHand({ pinch: p }); return computeFeatures(h.image, h.world).pinchRatio }
    const mid = Array.from({ length: 70 }, (_, i) => 0.25 + i * 0.01).find((p) => ratio(p) > c.th.pinchOn && ratio(p) < c.th.pinchOff)!
    expect(mid).toBeDefined()
    expect(feed(c, { pinch: mid }, 216, 200).pinching).toBe(true)
    expect(feed(c, { pinch: 0 }, 432, 200).pinching).toBe(false)
  })

  it('a fist is never a pinch even though the thumb is close to the index', () => {
    const c = new PoseClassifier()
    const s = feed(c, { curl: FIST, pinch: 1 }, 0, 300)
    expect(s.pinching).toBe(false)
    expect(s.pose).toBe('fist')
  })

  it('recognises point and victory', () => {
    expect(feed(new PoseClassifier(), { curl: [0, 1, 1, 1] }, 0, 300).pose).toBe('point')
    expect(feed(new PoseClassifier(), { curl: [0, 0, 1, 1] }, 0, 300).pose).toBe('victory')
  })
})

describe('GestureEngine', () => {
  const run = (g: GestureEngine, frames: { spec: Parameters<typeof synthHand>[0] | Parameters<typeof synthHand>[0][]; ms: number }[]) => {
    const events: GestureEvent[] = []
    let t = 0
    let last
    for (const f of frames) {
      for (let k = 0; k < f.ms; k += 16) {
        const specs = Array.isArray(f.spec) ? f.spec : [f.spec]
        last = g.update(specs.map(synthHand), (t += 16))
        events.push(...last.events)
      }
    }
    return { events, last: last! }
  }

  it('emits a tap for a quick pinch in place', () => {
    const { events } = run(new GestureEngine(), [{ spec: {}, ms: 200 }, { spec: { pinch: 1 }, ms: 150 }, { spec: {}, ms: 150 }])
    expect(events.map((e) => e.type)).toEqual(['pinchStart', 'pinchEnd', 'tap'])
  })

  it('a long pinch-drag is not a tap and the cursor follows the hand', () => {
    const g = new GestureEngine()
    const frames = [{ spec: { at: [0.5, 0.5] as [number, number] }, ms: 200 }, { spec: { at: [0.5, 0.5] as [number, number], pinch: 1 }, ms: 100 }]
    for (let i = 1; i <= 20; i++) frames.push({ spec: { at: [0.5 - i * 0.01, 0.5] as [number, number], pinch: 1 }, ms: 16 })
    const { events, last } = run(g, frames)
    expect(events.some((e) => e.type === 'tap')).toBe(false)
    // hand moved left in camera space → cursor moves right on screen (mirror)
    expect(last.hands[0].cursor[0]).toBeGreaterThan(0.6)
  })

  it('compensates pinch click-slip: cursor stays put while fingers close', () => {
    const g = new GestureEngine()
    let t = 0
    let before: [number, number] = [0, 0]
    for (let k = 0; k < 20; k++) before = g.update([synthHand({})], (t += 16)).hands[0].cursor
    // closing the pinch moves the thumb–index midpoint a lot in image space
    let during: [number, number] = [0, 0]
    for (let k = 0; k < 15; k++) during = g.update([synthHand({ pinch: 1 })], (t += 16)).hands[0].cursor
    expect(Math.hypot(during[0] - before[0], during[1] - before[1])).toBeLessThan(0.03)
  })

  it('detects a fast open-palm swipe once (cooldown)', () => {
    const frames = [{ spec: { at: [0.7, 0.5] as [number, number] }, ms: 300 }]
    for (let i = 1; i <= 8; i++) frames.push({ spec: { at: [0.7 - i * 0.05, 0.5] as [number, number] }, ms: 16 })
    frames.push({ spec: { at: [0.3, 0.5] as [number, number] }, ms: 200 })
    const { events } = run(new GestureEngine(), frames)
    const swipes = events.filter((e) => e.type === 'swipe')
    expect(swipes).toHaveLength(1)
    expect((swipes[0] as { direction: string }).direction).toBe('right')
  })

  it('two open palms held fires once', () => {
    const both = [{ handedness: 'Left' as const, at: [0.3, 0.5] as [number, number] }, { handedness: 'Right' as const, at: [0.7, 0.5] as [number, number] }]
    const { events } = run(new GestureEngine(), [{ spec: both, ms: 1200 }])
    expect(events.filter((e) => e.type === 'twoHandHold')).toHaveLength(1)
  })

  it('ends a pinch when the hand disappears', () => {
    const g = new GestureEngine()
    let t = 0
    for (let k = 0; k < 15; k++) g.update([synthHand({ pinch: 1 })], (t += 16))
    const evs: GestureEvent[] = []
    for (let k = 0; k < 25; k++) evs.push(...g.update([], (t += 16)).events)
    expect(evs.find((e) => e.type === 'pinchEnd')).toMatchObject({ lost: true })
  })
})

describe('OneEuro', () => {
  it('suppresses jitter at rest and tracks fast motion', () => {
    const f = new OneEuro(1.4, 2.5)
    let maxDev = 0
    for (let i = 0; i < 120; i++) {
      const y = f.filter(0.5 + (i % 2 ? 0.004 : -0.004), i / 60)
      if (i > 30) maxDev = Math.max(maxDev, Math.abs(y - 0.5))
    }
    expect(maxDev).toBeLessThan(0.002)
    const g = new OneEuro(1.4, 2.5)
    let y = 0
    for (let i = 0; i < 30; i++) y = g.filter(i * 0.02, i / 60) // 1.2 units/s ramp
    expect(Math.abs(y - 29 * 0.02)).toBeLessThan(0.05)
  })
})
