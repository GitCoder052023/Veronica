import type { P3 } from '../features'
import type { RawHand } from '../GestureEngine'

/** Procedural 21-landmark hand for tests. Fingers are straight chains from their MCP;
 *  `curl` folds a finger back towards the wrist; `pinch` brings thumb and index tips
 *  together. World coords are metres (palm ≈ 9 cm), image coords are normalised. */
export interface HandSpec {
  at?: [number, number] // image-space centre (unmirrored), 0..1
  scale?: number // image-space palm length
  curl?: [number, number, number, number] // index, middle, ring, pinky: 0 straight … 1 fist
  pinch?: number // 0 open … 1 tips touching
  handedness?: 'Left' | 'Right'
}

export function synthHand(s: HandSpec = {}): RawHand {
  const curl = s.curl ?? [0, 0, 0, 0]
  const pinch = s.pinch ?? 0
  const w: P3[] = new Array(21)
  w[0] = { x: 0, y: 0, z: 0 }
  const mcpX = [-0.03, -0.01, 0.01, 0.03]
  const lens = [0.075, 0.08, 0.075, 0.06]
  const fingerBase = [5, 9, 13, 17]
  for (let f = 0; f < 4; f++) {
    const mcp = { x: mcpX[f], y: -0.09, z: 0 }
    w[fingerBase[f]] = mcp
    for (let j = 1; j <= 3; j++) {
      const t = j / 3
      const c = curl[f]
      // straight: continue up (−y); curled: swing towards the palm/wrist
      const ang = c * Math.PI * 0.95 * t
      const L = lens[f] * t
      w[fingerBase[f] + j] = { x: mcp.x, y: mcp.y - Math.cos(ang) * L * (1 - 0.35 * c), z: -Math.sin(ang) * L * 0.6 }
    }
  }
  // thumb: open points out sideways; pinch moves its tip onto the index tip
  const thumbOpen = [{ x: -0.035, y: -0.02, z: 0 }, { x: -0.06, y: -0.045, z: 0 }, { x: -0.08, y: -0.07, z: 0 }, { x: -0.095, y: -0.095, z: 0 }]
  for (let j = 0; j < 4; j++) w[1 + j] = thumbOpen[j]
  const it = w[8]
  w[4] = { x: thumbOpen[3].x + (it.x + 0.004 - thumbOpen[3].x) * pinch, y: thumbOpen[3].y + (it.y - thumbOpen[3].y) * pinch, z: it.z * pinch }
  if (pinch > 0.5) {
    // a pinching index bends a little towards the thumb
    w[8] = { x: it.x - 0.01, y: it.y, z: it.z }
    w[4] = { x: w[8].x + 0.006 * (1 - pinch) * 10, y: w[8].y, z: w[8].z }
  }
  const at = s.at ?? [0.5, 0.5]
  const k = (s.scale ?? 0.18) / 0.09
  const image = w.map((p) => ({ x: at[0] + p.x * k, y: at[1] + 0.05 + p.y * k, z: p.z * k }))
  return { handedness: s.handedness ?? 'Right', score: 0.98, image, world: w }
}
