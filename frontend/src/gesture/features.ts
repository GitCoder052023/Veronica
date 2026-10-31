/** Geometric hand features from MediaPipe's 21 landmarks.
 *
 *  Shape features (finger extension, pinch, orientation) use the *world* landmarks —
 *  metric, hand-centred 3D — and are normalised by palm size, so they do not change
 *  with distance to the camera or hand size. Position features (cursor, apparent
 *  size as a depth proxy) use the normalised image landmarks.
 */
export type P3 = { x: number; y: number; z: number }

export const LM = {
  WRIST: 0, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_TIP: 20,
} as const

export interface HandFeatures {
  palmSize: number // world metres, wrist → middle MCP
  pinchRatio: number // thumb-tip ↔ index-tip / palm size
  extension: [number, number, number, number] // index, middle, ring, pinky: tip-to-wrist / pip-to-wrist
  thumbOut: number // thumb tip ↔ index MCP / palm size
  indexCurledIn: boolean // index tip folded into the palm (fist), not pinching
  cursor: [number, number] // image space, mirrored x (selfie view), pinch midpoint
  pointer: [number, number] // image space, mirrored, index tip
  palmCenter: [number, number] // image space, mirrored
  apparentSize: number // image-space wrist → middle MCP (bigger = closer)
  basis: [P3, P3, P3] // right-handed orthonormal frame in three.js camera axes (mirrored)
}

const sub = (a: P3, b: P3): P3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z })
const len = (a: P3) => Math.hypot(a.x, a.y, a.z)
const dist = (a: P3, b: P3) => len(sub(a, b))
const norm = (a: P3): P3 => {
  const l = len(a) || 1
  return { x: a.x / l, y: a.y / l, z: a.z / l }
}
const cross = (a: P3, b: P3): P3 => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x })

/** MediaPipe (x right, y down, z away from camera) → three.js camera axes, mirrored
 *  like the selfie preview the user sees: (-x, -y, -z). */
const toView = (p: P3): P3 => ({ x: -p.x, y: -p.y, z: -p.z })

export function computeFeatures(image: P3[], world: P3[]): HandFeatures {
  const w = world
  const palm = Math.max(1e-4, dist(w[LM.WRIST], w[LM.MIDDLE_MCP]))
  const ext = (pip: number, tip: number) => dist(w[LM.WRIST], w[tip]) / Math.max(1e-4, dist(w[LM.WRIST], w[pip]))
  const extension: HandFeatures['extension'] = [
    ext(LM.INDEX_PIP, LM.INDEX_TIP), ext(LM.MIDDLE_PIP, LM.MIDDLE_TIP), ext(LM.RING_PIP, LM.RING_TIP), ext(LM.PINKY_PIP, LM.PINKY_TIP),
  ]
  const m = (i: number): [number, number] => [1 - image[i].x, image[i].y]
  const mid = (a: number, b: number): [number, number] => [1 - (image[a].x + image[b].x) / 2, (image[a].y + image[b].y) / 2]
  const vw = w.map(toView)
  const x = norm(sub(vw[LM.PINKY_MCP], vw[LM.INDEX_MCP]))
  const yRaw = sub(vw[LM.MIDDLE_MCP], vw[LM.WRIST])
  // Gram–Schmidt so the frame is orthonormal
  const yd = x.x * yRaw.x + x.y * yRaw.y + x.z * yRaw.z
  const y = norm({ x: yRaw.x - yd * x.x, y: yRaw.y - yd * x.y, z: yRaw.z - yd * x.z })
  const z = cross(x, y)
  const palmIdx: number[] = [LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP]
  const pc: [number, number] = [1 - palmIdx.reduce((s, i) => s + image[i].x, 0) / 5, palmIdx.reduce((s, i) => s + image[i].y, 0) / 5]
  return {
    palmSize: palm,
    pinchRatio: dist(w[LM.THUMB_TIP], w[LM.INDEX_TIP]) / palm,
    extension,
    thumbOut: dist(w[LM.THUMB_TIP], w[LM.INDEX_MCP]) / palm,
    indexCurledIn: dist(w[LM.WRIST], w[LM.INDEX_TIP]) < 0.95 * dist(w[LM.WRIST], w[LM.INDEX_PIP]),
    cursor: mid(LM.THUMB_TIP, LM.INDEX_TIP),
    pointer: m(LM.INDEX_TIP),
    palmCenter: pc,
    apparentSize: Math.hypot(image[LM.WRIST].x - image[LM.MIDDLE_MCP].x, image[LM.WRIST].y - image[LM.MIDDLE_MCP].y),
    basis: [x, y, z],
  }
}
