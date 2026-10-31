import { describe, expect, it } from 'vitest'
import { History, type Snapshot } from './history'
import { parsePacket } from './packet'

const snap = (x: number): Snapshot => new Map([['a', { parentId: null, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1], visible: true, deleted: false }]])

describe('History', () => {
  it('skips no-op entries and supports undo / redo', () => {
    const h = new History()
    h.push('noop', snap(0), snap(0))
    expect(h.canUndo).toBe(false)
    h.push('move', snap(0), snap(1))
    expect(h.undo()?.label).toBe('move')
    expect(h.canRedo).toBe(true)
    expect(h.redo()?.after.get('a')!.matrix[12]).toBe(1)
    h.push('move2', snap(1), snap(2))
    expect(h.canRedo).toBe(false)
  })
})

describe('VRNC packet', () => {
  it('parses header and aligned attributes', () => {
    const header = JSON.stringify({ vertexCount: 2, indexCount: 0, attributes: { position: { offset: 0, count: 6, type: 'f32', itemSize: 3 }, observed: { offset: 24, count: 2, type: 'u8', itemSize: 1 } } })
    const hb = new TextEncoder().encode(header)
    const hl = Math.ceil(hb.length / 4) * 4
    const buf = new ArrayBuffer(8 + hl + 28)
    const u8 = new Uint8Array(buf)
    u8.set(new TextEncoder().encode('VRNC'), 0)
    new DataView(buf).setUint32(4, hl, true)
    u8.set(hb, 8)
    new Float32Array(buf, 8 + hl, 6).set([1, 2, 3, 4, 5, 6])
    u8.set([255, 0], 8 + hl + 24)
    const p = parsePacket(buf)
    expect(p.vertexCount).toBe(2)
    expect([...(p.attributes.position as Float32Array)]).toEqual([1, 2, 3, 4, 5, 6])
    expect([...(p.attributes.observed as Uint8Array)]).toEqual([255, 0])
  })
})
