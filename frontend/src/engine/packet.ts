/** Parser for the backend's "VRNC" binary packets (see backend/veronica/assets.py).
 *  "STRK" is the same format from scenes written before the VERONICA rename. */
export interface PacketAttr { offset: number; count: number; type: 'f32' | 'u32' | 'u8' | 'u16'; itemSize: number }
export interface Packet {
  vertexCount: number
  indexCount: number
  attributes: Record<string, Float32Array | Uint32Array | Uint8Array | Uint16Array>
  itemSizes: Record<string, number>
}

const CTORS = { f32: Float32Array, u32: Uint32Array, u8: Uint8Array, u16: Uint16Array } as const

export function parsePacket(buf: ArrayBuffer): Packet {
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4))
  if (magic !== 'VRNC' && magic !== 'STRK') throw new Error('not a VRNC packet')
  const n = new DataView(buf).getUint32(4, true)
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)).replace(/\0+$/, '')) as {
    vertexCount: number; indexCount: number; attributes: Record<string, PacketAttr>
  }
  const base = 8 + n
  const attributes: Packet['attributes'] = {}
  const itemSizes: Record<string, number> = {}
  for (const [name, a] of Object.entries(header.attributes)) {
    const C = CTORS[a.type]
    // slice() copies into an aligned buffer owned by this attribute
    attributes[name] = new C(buf.slice(base + a.offset, base + a.offset + a.count * C.BYTES_PER_ELEMENT))
    itemSizes[name] = a.itemSize
  }
  return { vertexCount: header.vertexCount, indexCount: header.indexCount, attributes, itemSizes }
}
