import * as THREE from 'three'
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh'

// BVH-accelerated raycasting: hover/pick against ~1M reconstructed triangles every frame
// (mouse and hand cursor) would otherwise cost tens of milliseconds.
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree
THREE.Mesh.prototype.raycast = acceleratedRaycast
import { parsePacket, type Packet } from './packet'

export function meshGeometry(p: Packet): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(p.attributes.position as Float32Array, 3))
  if (p.attributes.uv) g.setAttribute('uv', new THREE.BufferAttribute(p.attributes.uv as Float32Array, 2))
  else g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(p.vertexCount * 2), 2))
  // vertex colours (multi-view fused meshes) — named `vcolor` so ShaderMaterial does not
  // need three's USE_COLOR plumbing
  if (p.attributes.color) g.setAttribute('vcolor', new THREE.BufferAttribute(p.attributes.color as Uint8Array, 3, true))
  const obs = (p.attributes.observed as Uint8Array) ?? new Uint8Array(p.vertexCount).fill(255)
  g.setAttribute('observed', new THREE.BufferAttribute(obs, 1, true))
  if (p.attributes.index) g.setIndex(new THREE.BufferAttribute(p.attributes.index as Uint32Array, 1))
  g.computeBoundingBox()
  g.computeBoundingSphere()
  if (g.index && g.index.count > 0) g.computeBoundsTree()
  return g
}

export function pointsGeometry(p: Packet): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(p.attributes.position as Float32Array, 3))
  g.setAttribute('color', new THREE.BufferAttribute(p.attributes.color as Uint8Array, 3, true))
  g.computeBoundingSphere()
  return g
}

export async function fetchPacket(url: string, signal?: AbortSignal): Promise<Packet> {
  const r = await fetch(url, { signal })
  if (!r.ok) throw new Error(`failed to load ${url}: ${r.status}`)
  return parsePacket(await r.arrayBuffer())
}
