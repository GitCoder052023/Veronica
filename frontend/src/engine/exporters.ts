import * as THREE from 'three'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js'
import { PLYExporter } from 'three/examples/jsm/exporters/PLYExporter.js'
import type { VeronicaEngine, VeronicaNode } from './VeronicaEngine'

export interface ExportOptions {
  includeInferred: boolean
  includeStructure: boolean
  selectionOnly: boolean
}

/** Build a plain three.js scene mirroring the *edited* state (current transforms,
 *  hierarchy, visibility) with portable materials. Photo-textured surfaces become
 *  unlit textured materials (KHR_materials_unlit in glTF), which is what they are. */
function buildExportScene(engine: VeronicaEngine, o: ExportOptions): THREE.Scene {
  const out = new THREE.Scene()
  const root = new THREE.Group()
  root.name = engine.manifest?.source.name ?? 'veronica-scene'
  engine.world.updateMatrix()
  root.applyMatrix4(engine.world.matrix)
  out.add(root)
  const mats = new Map<THREE.Texture | null, THREE.MeshBasicMaterial>()
  const matFor = (tex: THREE.Texture | null) => {
    if (!mats.has(tex)) mats.set(tex, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, color: tex ? 0xffffff : 0xb8bcc4 }))
    return mats.get(tex)!
  }
  const vertexColorMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
  const include = (n: VeronicaNode) => {
    if (n.deleted || !n.visible) return false
    if (!o.includeStructure && (n.kind === 'structure' || n.kind === 'background')) return false
    if (o.selectionOnly && !engine.selection.some((s) => engine.isAncestorOrSelf(s, n.id))) return false
    return true
  }
  const visit = (n: VeronicaNode, parent: THREE.Object3D) => {
    if (!include(n) && n.kind !== 'group') return
    const g = new THREE.Group()
    g.name = n.name
    n.object.updateMatrix()
    g.applyMatrix4(n.object.matrix)
    g.userData = { veronicaId: n.id, label: n.label, kind: n.kind, provenance: n.meta?.provenance ?? null }
    for (const [key, m] of [['surface', n.surface], ['completion', n.completion]] as const) {
      if (!m) continue
      if (key === 'completion' && !o.includeInferred) continue
      let geom = m.geometry
      if (!o.includeInferred && key === 'surface') geom = dropInferred(geom)
      if (key === 'completion' && m.material.uniforms.discardObserved.value > 0.5) geom = dropObserved(geom)
      let mat: THREE.Material = matFor(m.material.uniforms.map.value)
      const vc = geom.getAttribute('vcolor')
      if (vc && !m.material.uniforms.map.value) {
        geom = geom.clone()
        geom.setAttribute('color', vc)
        mat = vertexColorMat
      }
      const mesh = new THREE.Mesh(geom, mat)
      mesh.name = `${n.name} · ${key === 'surface' ? 'observed' : 'inferred'}`
      g.add(mesh)
    }
    parent.add(g)
    for (const c of engine.children(n.id)) visit(c, g)
    if (!g.children.length) g.removeFromParent()
  }
  for (const n of engine.children(null)) visit(n, root)
  return out
}

function filterFaces(g: THREE.BufferGeometry, keep: (o: number) => boolean): THREE.BufferGeometry {
  const obs = g.getAttribute('observed')
  const idx = g.getIndex()
  if (!obs || !idx) return g
  const src = idx.array as Uint32Array
  const out: number[] = []
  for (let i = 0; i < src.length; i += 3) {
    const a = obs.getX(src[i]), b = obs.getX(src[i + 1]), c = obs.getX(src[i + 2])
    if (keep(Math.min(a, b, c)) && keep(Math.max(a, b, c))) out.push(src[i], src[i + 1], src[i + 2])
  }
  const ng = g.clone()
  ng.setIndex(out)
  return ng
}
const dropInferred = (g: THREE.BufferGeometry) => filterFaces(g, (o) => o > 0.5)
const dropObserved = (g: THREE.BufferGeometry) => filterFaces(g, (o) => o <= 0.5)

export async function exportGLB(engine: VeronicaEngine, o: ExportOptions): Promise<Blob> {
  const scene = buildExportScene(engine, o)
  const res = await new GLTFExporter().parseAsync(scene, { binary: true, onlyVisible: true, maxTextureSize: 4096 })
  return new Blob([res as ArrayBuffer], { type: 'model/gltf-binary' })
}

/** OBJ carries geometry and UVs; the texture is not embedded (OBJ needs a separate
 *  .mtl + image), which the UI states plainly. */
export function exportOBJ(engine: VeronicaEngine, o: ExportOptions): Blob {
  const scene = buildExportScene(engine, o)
  return new Blob([new OBJExporter().parse(scene)], { type: 'text/plain' })
}

/** PLY point cloud of the observed points (current transforms applied), with colour. */
export function exportPLY(engine: VeronicaEngine, o: ExportOptions): Promise<Blob> {
  const out = new THREE.Scene()
  engine.scene.updateMatrixWorld(true)
  for (const n of engine.liveNodes()) {
    if (!n.points || !n.visible) continue
    if (o.selectionOnly && !engine.selection.some((s) => engine.isAncestorOrSelf(s, n.id))) continue
    const p = new THREE.Points(n.points.geometry.clone().applyMatrix4(n.points.matrixWorld))
    out.add(p)
  }
  return new Promise((res) => new PLYExporter().parse(out, (data) => res(new Blob([data as ArrayBuffer], { type: 'application/octet-stream' })), { binary: true }))
}

export function download(blob: Blob, name: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 5000)
}
