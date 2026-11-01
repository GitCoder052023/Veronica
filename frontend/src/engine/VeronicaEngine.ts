import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import type { Manifest, ManifestNode } from '../api/types'
import { Emitter } from './emitter'
import { fetchPacket, meshGeometry, pointsGeometry } from './geometry'
import { History, type NodeState, type Snapshot } from './history'
import { createPointsMaterial, createSurfaceMaterial, setMode, type RenderMode } from './material'

export type RepMode = 'hybrid' | 'observed' | 'points'
export type GizmoMode = 'translate' | 'rotate' | 'scale'
export type Space = 'world' | 'local'
export type Axis = 'x' | 'y' | 'z'
export type ViewPreset = 'photo' | 'front' | 'back' | 'left' | 'right' | 'top' | 'iso'

export interface VeronicaNode {
  id: string
  name: string
  label: string
  kind: 'object' | 'structure' | 'background' | 'group'
  object: THREE.Group
  meta: ManifestNode | null
  parentId: string | null
  original: THREE.Matrix4 // local matrix at load / creation
  visible: boolean
  locked: boolean
  deleted: boolean
  duplicateOf?: string
  surface?: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>
  completion?: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>
  points?: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>
}

export interface ModalState {
  op: GizmoMode
  axis: Axis | null
  value: string
}

type Events = {
  change: void // structural / transform change → UI should re-read
  selection: string[]
  hover: string | null
  history: void
  modal: ModalState | null
  stats: { fps: number; triangles: number; calls: number }
  loading: { done: number; total: number } | null
}

const WORLD_ID = '__world'
const _m = new THREE.Matrix4()
const _v = new THREE.Vector3()

export class VeronicaEngine {
  readonly events = new Emitter<Events>()
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  readonly camera: THREE.PerspectiveCamera
  readonly orbit: OrbitControls
  readonly gizmo: TransformControls
  readonly world = new THREE.Group()
  readonly history = new History()
  readonly nodes = new Map<string, VeronicaNode>()
  manifest: Manifest | null = null
  selection: string[] = []
  hovered: string | null = null

  renderMode: RenderMode = 'photo'
  repMode: RepMode = 'hybrid'
  showInferred = true
  inferredStyle = true
  gizmoMode: GizmoMode = 'translate'
  space: Space = 'world'
  snap = { enabled: false, translate: 0.05, rotateDeg: 15, scale: 0.1 }
  axisLock: Record<Axis, boolean> = { x: true, y: true, z: true }
  isolateId: string | null = null
  precision = false

  private pivot = new THREE.Object3D()
  private gizmoHelper: THREE.Object3D
  private boxHelpers = new Map<string, THREE.BoxHelper>()
  private grid: THREE.GridHelper
  private textures = new Map<string, THREE.Texture>()
  private raycaster = new THREE.Raycaster()
  private frameHooks = new Set<(dt: number) => void>()
  private photoPose: { matrix: THREE.Matrix4; fov: number } | null = null
  private cameraRig = new THREE.Group()
  showCameras = true
  private tween: { from: THREE.Vector3; to: THREE.Vector3; qf: THREE.Quaternion; qt: THREE.Quaternion; tf: THREE.Vector3; tt: THREE.Vector3; fovF: number; fovT: number; t: number; dur: number } | null = null
  private manip: { ids: string[]; before: Snapshot; w0: Map<string, THREE.Matrix4>; pivot0: THREE.Matrix4; label: string } | null = null
  private modal: (ModalState & { start: THREE.Vector2; pivotScreen: THREE.Vector2; planeHit0: THREE.Vector3 | null; mouse: THREE.Vector2 }) | null = null
  private abort: AbortController | null = null
  private lastTime = performance.now()
  private fpsEma = 60
  private statsTimer = 0
  private copyCounter = 0
  private groupCounter = 0
  private disposed = false
  private worldOriginal = new THREE.Matrix4()
  private recording: { rec: MediaRecorder; chunks: Blob[]; turntable: boolean } | null = null

  constructor(readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.setClearColor(0x000000, 0)
    container.appendChild(this.renderer.domElement)
    this.renderer.domElement.classList.add('viewport-canvas')

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 400)
    this.camera.position.set(2, 1.6, 3)
    this.scene.add(this.world)
    this.scene.add(this.pivot)
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.2))

    this.grid = new THREE.GridHelper(20, 80, 0x8a9296, 0xc4c9cc)
    ;(this.grid.material as THREE.Material).transparent = true
    ;(this.grid.material as THREE.Material).opacity = 0.55
    ;(this.grid.material as THREE.Material).depthWrite = false
    this.scene.add(this.grid)
    this.world.add(this.cameraRig)

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement)
    this.orbit.enableDamping = true
    this.orbit.dampingFactor = 0.12
    this.orbit.screenSpacePanning = true
    this.orbit.zoomToCursor = true

    this.gizmo = new TransformControls(this.camera, this.renderer.domElement)
    this.gizmo.setSize(0.85)
    this.gizmoHelper = this.gizmo.getHelper()
    this.scene.add(this.gizmoHelper)
    this.gizmo.addEventListener('dragging-changed', (e) => {
      this.orbit.enabled = !(e as unknown as { value: boolean }).value
    })
    this.gizmo.addEventListener('mouseDown', () => this.beginManipulation(this.selection, `${this.gizmoMode} (gizmo)`))
    this.gizmo.addEventListener('objectChange', () => {
      if (!this.manip) return
      this.pivot.updateMatrixWorld()
      this.applyManipulation(_m.copy(this.pivot.matrixWorld).multiply(new THREE.Matrix4().copy(this.manip.pivot0).invert()))
    })
    this.gizmo.addEventListener('mouseUp', () => this.endManipulation(true))

    this.bindPointer()
    const ro = new ResizeObserver(() => this.resize())
    ro.observe(container)
    this.resize()
    this.renderer.setAnimationLoop(() => this.tick())
  }

  // ------------------------------------------------------------------ lifecycle
  dispose() {
    this.disposed = true
    this.renderer.setAnimationLoop(null)
    this.clearScene()
    this.gizmo.dispose()
    this.orbit.dispose()
    this.renderer.dispose()
    this.renderer.domElement.remove()
  }

  private clearScene() {
    this.abort?.abort()
    this.endModal(false)
    this.gizmo.detach()
    for (const n of this.nodes.values()) {
      n.object.removeFromParent()
      n.object.traverse((o) => {
        const m = o as THREE.Mesh
        if (m.geometry && !n.duplicateOf) m.geometry.dispose()
        if (m.material) (m.material as THREE.Material).dispose()
      })
    }
    for (const t of this.textures.values()) t.dispose()
    this.textures.clear()
    for (const b of this.boxHelpers.values()) b.removeFromParent()
    this.boxHelpers.clear()
    this.nodes.clear()
    this.selection = []
    this.hovered = null
    this.isolateId = null
    this.world.matrix.identity().decompose(this.world.position, this.world.quaternion, this.world.scale)
    this.history.clear()
  }

  resize() {
    const w = this.container.clientWidth || 1
    const h = this.container.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  onFrame(fn: (dt: number) => void) {
    this.frameHooks.add(fn)
    return () => this.frameHooks.delete(fn)
  }

  // ------------------------------------------------------------------ loading
  async load(manifest: Manifest, fileUrl: (name: string) => string, preserve?: Snapshot) {
    this.clearScene()
    this.manifest = manifest
    this.abort = new AbortController()
    const signal = this.abort.signal
    const loader = new THREE.TextureLoader()
    for (const [key, name] of Object.entries(manifest.textures)) {
      if (!name.endsWith('.jpg') && !name.endsWith('.png')) continue
      if (key === 'labels') continue
      const tex = await loader.loadAsync(fileUrl(name))
      tex.colorSpace = THREE.SRGBColorSpace
      tex.flipY = false
      tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy()
      tex.generateMipmaps = true
      this.textures.set(key, tex)
    }
    const jobs = manifest.nodes.flatMap((n) => Object.entries(n.representations).map(([rep, r]) => ({ n, rep, r })))
    let done = 0
    this.events.emit('loading', { done, total: jobs.length })
    const packets = new Map<string, Awaited<ReturnType<typeof fetchPacket>>>()
    await Promise.all(
      jobs.map(async ({ n, rep, r }) => {
        packets.set(`${n.id}/${rep}`, await fetchPacket(fileUrl(r!.file), signal))
        this.events.emit('loading', { done: ++done, total: jobs.length })
      }),
    )
    if (signal.aborted || this.disposed) return
    for (const n of manifest.nodes) {
      const g = new THREE.Group()
      g.name = n.name
      g.position.fromArray(n.transform.position)
      g.quaternion.fromArray(n.transform.quaternion)
      g.scale.fromArray(n.transform.scale)
      g.updateMatrix()
      const node: VeronicaNode = {
        id: n.id, name: n.name, label: n.label, kind: n.kind, object: g, meta: n, parentId: null,
        original: g.matrix.clone(), visible: true, locked: n.kind !== 'object', deleted: false,
      }
      const surf = packets.get(`${n.id}/surface`)
      if (surf) {
        node.surface = new THREE.Mesh(meshGeometry(surf), createSurfaceMaterial(this.textures.get(n.representations.surface!.texture ?? '') ?? null, !!surf.attributes.color))
        g.add(node.surface)
      }
      const comp = packets.get(`${n.id}/completion`)
      if (comp) {
        node.completion = new THREE.Mesh(meshGeometry(comp), createSurfaceMaterial(this.textures.get(n.representations.completion!.texture ?? '') ?? null, !!comp.attributes.color))
        g.add(node.completion)
      }
      const pts = packets.get(`${n.id}/points`)
      if (pts) {
        node.points = new THREE.Points(pointsGeometry(pts), createPointsMaterial())
        g.add(node.points)
      }
      g.traverse((o) => (o.userData.nodeId = n.id))
      this.world.add(g)
      this.nodes.set(n.id, node)
    }
    this.worldOriginal.identity()
    if (manifest.world.floorY !== null) this.grid.position.y = manifest.world.floorY
    this.grid.visible = true
    const W = manifest.camera.worldFromCamera
    this.photoPose = {
      matrix: new THREE.Matrix4().set(...(W.flat() as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number])),
      fov: manifest.camera.vfov,
    }
    this.buildCameraRig(manifest)
    if (preserve) this.applySnapshot(preserve, true)
    this.applyDisplay()
    this.setView('photo', false)
    this.events.emit('loading', null)
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  /** Wireframe frustums at every photo's recovered camera pose. */
  private buildCameraRig(m: Manifest) {
    this.cameraRig.clear()
    const cams = m.cameras ?? [{ worldFromCamera: m.camera.worldFromCamera, vfov: m.camera.vfov, image: m.textures.image }]
    const aspect = m.source.width / m.source.height
    cams.forEach((c, i) => {
      const M = new THREE.Matrix4().set(...(c.worldFromCamera.flat() as Parameters<THREE.Matrix4['set']>))
      const d = 0.18
      const hh = Math.tan(THREE.MathUtils.degToRad(c.vfov) / 2) * d
      const hw = hh * aspect
      const pts = [[0, 0, 0], [-hw, -hh, -d], [0, 0, 0], [hw, -hh, -d], [0, 0, 0], [hw, hh, -d], [0, 0, 0], [-hw, hh, -d],
        [-hw, -hh, -d], [hw, -hh, -d], [hw, -hh, -d], [hw, hh, -d], [hw, hh, -d], [-hw, hh, -d], [-hw, hh, -d], [-hw, -hh, -d],
        [-hw * 0.4, hh * 1.1, -d], [0, hh * 1.5, -d], [0, hh * 1.5, -d], [hw * 0.4, hh * 1.1, -d]]
      const g = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)))
      const line = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: i === 0 ? 0x15181a : 0x626a6f, transparent: true, opacity: 0.85 }))
      line.applyMatrix4(M)
      line.userData.cameraIndex = i
      this.cameraRig.add(line)
    })
    this.cameraRig.visible = this.showCameras
  }

  setShowCameras(v: boolean) {
    this.showCameras = v
    this.cameraRig.visible = v
    this.events.emit('change', undefined)
  }

  /** Fly to the pose of the i-th input photo. */
  viewFromCamera(i: number) {
    const c = this.manifest?.cameras?.[i]
    if (!c) return this.setView('photo')
    const M = new THREE.Matrix4().set(...(c.worldFromCamera.flat() as Parameters<THREE.Matrix4['set']>)).premultiply(this.world.matrixWorld)
    const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3()
    M.decompose(p, q, s)
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q)
    const dist = Math.max(0.3, this.sceneCenter().sub(p).dot(fwd))
    this.flyTo(p, q, p.clone().add(fwd.multiplyScalar(dist)), c.vfov, true)
  }

  // ------------------------------------------------------------------ display
  applyDisplay() {
    for (const n of this.nodes.values()) {
      const isolated = this.isolateId !== null && !this.isAncestorOrSelf(this.isolateId, n.id) && !this.isAncestorOrSelf(n.id, this.isolateId)
      n.object.visible = n.visible && !isolated
      const pointsMode = this.repMode === 'points' && !!n.points
      const revolve = n.meta?.provenance.completion === 'surface-of-revolution'
      if (n.surface) {
        n.surface.visible = !pointsMode
        setMode(n.surface.material, this.renderMode)
        n.surface.material.uniforms.showInferred.value = this.showInferred ? 1 : 0
        n.surface.material.uniforms.inferredStyle.value = this.inferredStyle ? 1 : 0
      }
      if (n.completion) {
        n.completion.visible = !pointsMode && this.repMode === 'hybrid' && this.showInferred
        setMode(n.completion.material, this.renderMode)
        // revolve completions re-describe the observed side too; let the true surface win there
        n.completion.material.uniforms.discardObserved.value = revolve ? 1 : 0
        n.completion.material.uniforms.inferredStyle.value = this.inferredStyle ? 1 : 0
      }
      if (n.points) n.points.visible = pointsMode
      const sel = this.selection.includes(n.id) || this.selection.some((s) => this.isAncestorOrSelf(s, n.id))
      const amt = sel ? 0.55 : this.hovered === n.id || (this.hovered && this.isAncestorOrSelf(this.hovered, n.id)) ? 0.3 : 0
      for (const m of [n.surface, n.completion]) if (m) m.material.uniforms.highlightAmount.value = amt
    }
  }

  setRenderMode(m: RenderMode) { this.renderMode = m; this.applyDisplay(); this.events.emit('change', undefined) }
  setRepMode(m: RepMode) { this.repMode = m; this.applyDisplay(); this.events.emit('change', undefined) }
  setShowInferred(v: boolean) { this.showInferred = v; this.applyDisplay(); this.events.emit('change', undefined) }
  setInferredStyle(v: boolean) { this.inferredStyle = v; this.applyDisplay(); this.events.emit('change', undefined) }
  setGridVisible(v: boolean) { this.grid.visible = v; this.events.emit('change', undefined) }
  get gridVisible() { return this.grid.visible }

  // ------------------------------------------------------------------ hierarchy helpers
  /** True when `ancestor` is `id` or one of its parents. */
  isAncestorOrSelf(ancestor: string, id: string): boolean {
    let cur: string | null = id
    while (cur) {
      if (cur === ancestor) return true
      cur = this.nodes.get(cur)?.parentId ?? null
    }
    return false
  }

  topLevelOf(id: string): string {
    let cur = this.nodes.get(id)!
    while (cur.parentId) cur = this.nodes.get(cur.parentId)!
    return cur.id
  }

  children(id: string | null): VeronicaNode[] {
    return [...this.nodes.values()].filter((n) => n.parentId === id && !n.deleted)
  }

  liveNodes(): VeronicaNode[] {
    return [...this.nodes.values()].filter((n) => !n.deleted)
  }

  // ------------------------------------------------------------------ picking & selection
  pick(ndc: THREE.Vector2): { id: string; point: THREE.Vector3; distance: number } | null {
    this.raycaster.setFromCamera(ndc, this.camera)
    this.raycaster.params.Points = { threshold: 0.01 }
    const targets: THREE.Object3D[] = []
    for (const n of this.nodes.values()) {
      if (n.deleted || !n.object.visible) continue
      for (const o of [n.surface, n.completion, n.points]) if (o?.visible) targets.push(o)
    }
    const hits = this.raycaster.intersectObjects(targets, false)
    for (const h of hits) {
      // skip fragments the shader discards (inferred hidden / observed-discarded)
      const mat = (h.object as THREE.Mesh).material as THREE.ShaderMaterial
      if (mat?.uniforms && h.face) {
        const obs = (h.object as THREE.Mesh).geometry.getAttribute('observed')
        const o = obs ? obs.getX(h.face.a) : 1
        if (mat.uniforms.discardObserved.value > 0.5 && o > 0.5) continue
      }
      return { id: h.object.userData.nodeId as string, point: h.point.clone(), distance: h.distance }
    }
    return null
  }

  /** Resolve a picked leaf node to what a click should select (its top-level group). */
  selectable(id: string, deep = false) {
    return deep ? id : this.topLevelOf(id)
  }

  setHovered(id: string | null) {
    if (id === this.hovered) return
    this.hovered = id
    this.applyDisplay()
    this.events.emit('hover', id)
  }

  select(ids: string[], mode: 'set' | 'add' | 'toggle' = 'set') {
    let next: string[]
    if (mode === 'set') next = [...new Set(ids)]
    else if (mode === 'add') next = [...new Set([...this.selection, ...ids])]
    else {
      next = [...this.selection]
      for (const id of ids) next = next.includes(id) ? next.filter((x) => x !== id) : [...next, id]
    }
    this.selection = next.filter((id) => this.nodes.has(id) && !this.nodes.get(id)!.deleted)
    this.syncSelectionHelpers()
    this.applyDisplay()
    this.events.emit('selection', this.selection)
  }

  selectAll() {
    this.select(this.liveNodes().filter((n) => !n.parentId && !n.locked && n.visible).map((n) => n.id))
  }

  private syncSelectionHelpers() {
    for (const [id, b] of this.boxHelpers) {
      if (!this.selection.includes(id)) {
        b.removeFromParent()
        b.dispose()
        this.boxHelpers.delete(id)
      }
    }
    for (const id of this.selection) {
      if (!this.boxHelpers.has(id)) {
        const b = new THREE.BoxHelper(this.nodes.get(id)!.object, 0xe39a12)
        ;(b.material as THREE.LineBasicMaterial).transparent = true
        ;(b.material as THREE.LineBasicMaterial).opacity = 0.7
        this.scene.add(b)
        this.boxHelpers.set(id, b)
      }
    }
    this.updatePivot()
  }

  /** Movable nodes among the selection (locked structure is excluded). */
  manipulable(ids = this.selection) {
    return ids.filter((id) => {
      const n = this.nodes.get(id)
      return n && !n.deleted && !n.locked
    })
  }

  private updatePivot() {
    const ids = this.manipulable()
    if (!ids.length) {
      this.gizmo.detach()
      return
    }
    const box = new THREE.Box3()
    for (const id of ids) box.expandByObject(this.nodes.get(id)!.object)
    box.getCenter(this.pivot.position)
    if (ids.length === 1 && this.space === 'local') this.nodes.get(ids[0])!.object.getWorldQuaternion(this.pivot.quaternion)
    else this.pivot.quaternion.identity()
    this.pivot.scale.set(1, 1, 1)
    this.pivot.updateMatrixWorld()
    this.gizmo.attach(this.pivot)
    this.applyGizmoSettings()
  }

  selectionCenter(ids = this.selection): THREE.Vector3 | null {
    if (!ids.length) return null
    const box = new THREE.Box3()
    for (const id of ids) box.expandByObject(this.nodes.get(id)!.object)
    return box.getCenter(new THREE.Vector3())
  }

  // ------------------------------------------------------------------ gizmo settings
  setGizmoMode(m: GizmoMode) { this.gizmoMode = m; this.applyGizmoSettings(); this.events.emit('change', undefined) }
  setSpace(s: Space) { this.space = s; this.updatePivot(); this.events.emit('change', undefined) }
  setSnap(s: Partial<VeronicaEngine['snap']>) { this.snap = { ...this.snap, ...s }; this.applyGizmoSettings(); this.events.emit('change', undefined) }
  setAxisLock(a: Record<Axis, boolean>) { this.axisLock = a; this.applyGizmoSettings(); this.events.emit('change', undefined) }
  setPrecision(v: boolean) { this.precision = v; this.events.emit('change', undefined) }

  private applyGizmoSettings() {
    this.gizmo.setMode(this.gizmoMode)
    this.gizmo.setSpace(this.space)
    this.gizmo.showX = this.axisLock.x
    this.gizmo.showY = this.axisLock.y
    this.gizmo.showZ = this.axisLock.z
    this.gizmo.setTranslationSnap(this.snap.enabled ? this.snap.translate : null)
    this.gizmo.setRotationSnap(this.snap.enabled ? THREE.MathUtils.degToRad(this.snap.rotateDeg) : null)
    this.gizmo.setScaleSnap(this.snap.enabled ? this.snap.scale : null)
  }

  // ------------------------------------------------------------------ manipulation core
  /** Start a transform session over `ids`. Every caller (gizmo, modal keys, gestures)
   *  then feeds world-space delta matrices; the session ends with commit or cancel. */
  beginManipulation(ids: string[], label: string): boolean {
    if (this.manip) this.endManipulation(true)
    const live = this.manipulable(ids)
    if (!live.length) return false
    this.world.updateMatrixWorld(true)
    const w0 = new Map<string, THREE.Matrix4>()
    for (const id of live) w0.set(id, this.nodes.get(id)!.object.matrixWorld.clone())
    this.pivot.updateMatrixWorld()
    this.manip = { ids: live, before: this.capture(), w0, pivot0: this.pivot.matrixWorld.clone(), label }
    return true
  }

  get manipulating() { return this.manip !== null }
  get manipulationPivot(): THREE.Vector3 { return new THREE.Vector3().setFromMatrixPosition(this.manip?.pivot0 ?? this.pivot.matrixWorld) }

  applyManipulation(delta: THREE.Matrix4) {
    if (!this.manip) return
    for (const id of this.manip.ids) {
      const obj = this.nodes.get(id)!.object
      const w = new THREE.Matrix4().multiplyMatrices(delta, this.manip.w0.get(id)!)
      this.setWorldMatrix(obj, w)
    }
    if (!this.gizmo.dragging) {
      this.pivot.matrix.multiplyMatrices(delta, this.manip.pivot0)
      this.pivot.matrix.decompose(this.pivot.position, this.pivot.quaternion, this.pivot.scale)
      this.pivot.updateMatrixWorld()
    }
    for (const b of this.boxHelpers.values()) b.update()
  }

  endManipulation(commit: boolean) {
    const m = this.manip
    if (!m) return
    this.manip = null
    if (commit) this.history.push(m.label, m.before, this.capture())
    else this.applySnapshot(m.before)
    this.updatePivot()
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  private setWorldMatrix(obj: THREE.Object3D, world: THREE.Matrix4) {
    obj.parent!.updateMatrixWorld(true)
    const local = new THREE.Matrix4().copy(obj.parent!.matrixWorld).invert().multiply(world)
    local.decompose(obj.position, obj.quaternion, obj.scale)
    obj.updateMatrixWorld(true)
  }

  /** World-space delta helpers (rotation / scale about a point). */
  static aboutPoint(p: THREE.Vector3, m: THREE.Matrix4): THREE.Matrix4 {
    return new THREE.Matrix4().makeTranslation(p.x, p.y, p.z).multiply(m).multiply(new THREE.Matrix4().makeTranslation(-p.x, -p.y, -p.z))
  }

  // ------------------------------------------------------------------ snapshots / history
  capture(): Snapshot {
    const s: Snapshot = new Map()
    for (const n of this.nodes.values()) {
      n.object.updateMatrix()
      s.set(n.id, { parentId: n.parentId, matrix: n.object.matrix.toArray(), visible: n.visible, deleted: n.deleted })
    }
    this.world.updateMatrix()
    s.set(WORLD_ID, { parentId: null, matrix: this.world.matrix.toArray(), visible: true, deleted: false })
    return s
  }

  applySnapshot(s: Snapshot, onlyExisting = false) {
    for (const n of this.nodes.values()) {
      const st: NodeState | undefined = s.get(n.id)
      if (!st) {
        if (!onlyExisting) this.setDeleted(n, true)
        continue
      }
      // a parent may be missing when restoring edits onto a rebuilt scene
      n.parentId = st.parentId && this.nodes.has(st.parentId) ? st.parentId : null
      n.visible = st.visible
      this.setDeleted(n, st.deleted)
      if (!st.deleted) {
        const parent = n.parentId ? this.nodes.get(n.parentId)!.object : this.world
        if (n.object.parent !== parent) parent.add(n.object)
      }
      new THREE.Matrix4().fromArray(st.matrix).decompose(n.object.position, n.object.quaternion, n.object.scale)
    }
    const w = s.get(WORLD_ID)
    if (w) new THREE.Matrix4().fromArray(w.matrix).decompose(this.world.position, this.world.quaternion, this.world.scale)
    this.scene.updateMatrixWorld(true)
    this.select(this.selection)
  }

  private setDeleted(n: VeronicaNode, del: boolean) {
    n.deleted = del
    if (del) n.object.removeFromParent()
  }

  private commit(label: string, before: Snapshot) {
    this.history.push(label, before, this.capture())
    this.applyDisplay()
    this.updatePivot()
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  undo() {
    if (this.manip) return this.endManipulation(false)
    const e = this.history.undo()
    if (e) this.applySnapshot(e.before)
    this.applyDisplay()
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  redo() {
    const e = this.history.redo()
    if (e) this.applySnapshot(e.after)
    this.applyDisplay()
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  // ------------------------------------------------------------------ object operations
  setVisible(ids: string[], visible: boolean) {
    const before = this.capture()
    for (const id of ids) {
      const n = this.nodes.get(id)
      if (n) n.visible = visible
    }
    this.commit(visible ? 'show' : 'hide', before)
  }

  showAll() {
    this.setVisible([...this.nodes.keys()], true)
  }

  setLocked(ids: string[], locked: boolean) {
    for (const id of ids) {
      const n = this.nodes.get(id)
      if (n) n.locked = locked
    }
    this.updatePivot()
    this.events.emit('change', undefined)
  }

  /** Set a node's local transform from inspector fields (rotation in degrees, XYZ). */
  setTransform(id: string, t: { position?: number[]; rotationDeg?: number[]; scale?: number[] }) {
    const n = this.nodes.get(id)
    if (!n || n.locked) return
    const before = this.capture()
    if (t.position) n.object.position.fromArray(t.position)
    if (t.rotationDeg) n.object.rotation.set(...(t.rotationDeg.map(THREE.MathUtils.degToRad) as [number, number, number]))
    if (t.scale) n.object.scale.fromArray(t.scale.map((v) => (Math.abs(v) < 1e-4 ? 1e-4 : v)))
    n.object.updateMatrixWorld(true)
    this.commit('edit transform', before)
  }

  rename(id: string, name: string) {
    const n = this.nodes.get(id)
    if (n) n.name = n.object.name = name
    this.events.emit('change', undefined)
  }

  deleteNodes(ids: string[]) {
    const live = ids.filter((id) => this.nodes.has(id))
    if (!live.length) return
    const before = this.capture()
    for (const id of live) {
      const n = this.nodes.get(id)!
      // deleting a group deletes its subtree
      for (const d of this.nodes.values()) if (this.isAncestorOrSelf(id, d.id)) this.setDeleted(d, true)
      n.deleted = true
    }
    if (this.isolateId && this.nodes.get(this.isolateId)?.deleted) this.isolateId = null
    this.selection = []
    this.syncSelectionHelpers()
    this.commit(`delete ${live.length}`, before)
    this.events.emit('selection', this.selection)
  }

  duplicate(ids: string[]): string[] {
    const live = this.manipulable(ids)
    if (!live.length) return []
    const before = this.capture()
    const created: string[] = []
    const offset = new THREE.Vector3(0.12, 0, 0.12)
    for (const id of live) {
      const src = this.nodes.get(id)!
      const made = this.cloneSubtree(src, src.parentId)
      const obj = made.object
      obj.updateMatrixWorld(true)
      const w = obj.matrixWorld.clone().premultiply(new THREE.Matrix4().makeTranslation(offset.x, offset.y, offset.z))
      this.setWorldMatrix(obj, w)
      obj.updateMatrix()
      made.original = obj.matrix.clone()
      created.push(made.id)
    }
    this.commit(`duplicate ${live.length}`, before)
    this.select(created)
    return created
  }

  private cloneSubtree(src: VeronicaNode, parentId: string | null): VeronicaNode {
    const id = `${src.id}~${++this.copyCounter}`
    const g = new THREE.Group()
    g.name = `${src.name} copy`
    src.object.updateMatrix()
    src.object.matrix.decompose(g.position, g.quaternion, g.scale)
    const node: VeronicaNode = { ...src, id, name: `${src.name} copy`, object: g, parentId, original: src.original.clone(), visible: true, deleted: false, duplicateOf: src.duplicateOf ?? src.id, surface: undefined, completion: undefined, points: undefined }
    for (const key of ['surface', 'completion'] as const) {
      const m = src[key]
      if (m) {
        const c = new THREE.Mesh(m.geometry, m.material.clone())
        c.visible = m.visible
        node[key] = c
        g.add(c)
      }
    }
    if (src.points) {
      node.points = new THREE.Points(src.points.geometry, src.points.material.clone())
      g.add(node.points)
    }
    g.traverse((o) => (o.userData.nodeId = id))
    const parent = parentId ? this.nodes.get(parentId)!.object : this.world
    parent.add(g)
    this.nodes.set(id, node)
    for (const child of this.children(src.id)) this.cloneSubtree(child, id)
    return node
  }

  resetTransform(ids: string[], part: 'all' | 'position' | 'rotation' | 'scale' = 'all') {
    const before = this.capture()
    for (const id of ids) {
      const n = this.nodes.get(id)
      if (!n || n.locked) continue
      const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3()
      n.original.decompose(p, q, s)
      if (n.parentId === null || part !== 'all') {
        if (part === 'all' || part === 'position') n.object.position.copy(p)
        if (part === 'all' || part === 'rotation') n.object.quaternion.copy(q)
        if (part === 'all' || part === 'scale') n.object.scale.copy(s)
      } else {
        // inside a group: restore the original *world* placement
        const parent = this.nodes.get(n.parentId)!.object
        this.world.updateMatrixWorld(true)
        this.setWorldMatrix(n.object, new THREE.Matrix4().multiplyMatrices(this.world.matrixWorld, n.original))
        void parent
      }
    }
    this.scene.updateMatrixWorld(true)
    this.commit(`reset ${part}`, before)
  }

  /** Back to exactly what the reconstruction produced. */
  restoreOriginal() {
    const before = this.capture()
    for (const n of this.nodes.values()) {
      if (n.duplicateOf || n.kind === 'group') {
        this.setDeleted(n, true)
        continue
      }
      n.parentId = null
      n.visible = true
      n.deleted = false
      if (n.object.parent !== this.world) this.world.add(n.object)
      n.original.decompose(n.object.position, n.object.quaternion, n.object.scale)
    }
    this.world.matrix.copy(this.worldOriginal).decompose(this.world.position, this.world.quaternion, this.world.scale)
    this.isolateId = null
    this.selection = []
    this.syncSelectionHelpers()
    this.scene.updateMatrixWorld(true)
    this.commit('restore original scene', before)
    this.events.emit('selection', this.selection)
  }

  group(ids: string[]): string | null {
    const live = this.manipulable(ids).filter((id) => this.nodes.get(id)!.parentId === null)
    if (live.length < 2) return null
    const before = this.capture()
    const id = `group-${++this.groupCounter}`
    const g = new THREE.Group()
    g.name = `Group ${this.groupCounter}`
    const c = this.selectionCenter(live)!
    this.world.updateMatrixWorld(true)
    g.position.copy(this.world.worldToLocal(c.clone()))
    this.world.add(g)
    g.updateMatrixWorld(true)
    g.updateMatrix()
    for (const cid of live) {
      g.attach(this.nodes.get(cid)!.object)
      this.nodes.get(cid)!.parentId = id
    }
    g.userData.nodeId = id
    this.nodes.set(id, { id, name: g.name, label: 'group', kind: 'group', object: g, meta: null, parentId: null, original: g.matrix.clone(), visible: true, locked: false, deleted: false })
    this.commit('group', before)
    this.select([id])
    return id
  }

  ungroup(id: string) {
    const n = this.nodes.get(id)
    if (!n || n.kind !== 'group') return
    const before = this.capture()
    const kids = this.children(id)
    for (const k of kids) {
      this.world.attach(k.object)
      k.parentId = null
    }
    this.setDeleted(n, true)
    this.commit('ungroup', before)
    this.select(kids.map((k) => k.id))
  }

  isolate(id: string | null) {
    this.isolateId = this.isolateId === id ? null : id
    this.applyDisplay()
    if (this.isolateId) this.frame([this.isolateId])
    this.events.emit('change', undefined)
  }

  /** Apply a world-space delta to the whole reconstructed scene (scene-level ops). */
  transformScene(delta: THREE.Matrix4, label = 'transform scene') {
    const before = this.capture()
    this.world.updateMatrix()
    this.world.matrix.premultiply(delta).decompose(this.world.position, this.world.quaternion, this.world.scale)
    this.world.updateMatrixWorld(true)
    this.updatePivot()
    this.history.push(label, before, this.capture())
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  /** Continuous scene manipulation (gestures) without a history entry per frame. */
  private sceneSession: { before: Snapshot; m0: THREE.Matrix4 } | null = null
  beginSceneManipulation() {
    this.world.updateMatrix()
    this.sceneSession = { before: this.capture(), m0: this.world.matrix.clone() }
  }
  applySceneManipulation(delta: THREE.Matrix4) {
    if (!this.sceneSession) return
    this.world.matrix.multiplyMatrices(delta, this.sceneSession.m0).decompose(this.world.position, this.world.quaternion, this.world.scale)
    this.world.updateMatrixWorld(true)
    for (const b of this.boxHelpers.values()) b.update()
  }
  endSceneManipulation(commit: boolean) {
    if (!this.sceneSession) return
    const s = this.sceneSession
    this.sceneSession = null
    if (commit) this.history.push('transform scene', s.before, this.capture())
    else this.applySnapshot(s.before)
    this.updatePivot()
    this.events.emit('change', undefined)
    this.events.emit('history', undefined)
  }

  sceneCenter(): THREE.Vector3 {
    const box = new THREE.Box3()
    for (const n of this.liveNodes()) if (n.kind === 'object' && n.object.visible) box.expandByObject(n.object)
    if (box.isEmpty()) return new THREE.Vector3()
    return box.getCenter(new THREE.Vector3())
  }

  // ------------------------------------------------------------------ camera
  setView(v: ViewPreset, animate = true) {
    if (v === 'photo') {
      if (!this.photoPose) return
      const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3()
      this.photoPose.matrix.clone().premultiply(this.world.matrixWorld).decompose(p, q, s)
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q)
      const center = this.sceneCenter()
      const dist = Math.max(0.3, center.clone().sub(p).dot(fwd))
      const target = p.clone().add(fwd.multiplyScalar(dist))
      this.flyTo(p, q, target, this.photoPose.fov, animate)
      return
    }
    const c = this.sceneCenter()
    const box = new THREE.Box3()
    for (const n of this.liveNodes()) if (n.object.visible && n.kind === 'object') box.expandByObject(n.object)
    const r = box.isEmpty() ? 2 : box.getBoundingSphere(new THREE.Sphere()).radius
    const d = Math.max(1, (r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov) / 2)) * 1.1)
    const dirs: Record<Exclude<ViewPreset, 'photo'>, THREE.Vector3> = {
      front: new THREE.Vector3(0, 0.18, 1), back: new THREE.Vector3(0, 0.18, -1), left: new THREE.Vector3(-1, 0.18, 0),
      right: new THREE.Vector3(1, 0.18, 0), top: new THREE.Vector3(0, 1, 0.001), iso: new THREE.Vector3(1, 0.8, 1),
    }
    const pos = c.clone().add(dirs[v].normalize().multiplyScalar(d))
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(pos, c, new THREE.Vector3(0, 1, 0)))
    this.flyTo(pos, q, c, this.camera.fov, animate)
  }

  frame(ids = this.selection) {
    const box = new THREE.Box3()
    for (const id of ids) {
      const n = this.nodes.get(id)
      if (n && !n.deleted) box.expandByObject(n.object)
    }
    if (box.isEmpty()) return
    const s = box.getBoundingSphere(new THREE.Sphere())
    const dir = this.camera.position.clone().sub(this.orbit.target).normalize()
    const d = Math.max(0.25, (s.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov) / 2)) * 1.25)
    const pos = s.center.clone().add(dir.multiplyScalar(d))
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(pos, s.center, new THREE.Vector3(0, 1, 0)))
    this.flyTo(pos, q, s.center, this.camera.fov, true)
  }

  private flyTo(pos: THREE.Vector3, q: THREE.Quaternion, target: THREE.Vector3, fov: number, animate: boolean) {
    if (!animate) {
      this.camera.position.copy(pos)
      this.camera.quaternion.copy(q)
      this.camera.fov = fov
      this.camera.updateProjectionMatrix()
      this.orbit.target.copy(target)
      this.orbit.update()
      return
    }
    this.tween = { from: this.camera.position.clone(), to: pos, qf: this.camera.quaternion.clone(), qt: q, tf: this.orbit.target.clone(), tt: target, fovF: this.camera.fov, fovT: fov, t: 0, dur: 0.7 }
  }

  /** Orbit the camera around the orbit target (radians). Used by gestures. */
  orbitBy(dAzimuth: number, dPolar: number) {
    const off = this.camera.position.clone().sub(this.orbit.target)
    const sph = new THREE.Spherical().setFromVector3(off)
    sph.theta -= dAzimuth
    sph.phi = THREE.MathUtils.clamp(sph.phi - dPolar, 0.05, Math.PI - 0.05)
    off.setFromSpherical(sph)
    this.camera.position.copy(this.orbit.target).add(off)
    this.camera.lookAt(this.orbit.target)
    this.orbit.update()
  }

  /** Pan in screen space; dx, dy are fractions of the viewport. */
  panBy(dx: number, dy: number) {
    const dist = this.camera.position.distanceTo(this.orbit.target)
    const h = 2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0)
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1)
    const delta = right.multiplyScalar(-dx * h * this.camera.aspect).add(up.multiplyScalar(dy * h))
    this.camera.position.add(delta)
    this.orbit.target.add(delta)
    this.orbit.update()
  }

  dollyBy(factor: number) {
    const off = this.camera.position.clone().sub(this.orbit.target)
    const len = THREE.MathUtils.clamp(off.length() * factor, 0.05, 150)
    this.camera.position.copy(this.orbit.target).add(off.setLength(len))
    this.orbit.update()
  }

  // ------------------------------------------------------------------ modal G / R / S
  private pointer = new THREE.Vector2()
  get modalState(): ModalState | null {
    return this.modal ? { op: this.modal.op, axis: this.modal.axis, value: this.modal.value } : null
  }

  startModal(op: GizmoMode) {
    if (this.modal) this.endModal(false)
    if (!this.beginManipulation(this.selection, `${op} (modal)`)) return
    const p = this.manipulationPivot.clone().project(this.camera)
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(this.camera.getWorldDirection(new THREE.Vector3()), this.manipulationPivot)
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const hit = this.raycaster.ray.intersectPlane(plane, new THREE.Vector3())
    this.modal = { op, axis: null, value: '', start: this.pointer.clone(), pivotScreen: new THREE.Vector2(p.x, p.y), planeHit0: hit, mouse: this.pointer.clone() }
    this.orbit.enabled = false
    this.updateModal()
  }

  setModalAxis(a: Axis) {
    if (!this.modal) return
    this.modal.axis = this.modal.axis === a ? null : a
    this.updateModal()
  }

  private updateModal() {
    const md = this.modal
    if (!md || !this.manip) return
    const pivot = this.manipulationPivot
    const fine = this.precision ? 0.2 : 1
    const axisVec = md.axis ? new THREE.Vector3(md.axis === 'x' ? 1 : 0, md.axis === 'y' ? 1 : 0, md.axis === 'z' ? 1 : 0) : null
    if (axisVec && this.space === 'local' && this.manip.ids.length === 1) {
      axisVec.applyQuaternion(this.nodes.get(this.manip.ids[0])!.object.getWorldQuaternion(new THREE.Quaternion()))
    }
    let delta = new THREE.Matrix4()
    if (md.op === 'translate') {
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(this.camera.getWorldDirection(new THREE.Vector3()), pivot)
      this.raycaster.setFromCamera(this.pointer, this.camera)
      const hit = this.raycaster.ray.intersectPlane(plane, new THREE.Vector3())
      if (hit && md.planeHit0) {
        const d = hit.sub(md.planeHit0).multiplyScalar(fine)
        if (axisVec) d.copy(axisVec).multiplyScalar(d.dot(axisVec))
        if (this.snap.enabled) d.divideScalar(this.snap.translate).round().multiplyScalar(this.snap.translate)
        delta.makeTranslation(d.x, d.y, d.z)
        md.value = `${(d.length() * 100).toFixed(1)} cm`
      }
    } else if (md.op === 'rotate') {
      const a0 = Math.atan2(md.start.y - md.pivotScreen.y, (md.start.x - md.pivotScreen.x) * this.camera.aspect)
      const a1 = Math.atan2(this.pointer.y - md.pivotScreen.y, (this.pointer.x - md.pivotScreen.x) * this.camera.aspect)
      let ang = (a1 - a0) * fine
      if (this.snap.enabled) ang = Math.round(ang / THREE.MathUtils.degToRad(this.snap.rotateDeg)) * THREE.MathUtils.degToRad(this.snap.rotateDeg)
      const axis = axisVec ?? this.camera.getWorldDirection(new THREE.Vector3()).negate()
      delta = VeronicaEngine.aboutPoint(pivot, new THREE.Matrix4().makeRotationAxis(axis.normalize(), ang))
      md.value = `${THREE.MathUtils.radToDeg(ang).toFixed(1)}°`
    } else {
      const d0 = Math.max(1e-3, Math.hypot((md.start.x - md.pivotScreen.x) * this.camera.aspect, md.start.y - md.pivotScreen.y))
      const d1 = Math.hypot((this.pointer.x - md.pivotScreen.x) * this.camera.aspect, this.pointer.y - md.pivotScreen.y)
      let f = 1 + (d1 / d0 - 1) * fine
      if (this.snap.enabled) f = Math.max(this.snap.scale, Math.round(f / this.snap.scale) * this.snap.scale)
      f = Math.max(0.01, f)
      delta = this.scaleDelta(pivot, md.axis, f)
      md.value = `×${f.toFixed(3)}`
    }
    this.applyManipulation(delta)
    this.events.emit('modal', this.modalState)
  }

  /** Scale about a pivot, optionally along one axis of the pivot's frame (no shear). */
  scaleDelta(pivot: THREE.Vector3, axis: Axis | null, f: number): THREE.Matrix4 {
    const s = new THREE.Vector3(axis && axis !== 'x' ? 1 : f, axis && axis !== 'y' ? 1 : f, axis && axis !== 'z' ? 1 : f)
    const q = this.space === 'local' && this.manip?.ids.length === 1 ? this.nodes.get(this.manip.ids[0])!.object.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion()
    const R = new THREE.Matrix4().makeRotationFromQuaternion(q)
    const M = R.clone().multiply(new THREE.Matrix4().makeScale(s.x, s.y, s.z)).multiply(R.clone().invert())
    return VeronicaEngine.aboutPoint(pivot, M)
  }

  endModal(commit: boolean) {
    if (!this.modal) return
    this.modal = null
    this.orbit.enabled = true
    this.endManipulation(commit)
    this.events.emit('modal', null)
  }

  get inModal() { return this.modal !== null }

  // ------------------------------------------------------------------ pointer (mouse / trackpad)
  private bindPointer() {
    const el = this.renderer.domElement
    let down: { x: number; y: number; t: number; button: number } | null = null
    const toNdc = (e: PointerEvent | MouseEvent) => {
      const r = el.getBoundingClientRect()
      return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    }
    el.addEventListener('pointermove', (e) => {
      this.pointer.copy(toNdc(e))
      if (this.modal) {
        this.updateModal()
        return
      }
      if (!down && !this.gizmo.dragging) {
        const h = this.pick(this.pointer)
        this.setHovered(h ? this.selectable(h.id) : null)
      }
    })
    el.addEventListener('pointerdown', (e) => {
      if (this.modal) {
        e.preventDefault()
        e.stopPropagation()
        this.endModal(e.button === 0)
        return
      }
      down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button }
    }, { capture: true })
    el.addEventListener('pointerup', (e) => {
      const d = down
      down = null
      if (!d || d.button !== 0 || this.gizmo.dragging) return
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return // was an orbit drag
      if ((this.gizmo as unknown as { axis: string | null }).axis) return // clicked the gizmo
      const h = this.pick(toNdc(e))
      const mode = e.shiftKey || e.metaKey || e.ctrlKey ? 'toggle' : 'set'
      if (h) this.select([this.selectable(h.id, e.altKey)], mode)
      else if (mode === 'set') this.select([])
    })
    el.addEventListener('dblclick', (e) => {
      const h = this.pick(toNdc(e))
      if (h) this.frame([this.selectable(h.id, e.altKey)])
    })
    el.addEventListener('contextmenu', (e) => e.preventDefault())
  }

  /** Box selection in normalised device coords. */
  boxSelect(a: THREE.Vector2, b: THREE.Vector2, mode: 'set' | 'add') {
    const minX = Math.min(a.x, b.x), maxX = Math.max(a.x, b.x), minY = Math.min(a.y, b.y), maxY = Math.max(a.y, b.y)
    const hits: string[] = []
    for (const n of this.liveNodes()) {
      if (n.parentId || n.locked || !n.object.visible) continue
      const box = new THREE.Box3().setFromObject(n.object)
      if (box.isEmpty()) continue
      const c = box.getCenter(new THREE.Vector3()).project(this.camera)
      if (c.z < 1 && c.x >= minX && c.x <= maxX && c.y >= minY && c.y <= maxY) hits.push(n.id)
    }
    this.select(hits, mode)
  }

  // ------------------------------------------------------------------ capture
  screenshot(): Promise<Blob> {
    const hide = [this.gizmoHelper, ...this.boxHelpers.values()]
    const vis = hide.map((o) => o.visible)
    hide.forEach((o) => (o.visible = false))
    this.renderer.setClearColor(0xdfe2e2, 1)
    this.renderer.render(this.scene, this.camera)
    return new Promise((res, rej) => {
      this.renderer.domElement.toBlob((b) => (b ? res(b) : rej(new Error('capture failed'))), 'image/png')
      this.renderer.setClearColor(0x000000, 0)
      hide.forEach((o, i) => (o.visible = vis[i]))
    })
  }

  startRecording(turntable: boolean) {
    if (this.recording) return
    const stream = this.renderer.domElement.captureStream(60)
    const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((m) => MediaRecorder.isTypeSupported(m)) ?? ''
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 })
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    this.renderer.setClearColor(0xdfe2e2, 1)
    rec.start(250)
    this.recording = { rec, chunks, turntable }
    this.events.emit('change', undefined)
  }

  stopRecording(): Promise<Blob | null> {
    const r = this.recording
    if (!r) return Promise.resolve(null)
    this.recording = null
    this.renderer.setClearColor(0x000000, 0)
    return new Promise((res) => {
      r.rec.onstop = () => res(new Blob(r.chunks, { type: r.rec.mimeType || 'video/webm' }))
      r.rec.stop()
      this.events.emit('change', undefined)
    })
  }

  get isRecording() { return this.recording !== null }

  // ------------------------------------------------------------------ loop
  private tick() {
    const now = performance.now()
    const dt = Math.min(0.1, (now - this.lastTime) / 1000)
    this.lastTime = now
    if (this.tween) {
      const tw = this.tween
      tw.t = Math.min(1, tw.t + dt / tw.dur)
      const k = 1 - Math.pow(1 - tw.t, 3)
      this.camera.position.lerpVectors(tw.from, tw.to, k)
      this.camera.quaternion.slerpQuaternions(tw.qf, tw.qt, k)
      this.camera.fov = THREE.MathUtils.lerp(tw.fovF, tw.fovT, k)
      this.camera.updateProjectionMatrix()
      this.orbit.target.lerpVectors(tw.tf, tw.tt, k)
      if (tw.t >= 1) {
        this.tween = null
        this.orbit.update()
      }
    } else {
      if (this.recording?.turntable) this.orbitBy(dt * 0.35, 0)
      this.orbit.update()
    }
    for (const fn of this.frameHooks) fn(dt)
    for (const b of this.boxHelpers.values()) b.update()
    this.renderer.render(this.scene, this.camera)
    this.fpsEma = this.fpsEma * 0.95 + (1 / Math.max(dt, 1e-3)) * 0.05
    this.statsTimer += dt
    if (this.statsTimer > 0.5) {
      this.statsTimer = 0
      const info = this.renderer.info.render
      this.events.emit('stats', { fps: Math.round(this.fpsEma), triangles: info.triangles, calls: info.calls })
    }
  }

  /** NDC → world ray (for gesture cursors). */
  rayFrom(ndc: THREE.Vector2): THREE.Ray {
    this.raycaster.setFromCamera(ndc, this.camera)
    return this.raycaster.ray.clone()
  }

  worldToNdc(p: THREE.Vector3): THREE.Vector3 {
    return _v.copy(p).project(this.camera).clone()
  }
}
