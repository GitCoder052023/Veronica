export type Vec3 = [number, number, number]

export interface MeshRep { file: string; vertices: number; faces: number; bytes: number; texture: string | null }
export interface PointsRep { file: string; points: number; bytes: number }

export interface ManifestNode {
  id: string
  name: string
  label: string
  kind: 'object' | 'structure' | 'background'
  flat?: boolean
  transform: { position: Vec3; quaternion: [number, number, number, number]; scale: Vec3 }
  bounds: { min: Vec3; max: Vec3 }
  dimensions: Vec3
  representations: { surface?: MeshRep; completion?: MeshRep; points?: PointsRep }
  detection?: { score: number; maskQuality: number; source: string; bbox: [number, number, number, number]; pixels: number; maskIndex: number }
  provenance: { observedFraction: number; completion: string; completionParams?: Record<string, unknown>; meanDepth?: number }
  relations?: { supportedBy: string | null }
  plane?: { normal: Vec3; inferredPixels: number; observedPixels: number }
  view?: number
}

export interface Manifest {
  version: number
  id: string
  created: string
  mode: 'single-view' | 'multi-view'
  source: { name: string; width: number; height: number; views: number }
  camera: {
    intrinsics: { fx: number; fy: number; cx: number; cy: number; width: number; height: number }
    hfov: number
    vfov: number
    focalSource: string
    worldFromCamera: number[][]
  }
  cameras?: { worldFromCamera: number[][]; vfov: number; image: string }[]
  world: { up: Vec3; units: string; upSource: string; floorY: number | null; scaleNote: string }
  textures: Record<string, string>
  nodes: ManifestNode[]
  pipeline: { models: { key: string; role: string; id: string; license: string }[]; timings: Record<string, number> }
  uncertainty: { notes: string[] }
  added?: string[]
}

export interface Job {
  id: string
  kind: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  stage: string
  progress: number
  message: string
  log: { t: number; stage: string; msg: string }[]
  scene_id: string | null
  error: string | null
}

export interface SceneSummary { id: string; created: string; name: string; mode: string; objects: number }

export interface Health {
  ok: boolean
  device: string
  loaded: string[]
  depthModel: string
  models: Record<string, { role: string; id: string; license: string; note: string }>
  multiView: { available: boolean; reason?: string; model?: string }
  completer: { configured: boolean; ok: boolean; url?: string; model?: string; device?: string; vram_gb?: number; reason?: string; error?: string }
}
