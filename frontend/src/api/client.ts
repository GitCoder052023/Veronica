import type { Health, Job, Manifest, SceneSummary } from './types'

const BASE = '/api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not json */ }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetch(`${BASE}/health`).then((r) => json<Health>(r)),
  scenes: () => fetch(`${BASE}/scenes`).then((r) => json<SceneSummary[]>(r)),
  scene: (id: string) => fetch(`${BASE}/scenes/${id}`).then((r) => json<Manifest>(r)),
  fileUrl: (id: string, name: string) => `${BASE}/scenes/${id}/files/${name}`,
  deleteScene: (id: string) => fetch(`${BASE}/scenes/${id}`, { method: 'DELETE' }).then((r) => json<unknown>(r)),
  job: (id: string) => fetch(`${BASE}/jobs/${id}`).then((r) => json<Job>(r)),
  cancel: (id: string) => fetch(`${BASE}/jobs/${id}/cancel`, { method: 'POST' }).then((r) => json<unknown>(r)),
  reconstruct(files: File[], options: { vocabulary?: string[]; depthModel?: string } = {}) {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    fd.append('options', JSON.stringify(options))
    return fetch(`${BASE}/scenes`, { method: 'POST', body: fd }).then((r) => json<{ job: Job }>(r))
  },
  extendScene(id: string, files: File[]) {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    return fetch(`${BASE}/scenes/${id}/extend`, { method: 'POST', body: fd }).then((r) => json<{ job: Job }>(r))
  },
  generate(id: string, body: { ids?: string[]; seed?: number; steps?: number }) {
    return fetch(`${BASE}/scenes/${id}/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then((r) => json<{ job: Job }>(r))
  },
  revertGenerated: (id: string, oid: string) => fetch(`${BASE}/scenes/${id}/objects/${oid}/generate`, { method: 'DELETE' }).then((r) => json<unknown>(r)),
  addObject(id: string, body: { points?: [number, number][]; pointLabels?: number[]; text?: string; label?: string }) {
    return fetch(`${BASE}/scenes/${id}/objects`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then((r) => json<{ job: Job }>(r))
  },
}

/** Poll a job until it settles. `onUpdate` sees every intermediate state. */
export async function waitForJob(id: string, onUpdate: (j: Job) => void, signal?: AbortSignal): Promise<Job> {
  for (;;) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
    const j = await api.job(id)
    onUpdate(j)
    if (j.status === 'done' || j.status === 'error' || j.status === 'cancelled') return j
    await new Promise((r) => setTimeout(r, 400))
  }
}
