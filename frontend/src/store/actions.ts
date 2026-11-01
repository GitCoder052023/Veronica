import { api, waitForJob } from '../api/client'
import { runtime, useStudio } from './studio'

const st = () => useStudio.getState()

export async function refreshHealth() {
  try {
    st().set({ health: await api.health(), healthError: null })
  } catch (e) {
    st().set({ healthError: `Reconstruction backend unreachable (${(e as Error).message}). Start it with: cd backend && uv run uvicorn veronica.api:app --port 8787` })
  }
}

export async function refreshScenes() {
  try {
    st().set({ scenes: await api.scenes() })
  } catch {
    /* health banner already explains */
  }
}

export async function openScene(id: string, preserveEdits = false) {
  const engine = runtime.engine
  if (!engine) return
  const manifest = await api.scene(id)
  const preserve = preserveEdits ? engine.capture() : undefined
  st().set({ manifest, sceneId: id, selection: [] })
  await engine.load(manifest, (name) => api.fileUrl(id, name), preserve)
  localStorage.setItem('veronica.lastScene', id)
  return manifest
}

export async function reconstruct(files: File[], options: { vocabulary?: string[]; depthModel?: string } = {}, extendFrom?: string) {
  try {
    const { job } = extendFrom ? await api.extendScene(extendFrom, files) : await api.reconstruct(files, options)
    st().set({ job })
    const final = await waitForJob(job.id, (j) => st().set({ job: j }))
    if (final.status === 'done' && final.scene_id) {
      await openScene(final.scene_id)
      await refreshScenes()
      const m = st().manifest
      st().notify(`Reconstructed ${m?.nodes.filter((n) => n.kind === 'object').length ?? 0} objects`)
      setTimeout(() => st().set({ job: null }), 1500)
    } else if (final.status === 'error') st().notify(final.error ?? 'Reconstruction failed', 'error')
  } catch (e) {
    st().notify((e as Error).message, 'error')
    st().set({ job: null })
  }
}

/** Add an object the detector missed, then reload while keeping the user's edits. */
export async function addObject(body: { points?: [number, number][]; text?: string; label?: string }) {
  const id = st().sceneId
  if (!id) return
  try {
    const { job } = await api.addObject(id, body)
    st().set({ job })
    const final = await waitForJob(job.id, (j) => st().set({ job: j }))
    if (final.status === 'done') {
      const m = await openScene(id, true)
      const added = m?.added ?? []
      if (added.length) runtime.engine?.select(added.filter((a) => runtime.engine?.nodes.has(a)))
      st().notify(added.length ? `Added ${added.length} object${added.length > 1 ? 's' : ''}` : 'Scene rebuilt')
    } else st().notify(final.error ?? 'Could not add object', 'error')
  } catch (e) {
    st().notify((e as Error).message, 'error')
  } finally {
    setTimeout(() => st().set({ job: null }), 800)
  }
}

/** Generate unseen sides on the completer worker (local Mac or CUDA), then reload keeping the user's edits. */
export async function generateHidden(ids?: string[], seed = 1) {
  const id = st().sceneId
  if (!id) return
  try {
    const { job } = await api.generate(id, { ids, seed })
    st().set({ job })
    const final = await waitForJob(job.id, (j) => st().set({ job: j }))
    if (final.status === 'done') {
      await openScene(id, true)
      const skipped = final.message.startsWith('Generated') ? ` — ${final.message}` : ''
      st().notify(`Hidden side generated${skipped}`)
    } else st().notify(final.error ?? 'Generation failed', 'error')
  } catch (e) {
    st().notify((e as Error).message, 'error')
  } finally {
    setTimeout(() => st().set({ job: null }), 800)
  }
}

export async function revertHidden(oid: string) {
  const id = st().sceneId
  if (!id) return
  try {
    await api.revertGenerated(id, oid)
    await openScene(id, true)
    st().notify('Reverted to the symmetry prior')
  } catch (e) {
    st().notify((e as Error).message, 'error')
  }
}

/** Re-reconstruct the current scene from its photo(s) plus new ones from other angles:
 *  hidden sides then come from real observations instead of priors. */
export function addAngles(files: File[]) {
  const id = st().sceneId
  if (id && files.length) return reconstruct(files, {}, id)
}
