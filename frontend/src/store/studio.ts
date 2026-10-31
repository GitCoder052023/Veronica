import { create } from 'zustand'
import type { Health, Job, Manifest, SceneSummary } from '../api/types'
import type { VeronicaEngine } from '../engine/VeronicaEngine'
import type { GestureController, GestureHud } from '../gesture/GestureController'
import type { HandTracker, TrackerStats } from '../gesture/HandTracker'

/** Long-lived imperative singletons; React only mirrors their state. */
export const runtime: { engine: VeronicaEngine | null; gestures: GestureController | null; tracker: HandTracker | null } = {
  engine: null, gestures: null, tracker: null,
}

interface StudioState {
  health: Health | null
  healthError: string | null
  scenes: SceneSummary[]
  manifest: Manifest | null
  sceneId: string | null
  job: Job | null
  loading: { done: number; total: number } | null
  selection: string[]
  hovered: string | null
  rev: number // bump to re-read engine state
  stats: { fps: number; triangles: number; calls: number }
  modal: { op: string; axis: string | null; value: string } | null
  gestureOn: boolean
  gestureHud: GestureHud | null
  trackerStats: TrackerStats | null
  gestureError: string | null
  onboarding: boolean
  sourceOpen: boolean
  toast: { text: string; kind: 'info' | 'error'; id: number } | null
  set: (p: Partial<StudioState>) => void
  bump: () => void
  notify: (text: string, kind?: 'info' | 'error') => void
}

export const useStudio = create<StudioState>((set) => ({
  health: null,
  healthError: null,
  scenes: [],
  manifest: null,
  sceneId: null,
  job: null,
  loading: null,
  selection: [],
  hovered: null,
  rev: 0,
  stats: { fps: 0, triangles: 0, calls: 0 },
  modal: null,
  gestureOn: false,
  gestureHud: null,
  trackerStats: null,
  gestureError: null,
  onboarding: false,
  sourceOpen: false,
  toast: null,
  set: (p) => set(p),
  bump: () => set((s) => ({ rev: s.rev + 1 })),
  notify: (text, kind = 'info') => set({ toast: { text, kind, id: Date.now() } }),
}))
