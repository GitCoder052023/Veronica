/** Snapshot-based undo/redo. Reconstructed scenes hold tens of nodes, so capturing
 *  every node's parent, local matrix and visibility per step is cheap and makes every
 *  operation (transform, duplicate, delete, group) undoable through one code path. */
export interface NodeState {
  parentId: string | null
  matrix: number[] // local matrix, column-major
  visible: boolean
  deleted: boolean
}
export type Snapshot = Map<string, NodeState>

export interface HistoryEntry {
  label: string
  before: Snapshot
  after: Snapshot
  time: number
}

export class History {
  private undoStack: HistoryEntry[] = []
  private redoStack: HistoryEntry[] = []
  constructor(private limit = 200) {}

  push(label: string, before: Snapshot, after: Snapshot) {
    if (snapshotsEqual(before, after)) return
    this.undoStack.push({ label, before, after, time: Date.now() })
    if (this.undoStack.length > this.limit) this.undoStack.shift()
    this.redoStack = []
  }
  undo(): HistoryEntry | undefined {
    const e = this.undoStack.pop()
    if (e) this.redoStack.push(e)
    return e
  }
  redo(): HistoryEntry | undefined {
    const e = this.redoStack.pop()
    if (e) this.undoStack.push(e)
    return e
  }
  clear() {
    this.undoStack = []
    this.redoStack = []
  }
  get entries(): readonly HistoryEntry[] {
    return this.undoStack
  }
  get canUndo() {
    return this.undoStack.length > 0
  }
  get canRedo() {
    return this.redoStack.length > 0
  }
}

export function snapshotsEqual(a: Snapshot, b: Snapshot): boolean {
  if (a.size !== b.size) return false
  for (const [id, s] of a) {
    const t = b.get(id)
    if (!t || t.parentId !== s.parentId || t.visible !== s.visible || t.deleted !== s.deleted) return false
    for (let i = 0; i < 16; i++) if (Math.abs(t.matrix[i] - s.matrix[i]) > 1e-7) return false
  }
  return true
}
