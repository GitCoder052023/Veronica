const P: Record<string, string> = {
  translate: 'M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3',
  rotate: 'M20 12a8 8 0 1 1-2.3-5.6M20 4v4.5h-4.5',
  scale: 'M4 20h7v-7H4zM14 4h6v6M20 4l-9 9',
  magnet: 'M6 4v7a6 6 0 0 0 12 0V4M6 8h4M14 8h4M10 4v7a2 2 0 0 0 4 0V4',
  target: 'M12 3v4M12 17v4M3 12h4M17 12h4M12 12m-5 0a5 5 0 1 0 10 0a5 5 0 1 0-10 0',
  eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0',
  eyeOff: 'M3 3l18 18M10.6 6.1A10 10 0 0 1 12 6c6.5 0 10 6 10 6a17 17 0 0 1-3.2 3.9M6.6 6.6C3.8 8.4 2 12 2 12s3.5 6 10 6c1.7 0 3.2-.4 4.5-1',
  lock: 'M6 11h12v10H6zM8.5 11V7.5a3.5 3.5 0 0 1 7 0V11',
  unlock: 'M6 11h12v10H6zM8.5 11V7.5a3.5 3.5 0 0 1 6.8-1.2',
  cube: 'M12 2l9 5v10l-9 5-9-5V7zM12 12l9-5M12 12v10M12 12L3 7',
  plane: 'M3 17l6-10h12l-6 10z',
  group: 'M3 7h7v7H3zM14 10h7v7h-7zM10 10h4',
  camera: 'M3 7h4l2-3h6l2 3h4v13H3zM12 13.5m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0',
  hand: 'M8 13V5a1.5 1.5 0 0 1 3 0v6M11 11V3.5a1.5 1.5 0 0 1 3 0V11M14 11V5a1.5 1.5 0 0 1 3 0v8M8 13l-1.6-2a1.6 1.6 0 0 0-2.6 1.8L7 18a6 6 0 0 0 5 3h1a5 5 0 0 0 5-5v-5',
  upload: 'M12 16V4M7 9l5-5 5 5M4 16v4h16v-4',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13',
  copy: 'M8 8h12v12H8zM4 16V4h12',
  reset: 'M4 12a8 8 0 1 0 2.3-5.6M4 4v4.5h4.5',
  focus: 'M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5',
  isolate: 'M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0M3 12h3M18 12h3M12 3v3M12 18v3',
  undo: 'M9 14L4 9l5-5M4 9h11a5 5 0 0 1 0 10h-3',
  redo: 'M15 14l5-5-5-5M20 9H9a5 5 0 0 0 0 10h3',
  download: 'M12 4v12M7 11l5 5 5-5M4 20h16',
  image: 'M3 5h18v14H3zM3 16l5-5 4 4 3-3 6 6M15.5 9.5m-1.5 0a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0',
  record: 'M12 12m-6 0a6 6 0 1 0 12 0a6 6 0 1 0-12 0',
  stop: 'M7 7h10v10H7z',
  plus: 'M12 5v14M5 12h14',
  close: 'M6 6l12 12M18 6L6 18',
  chevron: 'M9 6l6 6-6 6',
  layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5',
  photo: 'M3 5h18v14H3zM3 16l5-5 4 4 3-3 6 6',
  spark: 'M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6',
}

export function Icon({ name, size = 16 }: { name: string; size?: number }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={P[name] ?? P.cube} />
    </svg>
  )
}

/** VERONICA mark: a spatial cell (hexagonal projection of a cube) holding a V. */
export function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" width={size} height={size} aria-hidden>
      <path d="M16 3.5l10.8 6.25v12.5L16 28.5 5.2 22.25V9.75z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M10.2 11l5.8 11 5.8-11" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="16" cy="11" r="1.3" fill="currentColor" opacity=".55" />
    </svg>
  )
}
