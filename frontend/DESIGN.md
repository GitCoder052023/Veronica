# VERONICA studio: design

The studio is a tool people work in for long stretches, at a desk, in a lit room (hand
tracking needs light). It is light, quiet and dense, and it spends its colour on one idea.

## The idea

VERONICA's promise is honesty about what a photo can and cannot show. The UI borrows the
convention draftsmen use for exactly this:

- **Seen in a photo** is drawn in **ink** (`--seen`, #15181a).
- **Estimated** (hidden sides, wall behind objects) is **non-photo blue**: the pencil
  colour for construction lines that are not part of the final drawing. `--estimated`
  (#1d6aa8) for text and marks, `--estimated-pale` (#a9d6ea) for hatches.

The same pair appears everywhere the distinction does: the brand mark (a box whose hidden
edges are dashed blue), the seen/estimated bar in the object list, the inspector's
coverage bar, the viewport legend, the hatch on estimated surfaces in Photo shading, and
the blue of estimated geometry in Coverage shading.

**Amber** (`--acting`, #e39a12) is the only other hue. It marks what you are acting on in
3D: the selection outline, box select and the hand cursors. It never appears in the panels.

## Structure

One tinted frame (`--frame`) holds the top bar and both side panels. The viewport is a
sheet inset in it: a 1px edge, 14px corners, no shadow. Panels are sections divided by
1px rules, not boxes of cards. Things that float over the viewport (tools, views, legend,
hand control, dialogs) are opaque white with a small shadow, because they really float.
Nothing uses gradients, glass or glows.

| Token | Value |
|---|---|
| Panels | 272px and 304px (240 and 280 under 1280px); stacked under 900px |
| Spacing | 16px section padding; 6, 8 and 10px inside |
| Radii | sheet 14, floating cards 8, controls 6, chips 4 |
| Controls | 30px tall (26px small); one near-black primary per area |
| Selection in lists | a neutral grey pill (`--active`), never a colour |

## Type

IBM Plex Sans throughout, self-hosted from `@fontsource`; IBM Plex Mono only inside
`<kbd>` and `<code>`. Sizes: 12 (meta), 13 (body, labels), 16 (the scene name, the page's
h1), 22 (the lead figure: an object's measured size, the scene's seen share). Sentence
case everywhere, no tracked capitals. Every figure that can change is set in tabular
figures (`.tnum`).

## Words

The interface says *seen* and *estimated*, not observed/inferred; *Photo* for the
source image; *Coverage* for the shading that shows the split. Buttons name what they do
("Choose photos to reconstruct", "Go back to the symmetry estimate"). Meta lines are
sentences joined with commas, not strings of middle dots.

## Skills used

Built with the project skills in `.claude/skills/`: Anthropic's `frontend-design`,
`design-taste-frontend` (its app-screen rules and the *framed sheet* style) and
`impeccable` (its craft floor). Their helper scripts were not run.
