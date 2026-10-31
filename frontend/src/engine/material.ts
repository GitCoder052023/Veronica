import * as THREE from 'three'

/** How surfaces are shaded. Photo is unlit on purpose: the texture already contains the
 *  scene's real lighting, so adding synthetic light would double-light it. */
export type RenderMode = 'photo' | 'clay' | 'provenance' | 'normals'
export const RENDER_MODES: RenderMode[] = ['photo', 'clay', 'provenance', 'normals']
const MODE_INDEX: Record<RenderMode, number> = { photo: 0, clay: 1, provenance: 2, normals: 3 }

const vert = /* glsl */ `
attribute float observed;
attribute vec3 vcolor;
varying vec3 vColor;
varying vec2 vUv;
varying float vObserved;
varying vec3 vViewPos;
void main() {
  vUv = uv;
  vColor = vcolor;
  vObserved = observed;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewPos = mv.xyz;
  gl_Position = projectionMatrix * mv;
}`

const frag = /* glsl */ `
uniform sampler2D map;
uniform float hasMap;
uniform float hasColor;
uniform int mode;
uniform float discardObserved;
uniform float showInferred;
uniform float inferredStyle;
uniform vec3 highlight;
uniform float highlightAmount;
uniform float opacity;
uniform float time;
varying vec2 vUv;
varying vec3 vColor;
varying float vObserved;
varying vec3 vViewPos;

void main() {
  bool inferred = vObserved < 0.5;
  if (discardObserved > 0.5 && !inferred) discard;
  if (showInferred < 0.5 && inferred) discard;
  vec3 n = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
  vec3 v = normalize(-vViewPos);
  if (dot(n, v) < 0.0) n = -n;
  vec3 tex = hasMap > 0.5 ? texture2D(map, vUv).rgb : hasColor > 0.5 ? pow(vColor, vec3(2.2)) : vec3(0.72);
  vec3 col;
  if (mode == 0) {
    col = tex;
  } else if (mode == 1) {
    float l = 0.28 + 0.62 * max(dot(n, normalize(vec3(0.35, 0.65, 0.68))), 0.0) + 0.18 * max(dot(n, normalize(vec3(-0.6, -0.1, 0.4))), 0.0);
    col = vec3(0.78, 0.8, 0.83) * l;
  } else if (mode == 2) {
    float l = 0.45 + 0.55 * max(dot(n, v), 0.0);
    col = (inferred ? vec3(0.83, 0.42, 1.0) : vec3(0.22, 0.86, 0.78)) * l;
  } else {
    col = n * 0.5 + 0.5;
  }
  if (inferred && mode == 0 && inferredStyle > 0.5) {
    float s = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 9.0));
    col = mix(col * 0.8, vec3(0.72, 0.55, 1.0), 0.18 + 0.1 * s);
  }
  float rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
  col += highlight * highlightAmount * (0.18 + 0.9 * rim);
  gl_FragColor = vec4(col, opacity);
  #include <colorspace_fragment>
}`

export function createSurfaceMaterial(map: THREE.Texture | null, hasColor = false): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: vert,
    fragmentShader: frag,
    side: THREE.DoubleSide,
    uniforms: {
      map: { value: map },
      hasMap: { value: map ? 1 : 0 },
      hasColor: { value: hasColor ? 1 : 0 },
      mode: { value: 0 },
      discardObserved: { value: 0 },
      showInferred: { value: 1 },
      inferredStyle: { value: 1 },
      highlight: { value: new THREE.Color(0x5ee6d0) },
      highlightAmount: { value: 0 },
      opacity: { value: 1 },
      time: { value: 0 },
    },
  })
}

export function setMode(m: THREE.ShaderMaterial, mode: RenderMode) {
  m.uniforms.mode.value = MODE_INDEX[mode]
}

export function createPointsMaterial(): THREE.PointsMaterial {
  return new THREE.PointsMaterial({ size: 0.007, vertexColors: true, sizeAttenuation: true })
}
