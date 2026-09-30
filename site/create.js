// Turns one equirectangular 360° photo into a walkable 3D scene, entirely in
// the browser: Depth Anything V2 (transformers.js) estimates relative depth,
// which displaces a sphere mesh textured with the full-resolution photo.
// Per-vertex confidence from local depth consistency drives the evidence lens.
import * as THREE from 'three';

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';
const MODEL_ID = 'onnx-community/depth-anything-v2-small';
const GRID_WIDTH = 1024;
const NEAR = 0.8, FAR = 9;

let depthPipeline = null, RawImageClass = null;

async function supportsF16() {
  try { return !!(await navigator.gpu?.requestAdapter())?.features.has('shader-f16'); }
  catch { return false; }
}

// Candidate runtimes, fastest first. `?depth=webgpu-fp32` or `?depth=wasm` forces one.
async function runtimeCandidates() {
  const forced = new URLSearchParams(location.search).get('depth');
  const all = {'webgpu-fp16': ['webgpu', 'fp16', 'WebGPU'], 'webgpu-fp32': ['webgpu', 'fp32', 'WebGPU'], wasm: ['wasm', 'q8', 'WebAssembly']};
  if (all[forced]) return [all[forced]];
  return (await supportsF16()) ? [all['webgpu-fp16'], all.wasm] : [all.wasm];
}

async function loadDepthModel(status, skipDevice) {
  if (depthPipeline && depthPipeline.deviceLabel !== skipDevice) return depthPipeline;
  status('Loading the depth AI (first time only, cached afterwards)…', 0);
  const {pipeline, RawImage} = await import(TRANSFORMERS_URL);
  RawImageClass = RawImage;
  let lastError;
  for (const [device, dtype, label] of await runtimeCandidates()) {
    if (label === skipDevice) continue;
    const files = new Map();
    try {
      depthPipeline = await pipeline('depth-estimation', MODEL_ID, {
        device, dtype,
        progress_callback: event => {
          if (event.status === 'progress' && event.total) {
            files.set(event.file, [event.loaded, event.total]);
            let loaded = 0, total = 0;
            for (const [l, t] of files.values()) { loaded += l; total += t; }
            status(`Downloading depth AI · ${Math.round(100 * loaded / total)}% of ${(total / 2 ** 20).toFixed(0)} MB`, loaded / total);
          }
        },
      });
      depthPipeline.deviceLabel = label;
      return depthPipeline;
    } catch (error) { console.warn(`Depth model unavailable on ${label}`, error); lastError = error; }
  }
  throw lastError ?? new Error('No runtime is available for the depth AI in this browser.');
}

const MAX_WIDTH = 4096;

// Decodes any upload at most maxWidth wide (so 8K–16K files stay cheap),
// then returns a 2:1 equirectangular canvas. Near-2:1 images are stretched;
// wider ones (partial vertical coverage) are centred with empty bands.
async function normalizePanorama(source, status, maxWidth) {
  status('Preparing your photo…', null);
  const blob = await (await fetch(source)).blob();
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, {resizeWidth: maxWidth, resizeQuality: 'high'});
  } catch {
    try { bitmap = await createImageBitmap(blob); }
    catch { throw new Error('This file could not be read as an image (JPEG, PNG or WebP work best).'); }
  }
  const aspect = bitmap.width / bitmap.height;
  if (aspect < 1.8) {
    bitmap.close?.();
    throw new Error('This does not look like a 360° panorama. Use an equirectangular image about twice as wide as it is tall.');
  }
  const width = Math.min(maxWidth, bitmap.width) & ~1, height = width / 2;
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingQuality = 'high';
  if (aspect <= 2.2) ctx.drawImage(bitmap, 0, 0, width, height);
  else {
    const band = Math.round(width / aspect);
    ctx.drawImage(bitmap, 0, Math.round((height - band) / 2), width, band);
  }
  bitmap.close?.();
  return canvas;
}

// Scales the panorama to the model size, with wrapped strips on both sides
// so the model sees across the 0°/360° seam.
function paddedCanvas(image, width, height, pad) {
  const canvas = document.createElement('canvas');
  canvas.width = width + 2 * pad; canvas.height = height;
  const ctx = canvas.getContext('2d');
  const sw = image.width, sh = image.height, strip = sw * pad / width;
  ctx.drawImage(image, 0, 0, sw, sh, pad, 0, width, height);
  ctx.drawImage(image, sw - strip, 0, strip, sh, 0, 0, pad, height);
  ctx.drawImage(image, 0, 0, strip, sh, pad + width, 0, pad, height);
  return canvas;
}

function percentile(values, p) {
  const sample = [];
  const step = Math.max(1, Math.floor(values.length / 20000));
  for (let i = 0; i < values.length; i += step) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(p * sample.length))];
}

const VERTEX_SHADER = `
  attribute float confidence;
  varying vec2 vUv;
  varying float vConfidence;
  void main() {
    vUv = uv;
    vConfidence = confidence;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const FRAGMENT_SHADER = `
  uniform sampler2D map;
  uniform float lens;
  uniform float minConfidence;
  varying vec2 vUv;
  varying float vConfidence;
  vec3 evidenceRamp(float t) {
    vec3 a = vec3(0.878, 0.267, 0.361), b = vec3(0.941, 0.541, 0.294), c = vec3(0.949, 0.820, 0.361);
    vec3 d = vec3(0.345, 0.776, 0.635), e = vec3(0.227, 0.651, 0.851);
    if (t < 0.3) return mix(a, b, t / 0.3);
    if (t < 0.55) return mix(b, c, (t - 0.3) / 0.25);
    if (t < 0.8) return mix(c, d, (t - 0.55) / 0.25);
    return mix(d, e, (t - 0.8) / 0.2);
  }
  void main() {
    if (vConfidence < minConfidence) discard;
    // The photo texture holds display-space colors, written out unchanged.
    vec3 color = texture2D(map, vUv).rgb;
    color = mix(color, evidenceRamp(clamp((vConfidence - 0.08) / 0.42, 0.0, 1.0)), lens);
    gl_FragColor = vec4(color, 1.0);
  }
`;

export async function buildPhotoScene(source, status, {maxTextureSize = 4096, anisotropy = 1} = {}) {
  const image = await normalizePanorama(source, status, Math.min(MAX_WIDTH, maxTextureSize));
  const modelWidth = 1036, modelHeight = 518, pad = 126;
  const padded = paddedCanvas(image, modelWidth, modelHeight, pad);
  let depth = await loadDepthModel(status), predicted, started;
  for (;;) {
    status(`Estimating depth on ${depth.deviceLabel}…`, null);
    started = performance.now();
    try { ({predicted_depth: predicted} = await depth(RawImageClass.fromCanvas(padded))); break; }
    catch (error) {
      // Some GPUs expose WebGPU but fail at inference; fall back to the CPU runtime.
      if (depth.deviceLabel !== 'WebGPU') throw error;
      console.warn('WebGPU depth inference failed, retrying on WebAssembly', error);
      depth = await loadDepthModel(status, 'WebGPU');
    }
  }
  const inferenceSeconds = (performance.now() - started) / 1000;
  const [dh, dw] = predicted.dims.slice(-2);
  const disparity = predicted.data;

  status('Building the 3D scene…', null);
  await new Promise(requestAnimationFrame);
  const gw = GRID_WIDTH, gh = gw / 2;
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = gw; maskCanvas.height = gh;
  const maskContext = maskCanvas.getContext('2d', {willReadFrequently: true});
  maskContext.drawImage(image, 0, 0, gw, gh);
  const rgb = maskContext.getImageData(0, 0, gw, gh).data;

  // Bilinear sample of the model output at grid pixel centres, skipping padding.
  const sx = (modelWidth / (modelWidth + 2 * pad)) * dw / gw, ox = pad / (modelWidth + 2 * pad) * dw;
  const sy = dh / gh;
  const disp = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    const fy = Math.min(dh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(dh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < gw; x++) {
      const fx = Math.min(dw - 1, Math.max(0, ox + (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(dw - 1, x0 + 1), tx = fx - x0;
      const top = disparity[y0 * dw + x0] * (1 - tx) + disparity[y0 * dw + x1] * tx;
      const bottom = disparity[y1 * dw + x0] * (1 - tx) + disparity[y1 * dw + x1] * tx;
      disp[y * gw + x] = top * (1 - ty) + bottom * ty;
    }
  }
  // Relative disparity to metric-like depth for an indoor room.
  const lo = percentile(disp, 0.02), hi = percentile(disp, 0.98);
  const logDepth = new Float32Array(gw * gh);
  for (let i = 0; i < disp.length; i++) {
    const t = Math.min(1, Math.max(0, (disp[i] - lo) / Math.max(1e-6, hi - lo)));
    logDepth[i] = -Math.log(1 / FAR + t * (1 / NEAR - 1 / FAR));
  }
  // Confidence: depth is least reliable at discontinuities and near the poles.
  // A wide stencil plus box smoothing measures edges, not per-pixel noise.
  const step = 3, radius = 4;
  const raw = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    const yu = Math.max(0, y - step), yd = Math.min(gh - 1, y + step);
    for (let x = 0; x < gw; x++) {
      const xl = (x - step + gw) % gw, xr = (x + step) % gw;
      raw[y * gw + x] = Math.abs(logDepth[y * gw + xr] - logDepth[y * gw + xl])
        + Math.abs(logDepth[yd * gw + x] - logDepth[yu * gw + x]);
    }
  }
  const horizontal = new Float32Array(gw * gh), gradient = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) sum += raw[y * gw + (x + k + gw) % gw];
      horizontal[y * gw + x] = sum / (2 * radius + 1);
    }
  }
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) sum += horizontal[Math.min(gh - 1, Math.max(0, y + k)) * gw + x];
      gradient[y * gw + x] = sum / (2 * radius + 1);
    }
  }
  const gradientScale = 2.5 * Math.max(1e-4, percentile(gradient, 0.5));

  // Vertices sit on pixel corners; each takes the log-depth averaged over the
  // (up to four) pixels it touches. Pole rows collapse to one mean depth.
  const vw = gw + 1, vh = gh + 1;
  const positions = new Float32Array(vw * vh * 3), uvs = new Float32Array(vw * vh * 2);
  const confidence = new Float32Array(vw * vh), evidence = new Uint8Array(vw * vh);
  const empty = new Uint8Array(vw * vh);
  const rowMean = row => { let sum = 0; for (let x = 0; x < gw; x++) sum += logDepth[row * gw + x]; return sum / gw; };
  const topDepth = rowMean(0), bottomDepth = rowMean(gh - 1);
  for (let vy = 0; vy < vh; vy++) {
    const lat = (0.5 - vy / gh) * Math.PI, cosLat = Math.cos(lat), sinLat = Math.sin(lat);
    const rows = [Math.max(0, vy - 1), Math.min(gh - 1, vy)];
    for (let vx = 0; vx < vw; vx++) {
      const lon = (vx / gw - 0.5) * 2 * Math.PI;
      const cols = [(vx - 1 + gw) % gw, vx % gw];
      let logSum = 0, edgeSum = 0, black = 0;
      for (const r of rows) for (const c of cols) {
        const i = r * gw + c;
        logSum += logDepth[i]; edgeSum += gradient[i];
        // Pure black marks missing capture (e.g. masked regions), not a surface.
        if (rgb[4 * i] + rgb[4 * i + 1] + rgb[4 * i + 2] < 6) black++;
      }
      const d = Math.exp(vy === 0 ? topDepth : vy === gh ? bottomDepth : logSum / 4);
      const v = vy * vw + vx;
      positions[3 * v] = Math.sin(lon) * cosLat * d;
      positions[3 * v + 1] = sinLat * d;
      positions[3 * v + 2] = -Math.cos(lon) * cosLat * d;
      uvs[2 * v] = vx / gw; uvs[2 * v + 1] = 1 - vy / gh;
      const edge = edgeSum / 4 / gradientScale;
      const c = 0.05 + 0.55 * (1 / (1 + edge * edge)) * (0.55 + 0.45 * cosLat);
      confidence[v] = c;
      evidence[v] = Math.round(255 * c);
      empty[v] = black === 4 ? 1 : 0;
    }
  }
  const indices = [];
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const a = y * vw + x, b = a + 1, c = a + vw, d = c + 1;
      if (empty[a] && empty[b] && empty[c] && empty[d]) continue;
      indices.push(a, c, b, b, c, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('confidence', new THREE.BufferAttribute(confidence, 1));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));

  const texture = new THREE.CanvasTexture(image);
  texture.wrapS = THREE.RepeatWrapping;
  texture.anisotropy = anisotropy;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  const material = new THREE.ShaderMaterial({
    uniforms: {map: {value: texture}, lens: {value: 0}, minConfidence: {value: 0}},
    vertexShader: VERTEX_SHADER, fragmentShader: FRAGMENT_SHADER, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return {mesh, evidence, inferenceSeconds, device: depth.deviceLabel,
    vertices: vw * vh, textureWidth: image.width};
}
