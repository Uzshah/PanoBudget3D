// Turns one equirectangular 360° photo into Gaussian splats, entirely in the
// browser: Depth Anything V2 (transformers.js) estimates relative depth, each
// pixel becomes a disk-shaped Gaussian on the depth sphere, and a per-splat
// confidence is derived from local depth consistency for the evidence lens.
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

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('This file could not be read as an image.'));
    image.src = source;
  });
}

// Pads the panorama with wrapped strips so the model sees across the seam.
function paddedCanvas(image, width, height, pad) {
  const canvas = document.createElement('canvas');
  canvas.width = width + 2 * pad; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, width, height, pad, 0, width, height);
  ctx.drawImage(image, width - pad, 0, pad, height, 0, 0, pad, height);
  ctx.drawImage(image, 0, 0, pad, height, pad + width, 0, pad, height);
  return canvas;
}

function percentile(values, p) {
  const sample = [];
  const step = Math.max(1, Math.floor(values.length / 20000));
  for (let i = 0; i < values.length; i += step) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(p * sample.length))];
}

function trailingZeros(value, cap) {
  if (value === 0) return cap;
  let count = 0;
  while ((value & 1) === 0 && count < cap) { value >>= 1; count++; }
  return count;
}

// Rotation from the tangent frame (east, north, inward) to a quaternion.
function basisQuaternion(m00, m01, m02, m10, m11, m12, m20, m21, m22, out) {
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    out[3] = 0.25 / s; out[0] = (m21 - m12) * s; out[1] = (m02 - m20) * s; out[2] = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    out[3] = (m21 - m12) / s; out[0] = 0.25 * s; out[1] = (m01 + m10) / s; out[2] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    out[3] = (m02 - m20) / s; out[0] = (m01 + m10) / s; out[1] = 0.25 * s; out[2] = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    out[3] = (m10 - m01) / s; out[0] = (m02 + m20) / s; out[1] = (m12 + m21) / s; out[2] = 0.25 * s;
  }
}

export async function buildSplatsFromPanorama(packedSplats, source, status) {
  const image = await loadImage(source);
  if (Math.abs(image.naturalWidth / image.naturalHeight - 2) > 0.1) {
    throw new Error(`A 360° photo is twice as wide as it is tall; this one is ${image.naturalWidth}×${image.naturalHeight}.`);
  }
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

  status('Building 3D Gaussians…', null);
  await new Promise(requestAnimationFrame);
  const gw = Math.min(GRID_WIDTH, image.naturalWidth), gh = gw / 2;
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = gw; colorCanvas.height = gh;
  const colorContext = colorCanvas.getContext('2d', {willReadFrequently: true});
  colorContext.drawImage(image, 0, 0, gw, gh);
  const rgb = colorContext.getImageData(0, 0, gw, gh).data;

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

  const dLon = 2 * Math.PI / gw, dLat = Math.PI / gh;
  const splats = [];
  for (let y = 0; y < gh; y++) {
    const lat = (0.5 - (y + 0.5) / gh) * Math.PI;
    const cosLat = Math.cos(lat), sinLat = Math.sin(lat);
    const stride = Math.max(1, Math.round(1 / Math.max(cosLat, 0.02)));
    for (let x = 0, column = 0; x < gw; x += stride, column++) {
      const i = y * gw + x;
      // Pure black marks missing capture (e.g. masked regions), not a surface.
      if (rgb[4 * i] + rgb[4 * i + 1] + rgb[4 * i + 2] < 6) continue;
      const level = Math.min(trailingZeros(y, 4), trailingZeros(column, 4));
      splats.push([i, stride, level]);
    }
  }
  // Coarse lattice first, so a partial budget thins the scene evenly.
  splats.sort((a, b) => b[2] - a[2]);

  const center = new THREE.Vector3(), scales = new THREE.Vector3();
  const quaternion = new THREE.Quaternion(), color = new THREE.Color();
  const q = [0, 0, 0, 1];
  const evidence = new Uint8Array(splats.length);
  packedSplats.ensureSplats?.(splats.length);
  for (let k = 0; k < splats.length; k++) {
    const [i, stride] = splats[k];
    const x = i % gw, y = (i - x) / gw;
    const lon = ((x + 0.5 * stride) / gw - 0.5) * 2 * Math.PI;
    const lat = (0.5 - (y + 0.5) / gh) * Math.PI;
    const cosLat = Math.cos(lat), sinLat = Math.sin(lat), cosLon = Math.cos(lon), sinLon = Math.sin(lon);
    const d = Math.exp(logDepth[i]);
    const rx = sinLon * cosLat, ry = sinLat, rz = -cosLon * cosLat;
    center.set(rx * d, ry * d, rz * d);
    // Tangent frame: east, north, inward normal.
    const ex = cosLon, ey = 0, ez = sinLon;
    const nx = -sinLon * sinLat, ny = cosLat, nz = cosLon * sinLat;
    basisQuaternion(ex, nx, -rx, ey, ny, -ry, ez, nz, -rz, q);
    quaternion.set(q[0], q[1], q[2], q[3]).normalize();
    const east = 0.6 * d * dLon * Math.max(cosLat, 0.02) * stride;
    const north = 0.6 * d * dLat;
    scales.set(east, north, 0.15 * Math.min(east, north));
    // Spark stores display-space colors, as in SPZ files.
    color.r = rgb[4 * i] / 255; color.g = rgb[4 * i + 1] / 255; color.b = rgb[4 * i + 2] / 255;
    packedSplats.pushSplat(center, scales, quaternion, 1, color);
    const edge = gradient[i] / gradientScale;
    const confidence = (1 / (1 + edge * edge)) * (0.55 + 0.45 * cosLat);
    evidence[k] = Math.round(255 * (0.05 + 0.55 * confidence));
  }
  return {evidence, inferenceSeconds, device: depth.deviceLabel, splats: splats.length};
}
