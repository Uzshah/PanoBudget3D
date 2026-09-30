import * as THREE from 'three';
import {SparkRenderer, SplatMesh, SparkControls} from '@sparkjsdev/spark';

const $ = id => document.getElementById(id);
const requestedScene = new URLSearchParams(location.search).get('scene') || 'hotel_0';
const sceneKey = Object.hasOwn(window.PANOBUDGET_SCENES, requestedScene) ? requestedScene : 'hotel_0';
const config = window.PANOBUDGET_SCENES[sceneKey];
const storageKey = config.storageKey;
$('scene-picker').value = sceneKey;
$('scene-name').textContent = ` / ${config.label}`;
document.title = `PanoBudget3D · ${config.label} tour`;
$('scene').setAttribute('aria-label', `Interactive C6 ${config.label} Gaussian scene`);
$('scene-picker').onchange = () => {
  const url = new URL(location.href);
  url.search = ''; url.hash = '';
  url.searchParams.set('scene', $('scene-picker').value);
  location.assign(url);
};
const canvas = $('scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11120f);
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({canvas, antialias: false});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight);
const spark = new SparkRenderer({renderer});
scene.add(spark);
const controls = new SparkControls({canvas});
controls.fpsMovement && (controls.fpsMovement.moveSpeed = 0.3);

let ready = false, playing = false, progress = 0, duration = 42, fullCount = 0;
let lastTime = null, frames = 0, lastStat = 0, toastTimer;
const sampleCamera = new THREE.PerspectiveCamera();
function levelPose(source, heading) {
  const position = [source.position[0], -source.position[1], -source.position[2]];
  const r = source.panoRotation;
  const direction = [Math.sin(heading), 0.12, Math.cos(heading)];
  const rotated = r.map(row => row.reduce((sum, value, axis) => sum + value * direction[axis], 0));
  sampleCamera.position.fromArray(position);
  // Recover the panorama's upright direction before applying the scene's X flip.
  sampleCamera.up.set(-r[0][1], r[1][1], r[2][1]);
  sampleCamera.lookAt(position[0] + rotated[0], position[1] - rotated[1], position[2] - rotated[2]);
  return {position, quaternion: sampleCamera.quaternion.toArray(), fov: 75};
}
const trainingCameras = config.cameras;
const defaultStops = trainingCameras.map((pose, index) => levelPose(pose, config.heading + index * Math.PI * 2 / (trainingCameras.length - 1)));
// Finish at the starting capture center, so restarting does not jump across the room.
defaultStops.push(structuredClone(defaultStops[0]));
let stops = structuredClone(defaultStops);
let opening = structuredClone(stops[0]);

function toast(message) {
  $('toast').textContent = message;
  $('toast').classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 4500);
}
function validPose(pose) {
  return pose && Array.isArray(pose.position) && pose.position.length === 3
    && Array.isArray(pose.quaternion) && pose.quaternion.length === 4
    && [...pose.position, ...pose.quaternion].every(Number.isFinite)
    && Math.hypot(...pose.quaternion) > 1e-8
    && Number.isFinite(pose.fov) && pose.fov >= 20 && pose.fov <= 120;
}
function restore(data) {
  if (data.scene !== config.id || !validPose(data.opening)
      || !Array.isArray(data.stops) || data.stops.length < 1 || data.stops.length > 200
      || !data.stops.every(validPose) || !Number.isFinite(data.duration)
      || data.duration < 15 || data.duration > 120) throw new Error(`Invalid tour file. Choose a ${config.label} tour file.`);
  opening = structuredClone(data.opening);
  stops = structuredClone(data.stops);
  duration = data.duration;
}
function record() {
  return {format: 'panobudget-tour-v1', scene: config.id, renderer: 'Spark 2.3.0',
    modelRotation: [Math.PI, 0, 0], opening, stops, duration};
}
function persist() {
  try { localStorage.setItem(storageKey, JSON.stringify(record())); }
  catch { toast('Browser storage unavailable. Export the tour to keep your changes.'); }
  $('stop-count').textContent = `${stops.length} saved camera stops`;
}
try {
  const saved = localStorage.getItem(storageKey);
  if (saved) restore(JSON.parse(saved));
} catch (error) { console.warn('Ignoring invalid saved tour', error); }

function applyPose(pose) {
  camera.position.fromArray(pose.position);
  camera.quaternion.fromArray(pose.quaternion).normalize();
  camera.fov = pose.fov;
  camera.updateProjectionMatrix();
  $('fov').value = String(pose.fov);
  $('fov-label').textContent = `${Math.round(pose.fov)}°`;
}
function capturePose() {
  return {position: camera.position.toArray(), quaternion: camera.quaternion.toArray(), fov: camera.fov};
}
function timeLabel(seconds) {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}
function updateTourUI() {
  $('play').textContent = playing ? 'Pause tour' : 'Play guided tour';
  $('timeline').value = String(progress);
  $('tour-time').textContent = `${timeLabel(progress * duration)} / ${timeLabel(duration)}`;
  $('duration').value = String(duration);
  $('duration-label').textContent = `${duration} s`;
  $('stop-count').textContent = `${stops.length} saved camera stops`;
}
function pause() { playing = false; updateTourUI(); }
function sampleTour(amount) {
  if (stops.length < 2) return;
  const point = Math.min(amount, 1) * (stops.length - 1);
  const index = Math.min(Math.floor(point), stops.length - 2);
  const t = point - index;
  const blend = t * t * (3 - 2 * t);
  const a = stops[index], b = stops[index + 1];
  camera.position.fromArray(a.position).lerp(new THREE.Vector3().fromArray(b.position), blend);
  camera.quaternion.fromArray(a.quaternion).normalize().slerp(new THREE.Quaternion().fromArray(b.quaternion).normalize(), blend);
  camera.fov = THREE.MathUtils.lerp(a.fov, b.fov, blend);
  camera.updateProjectionMatrix();
}
applyPose(opening);
updateTourUI();

const mesh = new SplatMesh({
  url: config.asset, enableLod: false,
  onProgress(event) {
    if (event.lengthComputable && event.total) {
      $('load-progress').value = event.loaded / event.total;
      $('load-detail').textContent = event.loaded < event.total
        ? `Loading original C6 model · ${Math.round(event.loaded / event.total * 100)}%`
        : 'Preparing Gaussian geometry and view-dependent appearance…';
    }
  },
});
mesh.rotation.x = Math.PI;
mesh.maxSh = 3;
scene.add(mesh);

mesh.initialized.then(async () => {
  fullCount = mesh.numSplats;
  ready = true;
  $('loading').classList.add('done');
  for (const id of ['play', 'restart', 'timeline']) $(id).disabled = false;
  $('count').textContent = `${fullCount.toLocaleString()} stored`;
  try {
    await mesh.createLodSplats();
    $('budget').disabled = false;
  } catch (error) { console.warn(error); toast('Full quality is ready. Reduced-budget preview is unavailable.'); }
}).catch(error => {
  $('load-detail').textContent = `Scene could not load: ${error.message}`;
  console.error(error);
});

$('play').onclick = () => {
  if (stops.length < 2) return toast('Add at least two camera stops to play a tour.');
  playing = !playing;
  if (progress >= 1) progress = 0;
  updateTourUI();
};
$('restart').onclick = () => { pause(); progress = 0; applyPose(opening); updateTourUI(); };
$('timeline').oninput = () => { pause(); progress = Number($('timeline').value); sampleTour(progress); updateTourUI(); };
canvas.addEventListener('pointerdown', pause);
canvas.addEventListener('wheel', pause, {passive: true});
window.addEventListener('keydown', event => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.code)) pause();
});
$('settings-toggle').onclick = () => {
  $('settings').hidden = !$('settings').hidden;
  $('settings-toggle').setAttribute('aria-expanded', String(!$('settings').hidden));
};
$('save-view').onclick = () => { pause(); opening = capturePose(); persist(); toast('Opening view saved in this browser. Export the tour for a portable copy.'); };
$('reset-view').onclick = () => { pause(); applyPose(opening); };
$('new-tour').onclick = () => { pause(); stops = [capturePose()]; progress = 0; persist(); updateTourUI(); toast('Custom tour started here. Navigate to your next view and add a stop.'); };
$('add-stop').onclick = () => { pause(); stops.push(capturePose()); persist(); updateTourUI(); toast(`Camera stop ${stops.length} saved.`); };
$('duration').oninput = () => { duration = Number($('duration').value); persist(); updateTourUI(); };
$('fov').oninput = () => { pause(); camera.fov = Number($('fov').value); camera.updateProjectionMatrix(); $('fov-label').textContent = `${camera.fov}°`; };
$('export-tour').onclick = () => {
  const blob = new Blob([JSON.stringify(record(), null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = `${sceneKey}-c6-tour.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('import-tour').onclick = () => $('tour-file').click();
$('tour-file').onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  try { restore(JSON.parse(await file.text())); pause(); progress = 0; applyPose(opening); persist(); updateTourUI(); toast('Camera tour imported.'); }
  catch (error) { toast(error.message); }
  event.target.value = '';
};
$('budget').onchange = () => {
  const fraction = Number($('budget').value) / 100;
  spark.lodSplatCount = Math.floor(fullCount * fraction);
  mesh.enableLod = fraction < 1;
  mesh.updateGenerator();
};
$('fullscreen').onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(time => {
  const delta = lastTime === null ? 0 : Math.min((time - lastTime) / 1000, 0.1);
  lastTime = time;
  if (ready && playing) {
    progress = Math.min(1, progress + delta / duration);
    sampleTour(progress);
    if (progress >= 1) playing = false;
    updateTourUI();
  } else controls.update(camera);
  renderer.render(scene, camera);
  frames++;
  if (time - lastStat >= 1000) {
    $('fps').textContent = `${(frames * 1000 / (time - lastStat)).toFixed(1)} FPS`;
    $('count').textContent = mesh.enableLod
      ? `${mesh.numSplats.toLocaleString()} selected · ${fullCount.toLocaleString()} stored`
      : `${fullCount.toLocaleString()} stored · full quality`;
    $('pose-readout').textContent = `Position ${camera.position.toArray().map(x => x.toFixed(3)).join(', ')} · Rotation ${camera.quaternion.toArray().map(x => x.toFixed(3)).join(', ')}`;
    frames = 0; lastStat = time;
  }
});
// Read-only status plus explicit camera setters for local visual verification.
window.panobudget = {camera, mesh, spark, get ready() {return ready;}, get playing() {return playing;},
  get tour() {return record();}, setProgress(value) {pause(); progress = THREE.MathUtils.clamp(value, 0, 1); sampleTour(progress); updateTourUI();},
  reset() {pause(); applyPose(opening);}, saveOpening() {opening = capturePose(); persist();}};
