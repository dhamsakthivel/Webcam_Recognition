const video = document.querySelector("#camera");
const canvas = document.querySelector("#frame");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const processedFrame = document.querySelector("#processedFrame");
const processedCtx = processedFrame.getContext("2d");
const startButton = document.querySelector("#startCamera");
const classifyButton = document.querySelector("#classify");
const addButton = document.querySelector("#addSample");
const result = document.querySelector("#result");
const labelsEl = document.querySelector("#labels");
const countEl = document.querySelector("#modelCount");
const notice = document.querySelector("#sampleNotice");
const datasetNotice = document.querySelector("#datasetNotice");
let model = [];
const profile = (...colours) => colours.map(([r, g, b]) => Array.from({ length: 9 }, () => [r, g, b]).flat());
// Baseline profiles are hand-selected color references, not photos used for training.
const presets = [
  { label: "red apple", image: "https://commons.wikimedia.org/wiki/Special:FilePath/Red%20Apple.jpg?width=640", attribution: "https://commons.wikimedia.org/wiki/File:Red_Apple.jpg", source: "Abhijit Tembhekar · CC BY 2.0", samples: profile([180, 45, 30], [205, 65, 42], [155, 42, 28]) },
  { label: "banana", image: "https://commons.wikimedia.org/wiki/Special:FilePath/Banana-Single.jpg?width=640", attribution: "https://commons.wikimedia.org/wiki/File:Banana-Single.jpg", source: "Evan-Amos · CC BY-SA 3.0", samples: profile([220, 195, 65], [200, 175, 52], [235, 212, 95]) },
  { label: "carrot", image: "https://commons.wikimedia.org/wiki/Special:FilePath/Carrots.jpg?width=640", attribution: "https://commons.wikimedia.org/wiki/File:Carrots.jpg", source: "Medved' · Public domain", samples: profile([220, 100, 32], [195, 75, 22], [235, 125, 45]) },
  { label: "broccoli", image: "https://commons.wikimedia.org/wiki/Special:FilePath/Broccoli_and_cross_section_edit.jpg?width=640", attribution: "https://commons.wikimedia.org/wiki/File:Broccoli_and_cross_section_edit.jpg", source: "Fir0002 · GFDL 1.2", samples: profile([48, 100, 38], [37, 78, 30], [67, 120, 49]) }
];

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support camera access.");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  document.querySelector("#cameraState").hidden = true;
  document.querySelector("#focusGuide").hidden = false;
  video.hidden = true;
  processedFrame.hidden = false;
  startButton.textContent = "Camera ready";
  startButton.disabled = true;
  classifyButton.disabled = false; addButton.disabled = false;
  renderFocusedPreview();
}

function drawFocusedFrame() {
  const width = 480, height = 360;
  processedFrame.width = width; processedFrame.height = height;
  processedCtx.filter = "blur(14px)";
  processedCtx.drawImage(video, 0, 0, width, height);
  processedCtx.filter = "none";
  processedCtx.save();
  processedCtx.beginPath();
  processedCtx.ellipse(width / 2, height / 2, width * .29, height * .32, 0, 0, Math.PI * 2);
  processedCtx.clip();
  processedCtx.drawImage(video, 0, 0, width, height);
  processedCtx.restore();
}

function renderFocusedPreview() {
  if (video.videoWidth) drawFocusedFrame();
  if (video.srcObject) requestAnimationFrame(renderFocusedPreview);
}

// 3×3 mean RGB cells: simple, transparent and entirely local feature extraction.
function canvasFeature() {
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const values = [];
  const cellWidth = canvas.width / 3, cellHeight = canvas.height / 3;
  for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) {
    let r = 0, g = 0, b = 0, n = 0;
    for (let y = Math.floor(row * cellHeight); y < Math.floor((row + 1) * cellHeight); y++) for (let x = Math.floor(col * cellWidth); x < Math.floor((col + 1) * cellWidth); x++) {
      const i = (y * canvas.width + x) * 4; r += pixels[i]; g += pixels[i + 1]; b += pixels[i + 2]; n++;
    }
    values.push(r / n, g / n, b / n);
  }
  return values;
}

function cameraFeature() {
  if (!video.videoWidth) throw new Error("Wait for the camera image.");
  canvas.width = 180; canvas.height = 135;
  // Measure only the guided center region, not the blurred backdrop or holder.
  ctx.drawImage(processedFrame, processedFrame.width * .21, processedFrame.height * .18, processedFrame.width * .58, processedFrame.height * .64, 0, 0, canvas.width, canvas.height);
  return canvasFeature();
}

function focusedSnapshot() {
  if (!processedFrame.width) throw new Error("Wait for the camera image.");
  return processedFrame.toDataURL("image/png");
}

function classify(feature) {
  const distances = Object.entries(model).map(([label, samples]) => {
    const closest = Math.min(...samples.map(sample => Math.sqrt(sample.reduce((sum, x, i) => sum + (x - feature[i]) ** 2, 0) / sample.length)));
    return { label, distance: closest };
  }).sort((a, b) => a.distance - b.distance);
  if (!distances.length) return null;
  const best = distances[0], next = distances[1];
  // Blend closeness and separation from competing labels into a conservative confidence.
  const closeness = Math.max(0, 1 - best.distance / 115);
  const separation = next ? Math.min(1, (next.distance - best.distance) / 45) : 0.75;
  return { ...best, confidence: Math.round(100 * (0.65 * closeness + 0.35 * separation)) };
}

async function loadModel() {
  const [info, savedModel] = await Promise.all([fetch("/api/labels").then(response => response.json()), fetch("/api/model").then(response => response.json())]);
  model = presets.reduce((labels, preset) => {
    labels[preset.label] = [...preset.samples];
    return labels;
  }, {});
  savedModel.samples.forEach(sample => {
    (model[sample.label] ??= []).push(sample.feature.split(",").map(Number));
  });
  countEl.textContent = `${presets.length + info.labels.length} label${presets.length + info.labels.length === 1 ? "" : "s"} · ${info.sampleCount} personal view${info.sampleCount === 1 ? "" : "s"}`;
  labelsEl.innerHTML = info.labels.length ? "" : '<p class="hint">No labels yet. Teach the model using the camera above.</p>';
  info.labels.forEach(label => {
    const card = document.createElement("div"); card.className = "label-card";
    card.innerHTML = `<span>${label}</span><button aria-label="Remove ${label}">Remove</button>`;
    card.querySelector("button").onclick = async () => { await fetch("/api/samples", { method: "DELETE", headers: {"Content-Type": "application/x-www-form-urlencoded"}, body: new URLSearchParams({label}) }); delete model[label]; await loadModel(); };
    labelsEl.append(card);
  });
}

function renderPresets() {
  const container = document.querySelector("#presets");
  presets.forEach(preset => {
    const card = document.createElement("article"); card.className = "preset-card";
    const image = document.createElement("img");
    image.src = preset.image; image.alt = `Reference photo of ${preset.label}`; image.loading = "lazy";
    const title = document.createElement("h3"); title.textContent = preset.label;
    const source = document.createElement("a"); source.href = preset.attribution; source.target = "_blank"; source.rel = "noopener noreferrer"; source.textContent = `${preset.source} ↗`;
    card.append(image, title, source); container.append(card);
  });
}

async function imageSeedSamples(preset) {
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.src = preset.image;
  await image.decode();
  const crops = [[0, 0, 1, 1], [.1, .1, .8, .8], [.2, .15, .6, .7]];
  return crops.map(([left, top, width, height]) => {
    canvas.width = 180; canvas.height = 135;
    ctx.drawImage(image, left * image.naturalWidth, top * image.naturalHeight, width * image.naturalWidth, height * image.naturalHeight, 0, 0, canvas.width, canvas.height);
    return canvasFeature();
  });
}

async function loadOpenSourceImageSeeds() {
  const outcomes = await Promise.allSettled(presets.map(async preset => ({ label: preset.label, samples: await imageSeedSamples(preset) })));
  let sampleCount = 0;
  outcomes.forEach(outcome => {
    if (outcome.status !== "fulfilled") return;
    const { label, samples } = outcome.value;
    model[label].push(...samples); sampleCount += samples.length;
  });
  datasetNotice.textContent = sampleCount
    ? `${sampleCount} open-source image seed samples loaded from Wikimedia Commons.`
    : "Open-source image seed samples could not be loaded; baseline profiles are still available.";
}

startButton.onclick = async () => {
  try { await requestCamera(); } catch (error) { result.textContent = `Camera unavailable: ${error.message} Use localhost and allow the browser camera prompt.`; }
};
document.querySelector("#teachForm").onsubmit = async event => {
  event.preventDefault();
  try {
    const label = document.querySelector("#label").value.trim();
    const feature = cameraFeature();
    await fetch("/api/samples", { method: "POST", headers: {"Content-Type": "application/x-www-form-urlencoded"}, body: new URLSearchParams({ label, feature: feature.join(","), snapshot: focusedSnapshot() }) }).then(async response => { if (!response.ok) throw new Error((await response.json()).error); });
    (model[label] ??= []).push(feature);
    notice.textContent = `Saved a focused reference view of ${label}. Add several more views for a stronger model.`;
    await loadModel();
  } catch (error) { notice.textContent = error.message; }
};
classifyButton.onclick = () => {
  try {
    const prediction = classify(cameraFeature());
    if (!prediction) { result.textContent = "Teach at least one produce label before scanning."; return; }
    result.innerHTML = prediction.confidence >= 70 ? `<strong>${prediction.label}</strong> · ${prediction.confidence}% confidence` : `Not confident enough (${prediction.confidence}%). Add more varied samples or improve lighting.`;
  } catch (error) { result.textContent = error.message; }
};
async function boot() {
  renderPresets();
  await loadModel();
  await loadOpenSourceImageSeeds();
}
boot();
